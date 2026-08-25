"""Regression tests for adversarial and reconstruction-only training modes."""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn


FMGAN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FMGAN_ROOT not in sys.path:
    sys.path.insert(0, FMGAN_ROOT)

from models.r3gan_1d import R3GANTrainer  # noqa: E402
from train_refiner import (  # noqa: E402
    build_arg_parser,
    build_models,
    load_verified_checkpoint,
    resolve_runtime_options,
    seed_formal_run,
    validate_checkpoint_metadata,
)
from protocol import (  # noqa: E402
    derive_paired_random_stream_seeds,
    draw_standard_normal,
    make_torch_generator,
)


class OffsetGenerator(nn.Module):
    """Tiny differentiable generator used to test loss routing."""

    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.5))

    def forward(self, X_obs, mask, X_coarse):
        refined = X_coarse + self.offset
        return X_obs * mask + refined * (1 - mask)


class TrapDiscriminator(nn.Module):
    def forward(self, _):
        raise AssertionError('discriminator was called in reconstruction-only mode')


class MeanDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x.mean(dim=(1, 2))


class NoiseRecordingGenerator(nn.Module):
    """Tiny generator that records the explicit noise supplied by the trainer."""

    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.5))
        self.seen_noise = []

    def forward(self, X_obs, mask, X_coarse, noise=None):
        if noise is None:
            raise AssertionError('paired protocol must supply explicit noise')
        self.seen_noise.append(noise.detach().clone())
        refined = X_coarse + self.offset + noise[..., :1]
        return X_obs * mask + refined * (1 - mask)


def tiny_batch():
    X_real = torch.tensor([[[0.0], [1.0], [0.0], [3.0]]])
    mask = torch.tensor([[[1.0], [0.0], [1.0], [0.0]]])
    X_obs = X_real * mask
    X_coarse = torch.zeros_like(X_real)
    return X_obs, mask, X_coarse, X_real


class TrainingModeTests(unittest.TestCase):
    def test_formal_training_path_disables_warn_only_determinism(self):
        with mock.patch('train_refiner.seed_everything') as seed_helper:
            seed_helper.return_value = {'base_seed': 123}
            record = seed_formal_run(123, True)

        seed_helper.assert_called_once_with(
            123, deterministic=True, deterministic_warn_only=False,
        )
        self.assertEqual(record, {'base_seed': 123})

    def test_reconstruction_mode_never_evaluates_discriminator(self):
        generator = OffsetGenerator()
        trainer = R3GANTrainer(
            generator, TrapDiscriminator(), lambda_recon=2.0,
            lambda_freq=0.0, adversarial=False,
        )

        loss, info = trainer.generator_step(*tiny_batch())
        loss.backward()

        self.assertIsNone(trainer.D)
        self.assertAlmostEqual(loss.item(), 3.0, places=6)
        self.assertEqual(info['g_adv'], 0.0)
        self.assertIsNotNone(generator.offset.grad)
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            trainer.discriminator_step(*tiny_batch())

    def test_adversarial_mode_retains_existing_loss_paths(self):
        discriminator = MeanDiscriminator()
        trainer = R3GANTrainer(
            OffsetGenerator(), discriminator, lambda_recon=2.0,
            lambda_freq=0.0, adversarial=True,
        )

        g_loss, g_info = trainer.generator_step(*tiny_batch())
        d_loss, d_info = trainer.discriminator_step(*tiny_batch(), gamma=0.05)

        self.assertTrue(torch.isfinite(g_loss))
        self.assertTrue(torch.isfinite(d_loss))
        self.assertGreaterEqual(discriminator.calls, 4)
        self.assertGreater(g_info['g_adv'], 0.0)
        self.assertIn('d_r1', d_info)
        self.assertIn('d_r2', d_info)

    def test_cli_defaults_to_adversarial_and_accepts_reconstruction_only(self):
        parser = build_arg_parser()
        self.assertEqual(parser.parse_args([]).training_mode, 'adversarial')
        self.assertEqual(
            parser.parse_args(['--training-mode', 'reconstruction_only']).training_mode,
            'reconstruction_only',
        )

    def test_reconstruction_model_build_does_not_instantiate_discriminator(self):
        args = build_arg_parser().parse_args([
            '--training-mode', 'reconstruction_only',
            '--width', '8',
            '--n_stages', '1',
            '--n_blocks', '1',
            '--cardinality', '1',
            '--noise_dim', '2',
            '--device', 'cpu',
        ])
        generator, discriminator = build_models(args, n_features=1, device='cpu')

        self.assertIsNone(discriminator)
        loss, _ = R3GANTrainer(
            generator, discriminator, lambda_recon=1.0,
            lambda_freq=0.0, adversarial=False,
        ).generator_step(*tiny_batch())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_checkpoint_metadata_must_match_protocol_and_mode(self):
        checkpoint = {
            'G': {},
            'protocol_manifest_sha256': 'manifest-a',
            'training_mode': 'reconstruction_only',
        }
        self.assertIs(
            validate_checkpoint_metadata(
                checkpoint, 'manifest-a', 'reconstruction_only',
            ),
            checkpoint,
        )
        with self.assertRaisesRegex(RuntimeError, 'protocol mismatch'):
            validate_checkpoint_metadata(
                checkpoint, 'manifest-b', 'reconstruction_only',
            )
        with self.assertRaisesRegex(RuntimeError, 'training-mode mismatch'):
            validate_checkpoint_metadata(
                checkpoint, 'manifest-a', 'adversarial',
            )

    def test_checkpoint_without_binding_metadata_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'lacks protocol_manifest_sha256'):
            validate_checkpoint_metadata({}, 'manifest-a', 'adversarial')
        with self.assertRaisesRegex(RuntimeError, 'lacks training_mode'):
            validate_checkpoint_metadata(
                {'protocol_manifest_sha256': 'manifest-a'},
                'manifest-a', 'adversarial',
            )

    def test_checkpoint_loader_rejects_stale_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'best_model.pt')
            torch.save({
                'G': {},
                'protocol_manifest_sha256': 'old-manifest',
                'training_mode': 'adversarial',
            }, path)
            with self.assertRaisesRegex(RuntimeError, 'protocol mismatch'):
                load_verified_checkpoint(
                    path, 'cpu', 'current-manifest', 'adversarial',
                )

    def test_legacy_programmatic_args_receive_new_runtime_defaults(self):
        resolved = resolve_runtime_options(SimpleNamespace())
        self.assertEqual(resolved, {
            'seed': 42,
            'deterministic': True,
            'num_workers': 0,
            'training_mode': 'adversarial',
        })

    def test_generator_initialization_is_identical_across_modes(self):
        parser = build_arg_parser()
        common = [
            '--width', '8', '--n_stages', '1', '--n_blocks', '1',
            '--cardinality', '2', '--noise_dim', '2', '--seed', '314',
            '--device', 'cpu',
        ]
        adversarial_args = parser.parse_args(
            common + ['--training-mode', 'adversarial'],
        )
        reconstruction_args = parser.parse_args(
            common + ['--training-mode', 'reconstruction_only'],
        )

        adversarial_g, _ = build_models(
            adversarial_args, n_features=1, device='cpu',
        )
        reconstruction_g, reconstruction_d = build_models(
            reconstruction_args, n_features=1, device='cpu',
        )

        self.assertIsNone(reconstruction_d)
        self.assertEqual(
            list(adversarial_g.state_dict()),
            list(reconstruction_g.state_dict()),
        )
        for name, value in adversarial_g.state_dict().items():
            self.assertTrue(
                torch.equal(value, reconstruction_g.state_dict()[name]), name,
            )

    def test_model_build_does_not_advance_global_torch_rng(self):
        args = build_arg_parser().parse_args([
            '--training-mode', 'adversarial', '--width', '8',
            '--n_stages', '1', '--n_blocks', '1', '--cardinality', '2',
            '--noise_dim', '2', '--seed', '2718', '--device', 'cpu',
        ])

        torch.manual_seed(999)
        expected = torch.rand(5)
        torch.manual_seed(999)
        build_models(args, n_features=1, device='cpu')
        observed = torch.rand(5)

        self.assertTrue(torch.equal(expected, observed))

    def test_discriminator_noise_consumption_cannot_advance_g_update_stream(self):
        seeds = derive_paired_random_stream_seeds(123)

        def generator_update_sequence(adversarial):
            g_stream = make_torch_generator(seeds['generator_update_noise'])
            d_stream = make_torch_generator(seeds['discriminator_noise'])
            samples = []
            for _ in range(3):
                if adversarial:
                    draw_standard_normal((1, 4, 2), d_stream)
                samples.append(draw_standard_normal((1, 4, 2), g_stream))
            return samples

        adversarial = generator_update_sequence(True)
        reconstruction = generator_update_sequence(False)
        for paired_adversarial, paired_reconstruction in zip(
                adversarial, reconstruction):
            self.assertTrue(torch.equal(
                paired_adversarial, paired_reconstruction,
            ))

    def test_trainer_routes_separate_explicit_noise_to_d_and_g_forwards(self):
        d_noise = torch.full((1, 4, 1), -0.25)
        g_noise = torch.full((1, 4, 1), 0.75)
        adversarial_g = NoiseRecordingGenerator()
        adversarial_trainer = R3GANTrainer(
            adversarial_g, MeanDiscriminator(), lambda_recon=1.0,
            lambda_freq=0.0, adversarial=True,
        )
        adversarial_trainer.discriminator_step(
            *tiny_batch(), noise=d_noise,
        )
        adversarial_trainer.generator_step(
            *tiny_batch(), noise=g_noise,
        )

        reconstruction_g = NoiseRecordingGenerator()
        reconstruction_trainer = R3GANTrainer(
            reconstruction_g, None, lambda_recon=1.0,
            lambda_freq=0.0, adversarial=False,
        )
        reconstruction_trainer.generator_step(
            *tiny_batch(), noise=g_noise,
        )

        self.assertEqual(len(adversarial_g.seen_noise), 2)
        self.assertTrue(torch.equal(adversarial_g.seen_noise[0], d_noise))
        self.assertTrue(torch.equal(adversarial_g.seen_noise[1], g_noise))
        self.assertEqual(len(reconstruction_g.seen_noise), 1)
        self.assertTrue(torch.equal(reconstruction_g.seen_noise[0], g_noise))

    def test_validation_and_test_noise_are_fixed_and_split_specific(self):
        seeds = derive_paired_random_stream_seeds(456)
        shape = (3, 4, 2)
        validation_a = draw_standard_normal(
            shape, make_torch_generator(seeds['validation_noise']),
        )
        validation_b = draw_standard_normal(
            shape, make_torch_generator(seeds['validation_noise']),
        )
        test = draw_standard_normal(
            shape, make_torch_generator(seeds['test_noise']),
        )

        self.assertTrue(torch.equal(validation_a, validation_b))
        self.assertFalse(torch.equal(validation_a, test))


if __name__ == '__main__':
    unittest.main()
