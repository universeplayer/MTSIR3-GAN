import os
import re
from typing import List, Optional, Union
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
import json
import random
#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default='physioNet.pkl', required=True)
@click.option('--num_images', type=int, help='Number of images to generate', default=1000, required=True)
@click.option('--outdir', help='Where to save the output images', type=str, default='out_physioNet', required=True, metavar='DIR')
def generate_images(
    network_pkl: str,
    num_images: int,
    outdir: str
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    seeds = [random.randint(0, 1000000) for _ in range(num_images)]

    # 生成图像
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, num_images))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        label = torch.zeros([1, G.c_dim], device=device)  # 无条件生成，标签全为 0
        img = G(z, label)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_np = img[0].cpu().numpy()

        img_pil = PIL.Image.fromarray(img_np, 'RGB')

        img_resized = img_pil.resize((35, 35), PIL.Image.BICUBIC)

        img_resized_np = np.array(img_resized)

        img_list = img_resized_np.tolist()

        json_filename = f'{outdir}/seed{seed:04d}.json'
        with open(json_filename, 'w') as f:
            json.dump(img_list, f)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() 