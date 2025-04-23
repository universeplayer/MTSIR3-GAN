import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import utils
import models
import argparse
import data_loader
import pandas as pd
from math import sqrt
from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--model', type=str, default="Based_on_BRITS")   # gru_d, brits
parser.add_argument('--hid_size', type=int)
parser.add_argument('--impute_weight', type=float)
parser.add_argument('--label_weight', type=float)
args = parser.parse_args()

choose = 0
missing_rate = 50
dataset = 'AirQuality'
dimension = 36	

def train(Generator, Discriminator, Classifier):
    cuda = True if torch.cuda.is_available() else False
    optimizer_G = optim.Adam(Generator.parameters(), lr=1e-3)
    optimizer_D = optim.Adam(Discriminator.parameters(), lr=1e-3)
    optimizer_C = optim.Adam(Classifier.parameters(), lr=1e-3)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # data_iter = data_loader.get_loader(batch_size=args.batch_size)
    data_iter = data_loader.get_train_loader(batch_size=args.batch_size)
    Imputed_Result = [['Epoch', 'RMSE', 'MRE', 'MAE']]

    for epoch_pre in range(5):
        C_run_loss = 0
        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)
            # Update Classifier with Real Data
            ret_C = Classifier.run_on_batch(data)
            optimizer_C.zero_grad()
            ret_C['loss'].backward(retain_graph=True)
            optimizer_C.step()
            C_run_loss += ret_C['loss'].item()
            print('\r Pretraining Progress epoch {}, Classifier loss {}'.format(
                epoch_pre, C_run_loss / (idx * 2 + 2.0)))

    for epoch in range(args.epochs):
        Generator.train()
        G_run_loss = 0.0
        D_run_loss = 0.0
        C_run_loss = 0.0
        result = [epoch]
        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)
            valid = Variable(Tensor(len(data['labels']), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(len(data['labels']), 1).fill_(0.0), requires_grad=False)
            D_data_R = data
            D_data_F = data
            D_data_F['labels'] = fake
            Classifier_FakeData = data

            # Update Classifier with Real Data
            ret_C = Classifier.run_on_batch(data)
            optimizer_C.zero_grad()
            ret_C['loss'].backward(retain_graph=True)
            optimizer_C.step()
            C_run_loss += ret_C['loss'].item()

            # Update Generator
            ret_G = Generator.run_on_batch(data)
            optimizer_G.zero_grad()
            imputation_data = ret_G['imputations']
            D_data_F['forward']['values'] = imputation_data
            D_data_F['forward']['forwards'] = imputation_data
            D_data_F['backward']['values'] = imputation_data
            D_data_F['backward']['forwards'] = imputation_data
            Classifier_FakeData['forward']['values'] = imputation_data
            Classifier_FakeData['forward']['forwards'] = imputation_data
            Classifier_FakeData['backward']['values'] = imputation_data
            Classifier_FakeData['backward']['forwards'] = imputation_data
            D_loss_Fake = Discriminator.run_on_batch(D_data_F)
            C_loss_Fake = Classifier.run_on_batch(Classifier_FakeData)
            G_loss = ret_G['loss'] - D_loss_Fake['g_d_loss'] + 3 * C_loss_Fake['g_c_loss']
            G_loss.backward(retain_graph=True)
            optimizer_G.step()
            G_run_loss += ret_G['loss'].item()

            # Update Classifier with Fake Data
            ret_C = Classifier.run_on_batch(Classifier_FakeData)
            optimizer_C.zero_grad()
            ret_C['loss'].backward(retain_graph=True)
            optimizer_C.step()
            C_run_loss += ret_C['loss'].item()

            # Update Discriminator
            D_data_R['labels'] = valid
            for i in range(5):
                ret_D_R = Discriminator.run_on_batch(D_data_R)
                ret_D_F = Discriminator.run_on_batch(D_data_F)
                optimizer_D.zero_grad()
                D_loss = ret_D_R['loss'] + ret_D_F['loss']
                D_loss.backward(retain_graph=True)
                optimizer_D.step()
                D_run_loss += D_loss.item()

            print('\r Progress epoch {}, {:.2f}%, Generator loss {}, Discriminator loss {}, Classifier loss {}'.format(
                epoch, (idx + 1) * 100.0 / len(data_iter),
                G_run_loss / (idx + 1.0), D_run_loss / (idx + 1.0), C_run_loss / (idx * 2 + 2.0)))

        test_data_iter = data_loader.get_test_loader(
            batch_size=args.batch_size)
        RMSE, MRE, MAE = evaluate(Generator, test_data_iter)
        result.append(RMSE)
        result.append(MRE)
        result.append(MAE)
        Imputed_Result.append(result)
    df = pd.DataFrame(Imputed_Result)
    df.to_csv(dataset+'_Imputed_Result.csv', index=False, header=False)

def evaluate(model, val_iter):
    model.eval()
    labels = []
    preds = []
    evals = []
    imputations = []
    save_impute = []
    save_label = []

    for idx, data in enumerate(val_iter):

        data = utils.to_var(data)
        ret = model.run_on_batch(data)

        save_impute.append(ret['imputations'].data.cpu().numpy())
        save_label.append(ret['labels'].data.cpu().numpy())

        pred = ret['predictions'].data.cpu().numpy()
        label = ret['labels'].data.cpu().numpy()
        is_train = ret['is_train'].data.cpu().numpy()

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

        pred = pred[np.where(is_train == 0)]
        label = label[np.where(is_train == 0)]

        labels += label.tolist()
        preds += pred.tolist()

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    print('MAE', np.abs(evals - imputations).mean())
    print('MRE', np.abs(evals - imputations).sum() / np.abs(evals).sum())
    print('RMSE', sqrt(metrics.mean_squared_error(evals, imputations)))
    RMSE = sqrt(metrics.mean_squared_error(evals, imputations))
    MRE = np.abs(evals - imputations).sum() / np.abs(evals).sum()
    MAE = np.abs(evals - imputations).mean()

    return RMSE, MRE, MAE


def run():
    Generator = getattr(models,
                    args.model).Generator(args.hid_size, args.impute_weight,
                                          args.label_weight)
    Discriminator = getattr(models,
                        'discriminator').Discriminator(args.hid_size, args.impute_weight,
                                              args.label_weight)
    Classifier = getattr(models,
                            'classifier').Classifier(args.hid_size, args.impute_weight,
                                                           args.label_weight)

    if torch.cuda.is_available():
        Generator = Generator.cuda()
        Discriminator = Discriminator.cuda()
        Classifier = Classifier.cuda()

    train(Generator, Discriminator, Classifier)


if __name__ == '__main__':
    run()
