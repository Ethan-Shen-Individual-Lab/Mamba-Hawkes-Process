import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random

random.seed(1)

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer, Mamba_pure
from transformer.mamba import Mamba
from tqdm import tqdm

from transformer.mambapy import mamba

def uni(arr):
    n = len(arr)
    for i in range(n):
        start = arr[i][0]['time_since_start']
        for j in range(len(arr[i])):
            arr[i][j]['time_since_start'] -= start
    return arr

def uni_affine(arr, noise_error):
    n = len(arr)
    for i in range(n):
        for j in range(len(arr[i])):
            arr[i][j]['time_since_start'] += noise_error
    return arr

def uni_affine1(arr, noise_error):
    n = len(arr)
    for i in range(n):
        for j in range(len(arr[i])):
            p = random.gauss(0, noise_error)
            arr[i][j]['time_since_start'] += p
    return arr




            

def prepare_dataloader(opt, affine_error, index1):
    """ Load data and prepare dataloader. """


    miss_rate = opt.miss_rate / 100
    test_rate = opt.test_rate / 100
    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')
    print('num_types', num_types)
    total_length = 0
    total_num = 0
    for val in train_data:
        total_length += len(val)
        total_num += 1
    print(total_length / total_num)

    for i in range(1):
        print(train_data[i])

    if opt.test_type == 'future':
        data_set = train_data[:] + dev_data[:] + test_data[:]
        train_data, test_data = [], []
        for i in range(len(data_set)):
            index = min(int(len(data_set[i]) * test_rate), len(data_set[i]) - 1)
            train_data.append(data_set[i][:index])
            test_data.append(data_set[i][index:])
    if opt.test_type == 'miss':
        train_data_ref = train_data[:]
        train_data = []
        for i in range(len(train_data_ref)):
            train_data.append([])
            for j in range(len(train_data_ref[i])):
                p = random.random()
                if p > miss_rate:
                    train_data[-1].append(train_data_ref[i][j])
    if opt.test_type == 'try':
        train_data = uni(train_data)
        test_data = uni(test_data)
        dev_data = uni(dev_data)
    if opt.test_type == 'affine':
        noise_error = affine_error
        train_data = uni_affine(train_data, noise_error)
        test_data = uni_affine(test_data, noise_error)
        dev_data = uni_affine(dev_data, noise_error)
    if opt.test_type == 'affine1':
        noise_error = affine_error
        train_data = uni_affine1(train_data, noise_error)
        test_data = uni_affine1(test_data, noise_error)
        dev_data = uni_affine1(dev_data, noise_error)
    if opt.test_type == "robust":
        noise_error = affine_error
        train_data = uni_affine1(train_data, noise_error)
    if opt.test_type == 'extend':
        data_set = train_data[:] + dev_data[:] + test_data[:]
        data_set1 = [(len(data_set[i]), data_set[i]) for i in range(len(data_set))]
        data_set1.sort(key=lambda element: element[0])
        train_data, test_data = [], []
        index = min(int(len(data_set) * test_rate), len(data_set) - 1)
        min_len = 9999
        max_len = 0
        for i in range(index):
            train_data.append(data_set1[i][1])
            #test_data.append(data_set1[i][1])
            if len(data_set1[i][1]) > max_len:
                max_len = len(data_set1[i][1])
            if len(data_set1[i][1]) < min_len:
                min_len = len(data_set1[i][1])
        print("train:", min_len, max_len)
        min_len = 9999
        max_len = 0
        for i in range(index, len(data_set1)):
            #train_data.append(data_set1[i][1])
            test_data.append(data_set1[i][1])
            if len(data_set1[i][1]) > max_len:
                max_len = len(data_set1[i][1])
            if len(data_set1[i][1]) < min_len:
                min_len = len(data_set1[i][1])
        print("test:", min_len, max_len)
    if opt.test_type == 'miss1':
        train_data_ref = train_data[:]
        train_data = []
        for i in range(len(train_data_ref)):
            train_data.append([])
            for j in range(len(train_data_ref[i])):
                p = random.random()
                if p > miss_rate:
                    train_data[-1].append(train_data_ref[i][j])
    if opt.test_type == 'random':
        for i in range(len(train_data)):
            for j in range(len(train_data[i])):
                train_data[i][j]['type_event'] = random.randint(0, num_types - 1)
        for i in range(len(test_data)):
            for j in range(len(test_data[i])):
                test_data[i][j]['type_event'] = random.randint(0, num_types - 1)
        for i in range(len(dev_data)):
            for j in range(len(dev_data[i])):
                dev_data[i][j]['type_event'] = random.randint(0, num_types - 1)
    if opt.test_type == 'one':
        for i in range(len(train_data)):
            for j in range(len(train_data[i])):
                train_data[i][j]['type_event'] = 0
        for i in range(len(test_data)):
            for j in range(len(test_data[i])):
                test_data[i][j]['type_event'] = 0
        for i in range(len(dev_data)):
            for j in range(len(dev_data[i])):
                dev_data[i][j]['type_event'] = 0
        num_types = 1
    if opt.test_type == 'length_test':
        L = opt.length_test_type
        index = index1
        traindata_set1 = [(len(train_data[i]), train_data[i]) for i in range(len(train_data))]
        traindata_set1.sort(key=lambda element: element[0])
        testdata_set1 = [(len(test_data[i]), test_data[i]) for i in range(len(test_data))]
        testdata_set1.sort(key=lambda element: element[0])
        train_data, test_data = [], []
        n,m = len(traindata_set1), len(testdata_set1)
        for i in range(int(n*index/L), int((index+1)*n/L)):
            train_data.append(traindata_set1[i][1])
        for i in range(int(m*index/L), int((index+1)*m/L)):
            test_data.append(testdata_set1[i][1])


    """
    for arr in train_data:
        for dic in arr:
            p = random.random()
            #print(p)
            #num_pre[dic['type_event']] += 1
            if p < miss_rate:
                dic['type_event'] = random.randint(0, num_types - 1)
            #num_after[dic['type_event']] += 1
    """

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=False)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time)

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type, opt.model_type)
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = Utils.time_loss(prediction[1], event_time)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 10000
        scale_event_loss = 1.5
        loss = event_loss + pred_loss / scale_event_loss + se / scale_time_loss
        #print(event_loss, pred_loss, se)
        #loss = pred_loss
        #loss = 1000*se
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


def eval_epoch(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

            """ forward """
            enc_out, prediction = model(event_type, event_time)

            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type, opt.model_type)
            event_loss = -torch.sum(event_ll - non_event_ll)
            _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            se = Utils.RMSE_loss(prediction[1], event_time)

            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt, log_file):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        #for name,parameters in model.named_parameters():
        #    print(name,':',parameters.size())
        
        embed = torch.abs(model.encoder.event_emb.weight)
        #print(torch.sum(embed))

        #print(torch.sum(embed, -1))
        



        start = time.time()
        valid_event, valid_type, valid_time = eval_epoch(model, validation_data, pred_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        print('  - [Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
              .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))

        # logging
        with open(log_file, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time))

        scheduler.step()
    with open(log_file, 'a') as f:
        f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
            .format(epoch=100, ll=max(valid_event_losses), acc=max(valid_pred_losses), rmse=min(valid_rmse)))


def main():
    """ Main function. """

    torch.manual_seed(0)

    parser = argparse.ArgumentParser()

    dataset = 'Hawkes'
    test_type = 'length_test'
    model_type = 'Mamba_pure'
    affine_error1 = 1e-2
    length_test_type = 5

    dic = {
        'FI':{
            'file':'D:\\Transformer-Hawkes-Process\\NeuralHawkesData\\data_bookorder\\fold2\\',
            'batchsize':1,
            'dmodel':128,
            'dinner':2048,
            'dk':64,
            'dv':64,
            'head':6,
            'layers':6,
            'lr':1e-4
        },
        'SO':{
            'file':'D:\\Transformer-Hawkes-Process\\NeuralHawkesData\\data_so\\fold3\\',
            'batchsize':4,
            'dmodel':512,
            'dinner':1024,
            'dk':512,
            'dv':512,
            'head':4,
            'layers':4,
            'lr':1e-4
        },
        'Hawkes':{
            'file':'D:\\Transformer-Hawkes-Process\\NeuralHawkesData\\data_hawkes\\',
            'batchsize':4,
            'dmodel':64,
            'dinner':256,
            'dk':16,
            'dv':16,
            'head':3,
            'layers':3,
            'lr':1e-4
        },
        'Hawkes_tmp':{
            'file':'D:\\Transformer-Hawkes-Process\\NeuralHawkesData\\data_hawkes\\',
            'batchsize':4,
            'dmodel':16,
            'dinner':32,
            'dk':4,
            'dv':4,
            'head':2,
            'layers':2,
            'lr':1e-4
        },
        'Ret':{
            'file':'D:\\Transformer-Hawkes-Process\\NeuralHawkesData\\data_retweet\\',
            'batchsize':16,
            'dmodel':64,
            'dinner':256,
            'dk':16,
            'dv':16,
            'head':3,
            'layers':3,
            'lr':1e-2
        },
        'Meme':{
            'file':'D:\\Transformer-Hawkes-Process\\NeuralHawkesData\\data_meme\\',
            'batchsize':128,
            'dmodel':64,
            'dinner':256,
            'dk':16,
            'dv':16,
            'head':3,
            'layers':3,
            'lr':1e-3
        },
        'Mimic':{
            'file':'D:\\Transformer-Hawkes-Process\\NeuralHawkesData\\data_mimic\\fold1\\',
            'batchsize':1,
            'dmodel':64,
            'dinner':256,
            'dk':16,
            'dv':16,
            'head':3,
            'layers':3,
            'lr':1e-4
        }
    }

    parser.add_argument('-data', default=dic[dataset]['file'])

    parser.add_argument('-epoch', type=int, default=40)
    parser.add_argument('-batch_size', type=int, default=dic[dataset]['batchsize'])

    parser.add_argument('-d_model', type=int, default=dic[dataset]['dmodel'])
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=dic[dataset]['dinner'])
    parser.add_argument('-d_k', type=int, default=dic[dataset]['dk'])
    parser.add_argument('-d_v', type=int, default=dic[dataset]['dv'])

    parser.add_argument('-n_head', type=int, default=dic[dataset]['head'])
    parser.add_argument('-n_layers', type=int, default=dic[dataset]['layers'])

    parser.add_argument('-dropout', type=float, default=0.1)
    if model_type == "Mamba" and (dataset == "Mimic"):
        parser.add_argument('-lr', type=float, default=dic[dataset]['lr'] * 5)
    elif model_type == "Mamba" and (dataset == "SO" or dataset == "FI"):
        parser.add_argument('-lr', type=float, default=dic[dataset]['lr'] * 1)
    else:
        parser.add_argument('-lr', type=float, default=dic[dataset]['lr'])

    parser.add_argument('-smooth', type=float, default=0)

    #test_type = 'OOD'
    #model_type = 'RoPE_linear'
    miss_rate = 0
    test_rate = 70
    parser.add_argument('-test_type', type=str, default=test_type)  # future means we train on pre dataset and predict the future case, OOD means we train on trainset and predict the testset, miss means we throw some samples
    parser.add_argument('-model_type', type=str, default=model_type)  # RoPE & Pre
    parser.add_argument('-miss_rate', type=int, default=miss_rate)  
    parser.add_argument('-test_rate', type=int, default=test_rate)  
    parser.add_argument('-length_test_type', type=int, default=length_test_type)  
    for i in range(length_test_type):
        #affine_error1 = 2
        #parser.add_argument('-test_type', type=str, default=test_type)  # future means we train on pre dataset and predict the future case, OOD means we train on trainset and predict the testset, miss means we throw some samples
        #parser.add_argument('-model_type', type=str, default=model_type)  # RoPE & Pre
       # parser.add_argument('-miss_rate', type=int, default=miss_rate)  
        #parser.add_argument('-test_rate', type=int, default=test_rate)  
        #parser.add_argument('-affine_error', type=float, default=affine_error) 

        #parser.add_argument('-log', type=str, default='log_Fi0_' + test_type + '_' + model_type + '_miss' + str(miss_rate) + '_test' + str(test_rate) + '_error' + str(int(affine_error * 1000)) + '.txt')
        
        #log_file = 'log_'+ dataset + '_encoder_delta_test11' + test_type + '_' + model_type + '_miss' + str(miss_rate) + '_test' + str(test_rate) + '_error' + str(int(affine_error1 * 10000)) + '.txt'
        log_file = 'log_'+ dataset + '_newRMSE_loss10_' + test_type + '_' + str(i) + '_' + str(length_test_type) + '_' + model_type + '_miss' + str(miss_rate) + '_test' + str(test_rate) + '_error' + str(int(affine_error1 * 10000)) + '.txt'
        opt = parser.parse_args()

        # default device is CUDA
        opt.device = torch.device('cuda')

        # setup the log file
        with open(log_file, 'w') as f:
            f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

        print('[Info] parameters: {}'.format(opt))

        """ prepare dataloader """
        #trainloader, testloader, num_types = prepare_dataloader(opt, affine_error1)
        trainloader, testloader, num_types = prepare_dataloader(opt, affine_error1, i)

        """ prepare model """
        if model_type == 'Mamba_pure':
            config = mamba.MambaConfig(d_model=opt.d_model, n_layers=4)
            model = Mamba_pure(
            config=config,
            num_types=num_types,
            d_model=opt.d_model,
            d_rnn=opt.d_rnn,
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            d_k=opt.d_k,
            d_v=opt.d_v,
            dropout=opt.dropout,
            model_type=opt.model_type
        )
        #model.to(opt.device)
        else:
            config = mamba.MambaConfig(d_model=opt.d_model, n_layers=1)
            model = Transformer(
            config=config,
            num_types=num_types,
            d_model=opt.d_model,
            d_rnn=opt.d_rnn,
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            d_k=opt.d_k,
            d_v=opt.d_v,
            dropout=opt.dropout,
            model_type=opt.model_type
        )
        model.to(opt.device)

        """ optimizer and scheduler """
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                            opt.lr, betas=(0.9, 0.999), eps=1e-05)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

        """ prediction loss function, either cross entropy or label smoothing """
        if opt.smooth > 0:
            pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
        else:
            pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

        """ number of parameters """
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('[Info] Number of parameters: {}'.format(num_params))

        """ train the model """
        train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt, log_file)


if __name__ == '__main__':
    main()
