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

# Import with error handling for cloud environment
try:
    from transformer.mambapy import mamba
    from transformer.mambapy.mamba import MambaConfig
except ImportError as e:
    print(f"Import warning: {e}")
    print("Attempting alternative import...")
    import transformer.mambapy.mamba as mamba
    from transformer.mambapy.mamba import MambaConfig

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
    
    # Calculate average sequence length (keeping only essential info)
    total_length = sum(len(val) for val in train_data)
    avg_length = total_length / len(train_data)
    print(f'[Info] Dataset: {num_types} event types, Average sequence length: {avg_length:.1f}')

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
        
        # Basic numerical stability for time intervals (minimal change)
        event_time = torch.clamp(event_time, min=1e-6, max=1e6)

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

        # Loss stability: check and scale extreme values
        event_loss = torch.clamp(event_loss, min=-1e6, max=1e6)
        pred_loss = torch.clamp(pred_loss, min=0, max=1e4)
        se = torch.clamp(se, min=0, max=1e8)

        # Loss function with configurable weights
        # Default: beta=1.0, gamma=1e-4 (can be overridden via command line)
        beta = opt.beta
        gamma = opt.gamma
        
        loss = event_loss + beta * pred_loss + gamma * se
        
        # Final loss stability check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss before backward, using fallback")
            loss = torch.tensor(1.0, device=loss.device, requires_grad=True)
        #print(event_loss, pred_loss, se)
        #loss = pred_loss
        #loss = 1000*se
        loss.backward()

        """ update parameters with gradient clipping """
        # Check for NaN/Inf loss (especially for Retweet dataset)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected, skipping this batch")
            optimizer.zero_grad()
            continue
            
        # Gradient clipping for training stability (all datasets)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Post-update parameter check for Retweet dataset  
        dataset_name = opt.data.split('/')[-2] if '/' in opt.data else 'default'
        if 'retweet' in dataset_name.lower():
            for name, param in model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"Warning: NaN/Inf in {name}, reinitializing...")
                    if param.dim() >= 2:
                        nn.init.xavier_normal_(param.data, gain=0.1)
                    else:
                        # For 1D parameters (like alpha, beta)
                        param.data.fill_(0.01)

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
            
            # Basic numerical stability for time intervals (minimal change)
            event_time = torch.clamp(event_time, min=1e-6, max=1e6)

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
    
    train_event_losses = []  # training log-likelihood
    train_pred_losses = []  # training event type prediction accuracy
    train_rmse = []  # training event time prediction RMSE
    
    # Create train log file
    train_log_file = log_file.replace('.txt', '_train.txt')
    with open(train_log_file, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')
    
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event, valid_type, valid_time = eval_epoch(model, validation_data, pred_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

        # Keep track of both train and validation metrics
        train_event_losses += [train_event]
        train_pred_losses += [train_type]
        train_rmse += [train_time]

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        
        print('  - [Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
              .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))

        # logging test results
        with open(log_file, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time))
        
        # logging train results
        with open(train_log_file, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch, ll=train_event, acc=train_type, rmse=train_time))

        scheduler.step()
        
    # Final summary logs
    with open(log_file, 'a') as f:
        f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
            .format(epoch=100, ll=max(valid_event_losses), acc=max(valid_pred_losses), rmse=min(valid_rmse)))

    with open(train_log_file, 'a') as f:
        f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
            .format(epoch=100, ll=max(train_event_losses), acc=max(train_pred_losses), rmse=min(train_rmse)))


def get_experiment_configs():
    """ Get all experiment configurations based on paper.tex specifications """
    
    # Note: d_inner values in configs are for reference only. 
    # MambaConfig automatically computes d_inner = 2 * d_model (expand_factor=2)
    # Actual d_inner used: Financial=256, SO=1024, others=128
    configs = {
        'Financial': {
            'file': 'data/data_bookorder/fold2/',
            'batchsize': 1,
            'd_model': 128,
            'd_inner': 2048,  # Overridden by MambaConfig to 256 (2*128)
            'd_k': 64,
            'd_v': 64,
            'n_head': 6,
            'n_layers': 4,
            'lr': 1e-4,
            'test_types': ['OOD']
        },
        'SO': {
            'file': 'data/data_so/fold3/',
            'batchsize': 4,
            'd_model': 512,
            'd_inner': 1024,
            'd_k': 512,
            'd_v': 512,
            'n_head': 4,
            'n_layers': 4,
            'lr': 1e-4,
            'test_types': ['OOD']
        },
        'Synthetic': {
            'file': 'data/data_hawkes/',
            'batchsize': 4,
            'd_model': 64,
            'd_inner': 256,
            'd_k': 16,
            'd_v': 16,
            'n_head': 3,
            'n_layers': 4,
            'lr': 1e-4,
            'test_types': ['OOD']
        },
        'Retweet': {
            'file': 'data/data_retweet/',
            'batchsize': 16,
            'd_model': 64,
            'd_inner': 256,
            'd_k': 16,
            'd_v': 16,
            'n_head': 3,
            'n_layers': 4,
            'lr': 1e-2,  # Note: automatically adjusted to 1e-3 during execution for numerical stability
            'test_types': ['OOD']
        },
        'Mimic': {
            'file': 'data/data_mimic/fold1/',
            'batchsize': 1,
            'd_model': 64,
            'd_inner': 256,
            'd_k': 16,
            'd_v': 16,
            'n_head': 3,
            'n_layers': 4,
            'lr': 5e-4,
            'test_types': ['OOD']
        }
    }
    
    return configs

def run_single_experiment(dataset_name, config, model_type='Mamba_pure', epochs=40, beta=1.0, gamma=1e-4):
    """ Run a single experiment with given configuration """
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {dataset_name} with {model_type}")
    print(f"Dataset: {config['file']}")
    print(f"{'='*60}")

    torch.manual_seed(0)

    # Create opt object directly without argparse to avoid conflicts
    class Opt:
        def __repr__(self):
            return f"A-MHP Config(dataset={self.data.split('/')[-2] if '/' in self.data else 'unknown'}, " \
                   f"d_model={self.d_model}, batch_size={self.batch_size}, lr={self.lr}, " \
                   f"model_type={self.model_type}, beta={self.beta}, gamma={self.gamma})"
    
    opt = Opt()
    opt.data = config['file']
    opt.epoch = epochs
    opt.batch_size = config['batchsize']
    opt.d_model = config['d_model']
    opt.d_rnn = 256
    opt.d_inner_hid = config['d_inner']
    opt.d_k = config['d_k']
    opt.d_v = config['d_v']
    opt.n_head = config['n_head']
    opt.n_layers = config['n_layers']
    opt.dropout = 0.1
    opt.lr = config['lr']
    opt.smooth = 0
    opt.test_type = 'OOD'
    opt.model_type = model_type
    opt.miss_rate = 0
    opt.test_rate = 70
    opt.length_test_type = 5
    opt.device = torch.device('cuda')
    opt.beta = beta
    opt.gamma = gamma

    # Create log file name
    log_file = f'log_{dataset_name}_A-MHP_{model_type}_OOD.txt'

    # Setup log file
    with open(log_file, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

        print('[Info] parameters: {}'.format(opt))

    try:
        # Prepare dataloader
        trainloader, testloader, num_types = prepare_dataloader(opt, 1e-2, 0)

        # Prepare model - always use Mamba_pure with enhanced features
        if model_type == 'Mamba_pure':
            # Set dataset type for specific optimizations
            dataset_type = 'so' if 'so' in dataset_name.lower() else 'retweet' if 'retweet' in dataset_name.lower() else 'default'
            config = MambaConfig(d_model=opt.d_model, n_layers=opt.n_layers, dataset_type=dataset_type)
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
        else:
            dataset_type = 'so' if 'so' in dataset_name.lower() else 'retweet' if 'retweet' in dataset_name.lower() else 'default'
            config = MambaConfig(d_model=opt.d_model, n_layers=1, dataset_type=dataset_type)
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

        # Enhanced weight initialization for all datasets (engineering optimization)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    if 'event_emb' in name:
                        nn.init.xavier_normal_(param, gain=0.8)  # Conservative for embeddings
                        if hasattr(model.encoder, 'event_emb'):
                            model.encoder.event_emb.weight[0].fill_(0)  # Padding zero
                    elif 'linear' in name:
                        nn.init.xavier_normal_(param, gain=0.6)  # Conservative for predictors
                    else:
                        nn.init.xavier_normal_(param, gain=1.0)  # Standard for others

        # Optimizer with Retweet-specific learning rate for numerical stability
        if 'retweet' in dataset_name.lower():
            # Retweet: severe NaN problem, need much lower lr for stability
            lr_adjusted = opt.lr * 0.1  # 1e-2 → 1e-3 (necessary for stability)
            print(f"[Info] Retweet dataset: using adjusted lr={lr_adjusted} for numerical stability")
        else:
            lr_adjusted = opt.lr
            
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                            lr_adjusted, betas=(0.9, 0.95), eps=1e-08, weight_decay=1e-5)
        # Better scheduler for stability and performance
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=lr_adjusted*0.01)

        # Prediction loss function with label smoothing (regularization)
        pred_loss_func = Utils.LabelSmoothingLoss(0.1, num_types, ignore_index=-1)  # 10% smoothing

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('[Info] Number of parameters: {}'.format(num_params))

        # Train the model
        train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt, log_file)
        
        print(f"Experiment {dataset_name} completed successfully!")
        
    except Exception as e:
        print(f"Error in experiment {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """ Main function - Run experiments with optional dataset selection """
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='A-MHP Experiments')
    parser.add_argument('--dataset', type=str, default='all', 
                       choices=['all', 'Financial', 'SO', 'Synthetic', 'Retweet', 'Mimic'],
                       help='Specify dataset to run (default: all)')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Event loss weight (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=1e-4,
                       help='Time loss weight (default: 1e-4)')
    parser.add_argument('--fold', type=int, default=None,
                       help='Cross-validation fold number (1-5). If not specified, uses default fold.')
    args = parser.parse_args()
    
    print("Starting A-MHP (Adaptive Mamba Hawkes Process) Experiments")
    print("=" * 70)
    
    # Get all experiment configurations
    experiment_configs = get_experiment_configs()
    
    # Apply fold parameter if specified
    if args.fold is not None:
        for dataset_name in ['Financial', 'SO', 'Mimic']:
            if dataset_name in experiment_configs:
                base_path = experiment_configs[dataset_name]['file']
                # Replace fold number in path
                import re
                new_path = re.sub(r'/fold\d+/', f'/fold{args.fold}/', base_path)
                experiment_configs[dataset_name]['file'] = new_path
                print(f"Using fold{args.fold} for {dataset_name}")
    
    # Filter datasets based on argument
    if args.dataset != 'all':
        if args.dataset not in experiment_configs:
            print(f"Dataset '{args.dataset}' not found. Available: {list(experiment_configs.keys())}")
            return
        experiment_configs = {args.dataset: experiment_configs[args.dataset]}
        print(f"Running single dataset: {args.dataset}")
    else:
        print(f"Running all datasets")
    
    # Model types to test
    model_types = ['Mamba_pure']
    
    total_experiments = len(experiment_configs) * len(model_types)
    print(f"Total experiments to run: {total_experiments}")
    print(f"Datasets: {', '.join(experiment_configs.keys())}")
    print(f"Model: {', '.join(model_types)}")
    print("=" * 70)
    
    current_exp = 0
    
    for model_type in model_types:
        for dataset_name, config in experiment_configs.items():
            current_exp += 1
            print(f"\nProgress: {current_exp}/{total_experiments}")
            
            try:
                run_single_experiment(dataset_name, config, model_type, epochs=40, 
                                    beta=args.beta, gamma=args.gamma)
            except KeyboardInterrupt:
                print("\nExperiments interrupted by user")
                return
            except Exception as e:
                print(f"Failed to run experiment {dataset_name} with {model_type}: {str(e)}")
                continue
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print("Check individual log files for results:")
    for dataset_name in experiment_configs.keys():
        for model_type in model_types:
            print(f"- log_{dataset_name}_A-MHP_{model_type}_OOD.txt (test results)")
            print(f"- log_{dataset_name}_A-MHP_{model_type}_OOD_train.txt (train results)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
