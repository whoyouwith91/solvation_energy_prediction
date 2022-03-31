import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ELU, Tanh, SELU
from prettytable import PrettyTable
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
#from torchcontrib.optim import SWA
import sklearn.metrics as metrics

################### Configuration setting names ############################
data_config = ['dataset', 'model', 'style', 'data_path']
model_config = ['dataset', 'model', 'gnn_type',  'bn', 'batch_size', 'emb_dim', 'act_fn' , 'weights', 'num_atom_features', 'num_tasks', 'propertyLevel', \
         'test_level', 'num_bond_features', 'pooling', 'NumParas', 'num_layer', 'fully_connected_layer_sizes', 'aggregate', \
             'residual_connect', 'drop_ratio', 'energy_shift_value', 'energy_scale_value', 'deg_value', 'normalize']
train_config = ['running_path', 'seed', 'optimizer', 'loss', 'metrics', 'lr', 'lr_style', \
         'epochs', 'early_stopping', 'train_type', 'taskType', 'train_size', 'val_size', 'test_size', 'sample', 'data_seed', \
         'preTrainedPath', 'fix_test', 'vary_train_only', 'sample_size']
###############################################################################

def set_seed(seed):
# define seeds for training 
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_optimizer(args, model):
    # define optimizers
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    if args.optimizer == 'adamW':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.optimizer == 'SWA':
        base_opt = torch.optim.SGD(model.parameters(), lr=args.lr)
        optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.0005)
    return optimizer


param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_loss_fn(name):
    if name == 'l1':
        return F.l1_loss
    if name == 'l2':
        return F.mse_loss

def get_metrics_fn(name):
    if name == 'l1':
        return F.l1_loss
    if name == 'l2':
        return F.mse_loss

def activation_func(config):
    name = config['act_fn']
    if name == 'relu':
       return ReLU()
    if name == 'elu':
       return ELU()
    if name == 'leaky_relu':
       return LeakyReLU()
    if name == 'tanh':
       return Tanh()
    if name == 'selu':
       return SELU()

def he_norm(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
       torch.nn.init.kaiming_normal_(m.weight.data)

def xavier_norm(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
       torch.nn.init.xavier_normal_(m.weight.data)

def he_uniform(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
       torch.nn.init.kaiming_uniform_(m.weight.data)

def xavier_uniform(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
       torch.nn.init.xavier_uniform_(m.weight.data)

def init_weights(model, config):
    name = config['weights']
    if name == 'he_norm':
       model.apply(he_norm)
    if name == 'xavier_norm':
       model.apply(xavier_norm)
    if name == 'he_uni':
       model.apply(he_uniform)
    if name == 'xavier_uni':
       model.apply(xavier_uniform)
    return model

def createResultsFile(this_dic):
    ## create pretty table
    if this_dic['loss'] == 'l2':
        train_header = 'RMSE'
    if this_dic['loss'] == 'l1':
        train_header = 'MAE'

    if this_dic['metrics'] == 'l2':
        test_header = 'RMSE'
    if this_dic['metrics'] == 'l1':
        test_header = 'MAE'
    
    header = ['Epoch', 'Time', 'LR', 'Train {}'.format(train_header), 'Valid {}'.format(test_header), 'Test {}'.format(test_header), 'PNorm', 'GNorm']
    x = PrettyTable(header)
    return x 

def saveToResultsFile(table, this_dic, name='data.txt'):
    with open(os.path.join(this_dic['running_path'], 'data.txt'), 'w') as f1:
        f1.write(str(table))
    f1.close()

def saveConfig(this_dic, name='config.json'):
    all_ = {'data_config': {key:this_dic[key] for key in data_config if key in this_dic.keys()},
            'model_config':{key:this_dic[key] for key in model_config if key in this_dic.keys()},
            'train_config': {key:this_dic[key] for key in train_config if key in this_dic.keys()}}
    
    with open(os.path.join(this_dic['running_path'], name), 'w') as f:
        json.dump(all_, f, indent=2)

def saveModel(config, epoch, model, bestValError, valError):
    if config['early_stopping']:
        if bestValError > valError:
            patience = 0
            bestValError = valError
        else:
            patience += 1
            if patience > config['patience_epochs']:
                torch.save(model.state_dict(), os.path.join(config['running_path'], 'best_model'))
    else:
        if bestValError > valError:
            bestValError = valError
            #logging.info('Saving models...')
            torch.save(model.state_dict(), os.path.join(config['running_path'], 'best_model', 'model_best.pt'))
            
    return bestValError


def build_lr_scheduler(optimizer, config):
    """
    Builds a PyTorch learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing learning rate arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """
    if config['scheduler'] == 'NoamLR':
        # Learning rate scheduler
        return NoamLR(
            optimizer=optimizer,
            warmup_epochs=[config['warmup_epochs']],
            total_epochs=[config['epochs']],
            steps_per_epoch=config['train_size'] // config['batch_size'],
            init_lr=[config['init_lr']],
            max_lr=[config['max_lr']],
            final_lr=[config['final_lr']]
        )
    elif config['scheduler'] == 'decay':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                    factor=config['decay_factor'], 
                    patience=config['patience_epochs'], 
                    min_lr=0.00001)
    
    elif config['scheduler'] == 'step':
        # gamma = 0.95 is only for naive trial for aleatoric uncertainty training
        return torch.optim.lr_scheduler.StepLR(optimizer, 
                    step_size=10, gamma=0.95) 
    else:
        return None

def getDegreeforPNA(loader, degree_num):
    from torch_geometric.utils import degree
    
    deg = torch.zeros(degree_num, dtype=torch.long)
    for data in loader:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg, str(list(deg.numpy()))

def getScaleandShift_from_scratch(config, loader):
    # loader: train_loader 
    train_values = []
    train_N = []
    for data in loader:
        train_values.extend(list(data.mol_y.numpy()))
        train_N.extend(list(data.N.numpy()))
        
    shift, scale = atom_mean_std(train_values, train_N, range(len(train_values)))
    config['energy_shift'], config['energy_shift_value'] = torch.tensor([shift]), shift
    config['energy_scale'], config['energy_scale_value'] = torch.tensor([scale]), scale
    return config 

def atom_mean_std(E, N, index):
    """
    calculate the mean and stand variance of Energy in the training set
    :return:
    """
    mean = 0.0
    std = 0.0
    num = len(index)
    for _i in range(num):
        i = index[_i]
        m_prev = mean
        x = E[i] / N[i]
        mean += (x - mean) / (i + 1)
        std += (x - mean) * (x - m_prev)
    std = math.sqrt(std / num)
    return mean, std