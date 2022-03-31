import sys
from helper import *
from data import *
from models import *
from utils_functions import floating_type

def train(model, optimizer, dataloader, config, scheduler=None):
    '''
    Define loss and backpropagation
    '''
    model.train()
    all_loss = 0
    
    for data in dataloader:
        data = data.to(config['device'])
        optimizer.zero_grad()
        pred = model(data) # y contains different outputs depending on the # of tasks
        
        loss = get_loss_fn(config['loss'])(pred, data.mol_y)
        if config['gnn_type'] == 'dmpnn':
            all_loss += loss.item() * data.y.shape[0]
        else:
            all_loss += loss.item() * data.num_graphs

        loss.backward()
        if config['clip']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        
    if config['scheduler'] == 'NoamLR':
        scheduler.step()
    
    if config['metrics'] == 'l2':
        return np.sqrt(all_loss / len(dataloader.dataset)) # RMSE for mol level properties.
    
    if config['metrics'] == 'l1':
        return all_loss / len(dataloader.dataset) 

def test(model, dataloader, config, onData=''):
    '''
    Test model's performance
    '''
    model.eval()
    error = 0
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(config['device'])
            pred = model(data)

            if config['gnn_type'] == 'dmpnn':
                error += get_metrics_fn(config['metrics'])(y[1], data.mol_y) * data.mol_y.shape[0]
            else:
                error += get_metrics_fn(config['metrics'])(y[1], data.mol_y) * data.num_graphs
                
        if config['metrics'] == 'l2':
            return np.sqrt(error.item() / len(dataloader.dataset)) # RMSE for mol level properties.
    
        if config['metrics'] == 'l1':
           return error.item() / len(dataloader.dataset)


