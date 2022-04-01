import argparse, time, os, warnings
import torch
from args import *
from helper import *
from dataLoader import *
from trainer import *
from model import *

def main():
    warnings.filterwarnings("ignore")
    args = get_parser()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    this_dic = vars(args)
    this_dic['device'] = device
    # define a path to save training results: save models and data
    this_dic['running_path'] = os.path.join(args.running_path, args.dataset, args.model, args.gnn_type, args.experiment) 
    if not os.path.exists(os.path.join(args.running_path, 'trained_model/')):
        os.makedirs(os.path.join(args.running_path, 'trained_model/'))
    if not os.path.exists(os.path.join(args.running_path, 'best_model/')):
        os.makedirs(os.path.join(args.running_path, 'best_model/'))
    results = createResultsFile(this_dic) # create pretty table

    # ------------------------------------load processed data-----------------------------------------------------------------------------
    this_dic['data_path'] = os.path.join(args.data_path, args.dataset, 'graphs', args.style)
    loader = get_data_loader(this_dic)
    train_loader, val_loader, test_loader, num_atom_features, num_bond_features = loader.train_loader, loader.val_loader, loader.test_loader, loader.num_features, loader.num_bond_features
    this_dic['num_atom_features'], this_dic['num_bond_features'] = int(num_atom_features), num_bond_features
    this_dic['train_size'], this_dic['val_size'], this_dic['test_size'] = len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)
    this_dic = getScaleandShift_from_scratch(this_dic, train_loader)
    
    #-----------------------------------loading model------------------------------------------------------------------------------------
    
    if this_dic['gnn_type'] == 'pnaconv': #
        this_dic['deg'], this_dic['deg_value'] = getDegreeforPNA(train_loader, 6) # TODO

    model = get_model(this_dic)
    # model weights initializations
    model = init_weights(model, this_dic)
    
    if this_dic['train_type'] == 'FT':
        if args.normalize:
            state_dict = torch.load(os.path.join(args.preTrainedPath, 'best_model', 'model_best.pt'), map_location=torch.device('cpu'))
            #state_dict.update({key:value for key,value in model.state_dict().items() if key in ['scale', 'shift']}) # scale and shift from new train loader
            #model.load_state_dict(state_dict) 
            own_state = model.state_dict()
            for name, param in state_dict.items():
                #if name.startswith('gnn'):
                own_state[name].copy_(param)
        else:
            model.from_pretrained(os.path.join(args.preTrainedPath, 'best_model', 'model_best.pt')) # load weights for encoders 

    # count total # of trainable params
    this_dic['NumParas'] = count_parameters(model)
    # save out all input parameters 
    saveConfig(this_dic, name='config.json')
    
    # ----------------------------------------training parts----------------------------------------------------------------------------
    model_ = model.to(device)
    
    optimizer = get_optimizer(args, model_)
    scheduler = build_lr_scheduler(optimizer, this_dic)
    
    best_val_error = float("inf")
    for epoch in range(this_dic['epochs']+1):
        # testing parts
        if this_dic['dataset'] == 'sol_calc': # 
            train_error = 0. # coz train set is too large to be tested every epoch
        else:
            train_error = test(model_, train_loader, this_dic) # test on entire dataset
            val_error = test(model_, val_loader, this_dic) # test on entire dataset
            test_error = test(model_, test_loader, this_dic) # test on entire dataset

        # model saving
        best_val_error = saveModel(this_dic, epoch, model_, best_val_error, val_error) # save model if validation error hits new lower 

        # training parts
        time_tic = time.time() # ending time 
        if this_dic['scheduler'] == 'const': lr = args.lr
        elif this_dic['scheduler'] in ['NoamLR', 'step']: lr = scheduler.get_lr()[0]
        else:lr = scheduler.optimizer.param_groups[0]['lr'] # decaying on val error

        loss = train(model_, optimizer, train_loader, this_dic, scheduler=scheduler) # training loss
        time_toc = time.time() # ending time 
        
        if this_dic['scheduler'] == 'decay':
            scheduler.step(val_error)
        
        # write out models and results
        contents = [epoch, round(time_toc-time_tic, 2), round(lr,7), round(train_error,6),  \
                round(val_error,6), round(test_error,6), round(param_norm(model_),2), round(grad_norm(model_),2)]
        results.add_row(contents) # updating pretty table 
        saveToResultsFile(results, this_dic, name='data.txt') # save instant data to directory

        torch.save(model_.state_dict(), os.path.join(this_dic['running_path'], 'trained_model', 'model_last.pt'))

if __name__ == "__main__":
    #cycle_index(10,2)
    main()

