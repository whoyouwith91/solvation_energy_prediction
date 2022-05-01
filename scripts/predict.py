from helper import *
from data_process import *
from trainer import *
from model import *
from layers import *
from gen_conf import *
from utils import *

def parse_input_arguments():
    parser = argparse.ArgumentParser(description='Physicochemical prediction')
    parser.add_argument('--SMILES', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--file_path', type=str)
    return parser.parse_args()

def main():
    args = parse_input_arguments()
    if not os.path.exists(args.file_path):
        os.makedirs(args.file_path)
        
    'Creating mol object...'
    mol = Chem.MolFromSmiles(args.SMILES)

    print('Generating conformers...')
    name = 'example_1' # you may change this name
    generator(mol, name, args.file_path)
    
    print('Generate initial features for the given molecule by using sdf and xyz file...')
    example_graph = generate_graphs(os.path.join(args.file_path, name+'.sdf'), os.path.join(file_path, name+'.xyz'))
    
    print('Saving the preprocessed graphs...')
    torch.save([example_graph], os.path.join(args.file_path, 'temp.pt'))
    
    print('Creating data loder...')
    example_dataset = GraphDataset_test(args.file_path)
    example_loader = DataLoader(example_dataset, batch_size=1, num_workers=0)

    print('Load configuration file...')
    config = loadConfig(os.path.join(model_path))
    config_all = {**config['data_config'], **config['model_config'], **config['train_config']}
    config_all['deg'] = torch.tensor([0, 5351, 348, 1867, 1565, 0]) # from function: getDegree in helper.py 
    config_all['normalize'] = True
    config_all['energy_shift'] = torch.tensor([config_all['energy_shift_value']])
    config_all['energy_scale'] = torch.tensor([config_all['energy_scale_value']])
    config_all['device'] = 'cpu'

    print('Building models...')
    model = get_model(config_all)
    state_dic = torch.load(os.path.join(model_path, 'model_best.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dic)
    model.eval()
        
    print('Evaluating...')
    with torch.no_grad():
        for data in example_loader:
            print('The predicted solvation energy for molecule {} is {}'.format(args.SMILES, model(data)))
    
    if __name__ == '__main__':
        main()


