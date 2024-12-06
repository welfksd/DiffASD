import torch
import os
import numpy as np
from omegaconf import OmegaConf
from train import train
from test_gmmanno import test

os.environ['CUDA_VISIBLE_DEVICES'] = "2, 4"

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(prog='DiffASD', description='用于异音检测')
    parser.add_argument('-cfg', '--config', 
                                default= os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.yaml'), 
                                help='config file')
    parser.add_argument('--train', 
                                default= True, 
                                help='Train the DiffASD model')
    parser.add_argument('--eval', 
                                default= False, 
                                help='Evaluate the model')
    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    



if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    args = parse_args()
    
    config = OmegaConf.load(args.config)
    
    set_seed(config.model.seed)
    
    if args.train:
        print('Training')
        train(config)
    if args.eval:
        print('Evaluating')
        test(config)
    