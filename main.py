
import argparse
import matplotlib.pyplot as plt
import os
import torch

from model import Encoder, DecoderWithBatchLabels, BatchDiscriminator
from model import AEGAN
from train import train_aegan
from load_data import load_labeled_data, save_model_outputs
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data')
    parser.add_argument('--path_to_labels')
    parser.add_argument('--output_dir')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--vis', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    return parser.parse_args()

def main():
    
    args = parse_args()
    
    path_to_data   = args.path_to_data
    path_to_labels = args.path_to_labels
    output_dir     = args.output_dir
    num_epochs     = args.num_epochs
    batch_size     = args.batch_size
    n_latent       = args.latent_dim
    vis            = args.vis
    verbose        = args.verbose
    
    DEVICE = torch.device("cpu")
    
    if verbose: print('Loading data... ', end='')
        
    train_data, test_data, metadata = load_labeled_data(path_to_data, path_to_labels)
    featureNames, sampleNames, n_features, n_batches = metadata
    
    if verbose: print('done!\nLoading PyTorch model... ', end='')
        
    model = AEGAN(n_features, n_batches, n_latent=10)
    model = model.to(DEVICE)
    
    if verbose: print('done!\nBeginning training for {} epochs'.format(num_epochs))
    
    model, metrics = train_aegan(model, train_data, num_epochs, batch_size)
    
    if verbose: print('Saving model outputs... ', end='')
        
    save_model_outputs(model,
                       test_data, train_data, metadata,
                       output_dir)
    
    fig, (ax1) = plt.subplots(1, 1)
    ax1.set_xlabel('Epoch')
    ax1.plot(metrics.T[0], label='Classification eror')
    ax1.plot(metrics.T[1], label='Reconstruction error')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'metrics.pdf'))
    
    if vis:
        if verbose: print('done!\nPlotting results... ', end='')
        os.system('Rscript.exe visualization.R -p {}'.format(output_dir))
        if verbose: print('done.')
    
if __name__ == '__main__':
    main()
    