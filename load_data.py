
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

# TODO: more specific dataset requirements
# assume the crosstab has first row = "feature", remaining rows = sample names
# and the columns = samples
# the batch data has first column = "sample" and second column = "batch"

def load_unlabeled_data(path_to_data):
    
    df = pd.read_csv(path_to_data, delimiter = '\t')

    # Drop NA
    df = df.dropna()

    # Sort crosstab by sample name
    df = df.transpose()
    df.index = df.index.map(int)
    df = df.sort_index()
    df = df.transpose()

    # random shuffle
    df = df.T.sample(frac=1, random_state=42).T

    featureNames = df.index
    sampleNames = df.columns
    n_features = df.shape[0]
    n_samples = df.shape[1]


    x = torch.tensor(df.values, dtype=torch.float64)
    x = x.T.float()

    dataset = TensorDataset(x)
    return dataset, sampleNames, featureNames
    
def load_labeled_data(path_to_data, path_to_labels, test_size=0.5, random_seed=42):

    # load expression data
    x = pd.read_csv(path_to_data, delimiter = '\t')
    x = x.set_index('feature')
    x.columns = x.columns.astype(str)
    x = x.dropna()

    featureNames = x.index
    sampleNames  = x.columns
    
    # load labels
    y = pd.read_csv(path_to_labels, delimiter = '\t')
    y = y.set_index('sample')
    y.index = y.index.astype(str)
    
    # convert labels to zero-indexed ints
    y['batch'] = [pd.Index(pd.unique(y['batch'])).get_loc(i) for i in y['batch']]

    # reorder to match expression data
    y = y.reindex(x.columns)
    
    # convert to tensor
    x = torch.tensor(x.values, dtype=torch.float64)
    y = torch.tensor(y.values)
    
    # remove extra dimension
    y = y.squeeze(1)
    
    # normalization
    x = x.T.float()
    x = (x - torch.mean(x, dim=0)) / torch.sqrt(torch.var(x, dim=0))
    
    # join expression data with labels
    dataset = TensorDataset(x, y)
    
    # gather metadata
    n_features   = len(dataset[0][:][0])
    n_batches    = len(torch.unique(dataset[:][1]))
    
    metadata = {'featureNames':featureNames,
                'sampleNames':sampleNames,
                'n_features':n_features,
                'n_batches':n_batches}
    
    
    from torch.utils.data import DataLoader, Subset
    from sklearn.model_selection import train_test_split

    # generate indices: instead of the actual data we pass in integers instead
    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        random_state=random_seed
    )
    
    metadata['train_idx'] = train_idx
    metadata['test_idx'] = test_idx

    # generate subset based on indices
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    return train_dataset, test_dataset, metadata


def load_all_data(path_to_data, path_to_labels):

    # load expression data
    x = pd.read_csv(path_to_data, delimiter = '\t')
    x = x.set_index('feature')
    x.columns = x.columns.astype(str)
    x = x.dropna()

    featureNames = x.index
    sampleNames  = x.columns
    
    # load labels
    y = pd.read_csv(path_to_labels, delimiter = '\t')
    y = y.set_index('sample')
    y.index = y.index.astype(str)
    
    # convert labels to zero-indexed ints
    y['batch'] = [pd.Index(pd.unique(y['batch'])).get_loc(i) for i in y['batch']]

    # reorder to match expression data
    y = y.reindex(x.columns)
    
    # convert to tensor
    x = torch.tensor(x.values, dtype=torch.float64)
    y = torch.tensor(y.values)
    
    # remove extra dimension
    y = y.squeeze(1)
    
    # normalization
    x = x.T.float()
    mu = torch.mean(x, dim=0)
    mu = mu.unsqueeze(1).repeat(1, x.shape[0])
    mu = mu.T
    x -= mu
    
    # join expression data with labels
    dataset = TensorDataset(x, y)
    
    # gather metadata
    n_features   = len(dataset[0][:][0])
    n_batches    = len(torch.unique(dataset[:][1]))
    
    metadata = {'featureNames':featureNames,
                'sampleNames':sampleNames,
                'n_features':n_features,
                'n_batches':n_batches}
    
    return dataset, metadata

def save_model_outputs(model, test_dataset, train_dataset, metadata, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    featureNames, sampleNames, n_features, n_batches = metadata
    
    train_size = len(train_dataset)
    n_samples = len(train_dataset) + len(test_dataset)
    # Save original dataset in the same format as everything else
    
    for i in range(2):
        dataset = [test_dataset, train_dataset][i]
        datatype = ['test','train'][i]
        
        if i==0:
            idx = np.arange(n_samples)[train_size:]
        else:
            idx = np.arange(n_samples)[:train_size]
        
        # export labels
        y = dataset[:][1]
        y = y.data.cpu().numpy()
        
        df = pd.DataFrame(y, index = sampleNames[idx], columns=['batch'])
        df.index.name = 'sample'
        df.to_csv(os.path.join(output_dir, '{}_labels.csv'.format(datatype)))
        
        # export original data
        x = dataset[:][0]
        x = x.data.cpu().numpy()

        df = pd.DataFrame(x.T, columns = sampleNames[idx], index = featureNames)
        df.to_csv(os.path.join(output_dir, 'original_{}_data.csv'.format(datatype)))

        # Pass test dataset through model
        x = dataset[:][0]
        y = dataset[:][1]
        x_pred, y_pred, z = model.forward(x, y)
        z      = z.data.cpu().numpy()
        x_pred = x_pred.data.cpu().numpy()

        df = pd.DataFrame(z.T, columns = sampleNames[idx])
        df.to_csv(os.path.join(output_dir, 'encoded_{}_data.csv'.format(datatype)))

        df = pd.DataFrame(x_pred.T, columns = sampleNames[idx], index = featureNames)
        df.to_csv(os.path.join(output_dir, 'reconstructed_{}_data.csv'.format(datatype)))

        # Apply batch correction
        x = dataset[:][0]
        
        x_star = model.correct(x)

        x_star = x_star.data.cpu().numpy()

        df = pd.DataFrame(x_star.T, columns = sampleNames[idx], index = featureNames)
        df.to_csv(os.path.join(output_dir, 'corrected_{}_data.csv'.format(datatype)))
        
        
        
def save_vaegan_outputs(model, test_dataset, train_dataset, metadata, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    featureNames, sampleNames, n_features, n_batches = metadata
    
    train_size = len(train_dataset)
    n_samples = len(train_dataset) + len(test_dataset)
    # Save original dataset in the same format as everything else
    
    for i in range(2):
        dataset = [test_dataset, train_dataset][i]
        datatype = ['test','train'][i]
        
        if i==0:
            idx = np.arange(n_samples)[train_size:]
        else:
            idx = np.arange(n_samples)[:train_size]
        
        # export labels
        y = dataset[:][1]
        y = y.data.cpu().numpy()
        
        df = pd.DataFrame(y, index = sampleNames[idx], columns=['batch'])
        df.index.name = 'sample'
        df.to_csv(os.path.join(output_dir, '{}_labels.csv'.format(datatype)))
        
        # export original data
        x = dataset[:][0]
        x = x.data.cpu().numpy()

        df = pd.DataFrame(x.T, columns = sampleNames[idx], index = featureNames)
        df.to_csv(os.path.join(output_dir, 'original_{}_data.csv'.format(datatype)))

        # Pass test dataset through model
        x = dataset[:][0]
        y = dataset[:][1]
        x_pred, y_pred, z, z_mean, z_logvar = model.forward(x, y)
        z      = z.data.cpu().numpy()
        x_pred = x_pred.data.cpu().numpy()

        df = pd.DataFrame(z.T, columns = sampleNames[idx])
        df.to_csv(os.path.join(output_dir, 'encoded_{}_data.csv'.format(datatype)))

        df = pd.DataFrame(x_pred.T, columns = sampleNames[idx], index = featureNames)
        df.to_csv(os.path.join(output_dir, 'reconstructed_{}_data.csv'.format(datatype)))

        # Apply batch correction
        x = dataset[:][0]
        
        x_star = model.correct(x)

        x_star = x_star.data.cpu().numpy()

        df = pd.DataFrame(x_star.T, columns = sampleNames[idx], index = featureNames)
        df.to_csv(os.path.join(output_dir, 'corrected_{}_data.csv'.format(datatype)))

        
def make_labeled_dataset(x, y, batch = 'batch', test_size = 0.5, random_seed = 42):

    # load expression data
    
    x.columns = x.columns.astype(str)
    x = x.dropna() # TODO: warning message here

    featureNames = x.index
    sampleNames  = x.columns
    
    # load labels
    y = y[[batch]]
    y.index = y.index.astype(str)
    
    # convert labels to zero-indexed ints
    y[batch] = [pd.Index(pd.unique(y[batch])).get_loc(i) for i in y[batch]]

    # reorder to match expression data
    y = y.reindex(x.columns) # TODO: warning message here
    
    # convert to tensor
    x = torch.tensor(x.values, dtype=torch.float64)
    y = torch.tensor(y.values)
    
    # remove extra dimension
    y = y.squeeze(1)
    
    # normalization
    x = x.T.float()
    x = (x - torch.mean(x, dim=0)) / torch.sqrt(torch.var(x, dim=0))
    
    # join expression data with labels
    dataset = TensorDataset(x, y)
    
    # gather metadata
    n_features   = len(dataset[0][:][0])
    n_batches    = len(torch.unique(dataset[:][1]))
    
    metadata = {'featureNames':featureNames,
                'sampleNames':sampleNames,
                'n_features':n_features,
                'n_batches':n_batches}
    
    
    from torch.utils.data import DataLoader, Subset
    from sklearn.model_selection import train_test_split

    # generate indices: instead of the actual data we pass in integers instead
    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        random_state=random_seed
    )
    
    metadata['train_idx'] = train_idx
    metadata['test_idx'] = test_idx

    # generate subset based on indices
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    return train_dataset, test_dataset, metadata

