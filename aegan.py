
import torch
from torch import nn
from torch.autograd import Variable
import itertools
from tqdm import trange
import numpy as np


class AEGANModule(nn.Module):
    def __init__(self, n_features, n_batches, n_latent=100,
                n_hidden = 2):
        super(AEGANModule, self).__init__()
        
        self.n_features = n_features
        self.n_batches  = n_batches
        self.n_latent   = n_latent
        self.n_hidden   = n_hidden
        
        encoder_dims       = [n_features, 1000, 1000, n_latent]
        decoder_dims       = [n_latent+n_batches, 1000, 1000, n_features]
        discriminator_dims = [n_latent, 100, 100, n_batches]
        
        self.encoder       = self.multi_layer_perceptron(encoder_dims)
        self.decoder       = self.multi_layer_perceptron(decoder_dims)
        self.discriminator = self.multi_layer_perceptron(discriminator_dims)
        self.discriminator = nn.Sequential(self.discriminator, nn.Softmax(dim = 1))
        
        self.training = True
    
    def multi_layer_perceptron(self, dims, dropout_p = 0.3):
            MLP = nn.Sequential()
            for i in range(len(dims) - 1):
                dim1, dim2 = dims[i], dims[i + 1]
                MLP = nn.Sequential(MLP, nn.Linear(dim1, dim2))
                if i + 2 < len(dims):
                    MLP = nn.Sequential(MLP,
                                        nn.BatchNorm1d(dim2),
                                        nn.LeakyReLU(),
                                        nn.Dropout(p = dropout_p))
            return MLP
        
    def to_one_hot(self, y):
        y = y.view(-1,1)
        y_onehot = torch.empty(y.shape[0], self.n_batches, device=y.device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)
        return(y_onehot)
    
    def forward(self, x, y):
        z = self.encoder(x)
        y_pred = self.discriminator(z)
        y = self.to_one_hot(y)
        z_plus_y = torch.column_stack((z, y))
        x_pred = self.decoder(z_plus_y)
        return x_pred, y_pred

    def correct(self, x):
        z = self.encoder(x)
        y = torch.zeros((x.shape[0], self.n_batches), device=x.device)
        z_plus_y = torch.column_stack((z, y))
        x_star = self.decoder(z_plus_y)
        return x_star

class AEGAN():
    
    def __init__(self, n_features, n_batches, n_latent=100, n_hidden = 2):
        self.n_features = n_features
        self.n_batches  = n_batches
        self.n_latent   = n_latent
        self.n_hidden   = n_hidden
        
        self.model = AEGANModule(n_features, n_batches, n_latent, n_hidden)
        
        
    
    
    def train(self, train_data, device,
                num_epochs = 1000,
                batch_size=8,
                autoencoder_learning_rate = 2e-4,
                discriminator_learning_rate = 5e-3,
                regularization = 1,
                verbose = True):
    
        trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

        optimizer_ae = torch.optim.Adam(itertools.chain(self.model.encoder.parameters(),
                                                        self.model.decoder.parameters()),
                                        lr = autoencoder_learning_rate,
                                        betas=(0.5, 0.9))
        optimizer_disc = torch.optim.Adam(self.model.discriminator.parameters(),
                                          lr = discriminator_learning_rate,
                                          betas=(0.5, 0.9))
        self.optimizers = [optimizer_ae, optimizer_disc]

        
        def loss_fn(x, y, x_pred, y_pred):
            rec_loss  = nn.MSELoss()(x_pred, x)
            disc_loss = nn.CrossEntropyLoss()(y_pred, y)
            loss = rec_loss - regularization * disc_loss
            metrics = {'rec_loss'  : rec_loss,
                       'disc_loss' : disc_loss,
                       'loss'      : loss}
            return metrics
    
        history = []
        self.model = self.model.to(device)
        t = trange(num_epochs,  position=0, leave=True)
        for epoch in t:
            epoch_history = []
            for x, y in trainloader:
                x = x.to(device)
                y = y.to(device)
                for optimizer in self.optimizers:
                    optimizer.zero_grad()

                x_pred, y_pred = self.model.forward(x, y)
                
                metrics = loss_fn(x, y, x_pred, y_pred)
                
                if epoch % 2 == 0:
                    loss = metrics['loss']
                    optimizer = self.optimizers[0]
                else:
                    loss = -1 * metrics['loss']
                    optimizer = self.optimizers[1]
                    
                loss.backward()
                optimizer.step()
                    
                epoch_history += [[metric.data.item() for metric in metrics.values()]]
            history += [np.mean(epoch_history, axis=0)]
            
            if verbose:
                t.set_description('RecLoss: {:.4f} | DiscLoss: {:.4f}'.format(history[-1][0], history[-1][1]))

        self.history = np.array(history)
        
        
    def plot_history(self, tmin=0, tmax=None, file = None):
        history = self.history
        
        import matplotlib.pyplot as plt
        fig, (ax1) = plt.subplots(1, 1)
        tmax = self.history.shape[0] if tmax is None else tmax
        ax1.set_xlabel('Epoch')
        ax1.plot(history.T[0][tmin:tmax],
                 label='Reconstruction error')
        ax1.plot(history.T[1][tmin:tmax] - np.log(self.n_batches),
                 label='Classification error')
        plt.legend()
        
        plt.show() if file is None else plt.savefig(file)
    
    
    def get_model_outputs(self, device, data):
        self.model = self.model.to(device)
        x = data[:][0].to(device)
        y = data[:][1].to(device)
        
        z = self.model.encoder(x)
        x_pred, _ = self.model.forward(x, y)
        x_star    = self.model.correct(x)
        
        # Original data
        model_outputs = {'original'      : x,
                         'encoded'       : z,
                         'reconstructed' : x_pred,
                         'corrected'     : x_star}
        model_outputs = {_ : x.detach().numpy() for (_, x) in model_outputs.items()}
        return model_outputs

        
    def compute_pca(self, data, labels):
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from scipy.linalg import eigh
        import pandas as pd 

        X = StandardScaler().fit_transform(data)
        A = np.matmul(X.T , X)

        k = X.shape[1]
        values, vectors = eigh(A, eigvals=(k-2, k-1))
        vectors = vectors.T
        T = np.matmul(vectors, X.T)

        T = np.vstack((T, labels)).T
        df = pd.DataFrame(data=T, columns=('PC1', 'PC2', 'label'))
        df['label'] = df['label'].astype('category')
        return df
    
    def plot_all_pca(self, data, device, title=None):
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle(title)

        y = data[:][1]
        
        model_outputs = self.get_model_outputs(device, data)
        
        
        # Original data
        x = model_outputs['original']
        df = self.compute_pca(x, y)
        sns.scatterplot(x='PC1', y='PC2', data=df, hue='label', legend=False, ax=axs[0,0])
        axs[0,0].set_title('Original')

        # Encoded data
        x = model_outputs['encoded']
        df = self.compute_pca(x, y)
        sns.scatterplot(x='PC1', y='PC2', data=df, hue='label', legend=False, ax=axs[0,1])
        axs[0,1].set_title('Encoded')

        # Reconstructed data
        x = model_outputs['reconstructed']
        df = self.compute_pca(x, y)
        sns.scatterplot(x='PC1', y='PC2', data=df, hue='label', legend=False, ax=axs[1,0])
        axs[1,0].set_title('Reconstructed')

        # Corrected data
        x = model_outputs['corrected']
        df = self.compute_pca(x, y)
        sns.scatterplot(x='PC1', y='PC2', data=df, hue='label', legend=False, ax=axs[1,1])
        axs[1,1].set_title('Corrected')

    

