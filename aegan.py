
import torch
from torch import nn
from torch.autograd import Variable
import itertools
from tqdm import trange
import numpy as np

class AEGAN(nn.Module):
    
    
    def __init__(self, n_features, n_batches, n_latent=100,
                n_hidden = 2):
        super(AEGAN, self).__init__()
        
        self.n_features = n_features
        self.n_batches  = n_batches
        self.n_latent   = n_latent
        self.n_hidden   = n_hidden
        
        
       
        def multi_layer_perceptron(dims, dropout_p = 0.3):
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
        
        encoder_dims       = [n_features, 1000, 1000, n_latent]
        decoder_dims       = [n_latent+n_batches, 1000, 1000, n_features]
        discriminator_dims = [n_latent, 100, 100, n_batches]
        
        self.encoder       = multi_layer_perceptron(encoder_dims)
        self.decoder       = multi_layer_perceptron(decoder_dims)
        self.discriminator = multi_layer_perceptron(discriminator_dims)
        
        self.training = True
        
    def to_one_hot(self, y):
        onehot = torch.zeros(len(y), self.n_batches)
        onehot[torch.arange(len(y)), y] = 1
        return onehot

    def forward(self, x, y):
        z = self.encoder(x)
        y_pred = self.discriminator(z)
        y = self.to_one_hot(y)
        z_plus_y = torch.column_stack((z, y))
        x_pred = self.decoder(z_plus_y)
        return x_pred, y_pred

    def correct(self, x):
        z = self.encoder(x)
        y = torch.zeros((x.shape[0], self.n_batches))
        z_plus_y = torch.column_stack((z, y))
        x_star = self.decoder(z_plus_y)
        return x_star
    
    def train(self, train_data, 
                num_epochs = 1000,
                batch_size=8,
                autoencoder_learning_rate = 2e-4,
                discriminator_learning_rate = 5e-3,
                regularization = 1,
                verbose = True):
    
        trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

        optimizer_ae = torch.optim.Adam(itertools.chain(self.encoder.parameters(),
                                                        self.decoder.parameters()),
                                        lr = autoencoder_learning_rate,
                                        betas=(0.5, 0.9))
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(),
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
        t = trange(num_epochs)
        for epoch in t:
            epoch_history = []
            for x, y in trainloader:
                
                for optimizer in self.optimizers:
                    optimizer.zero_grad()

                x_pred, y_pred = self.forward(x, y)
                
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
        
        
    def plot_history(self, file = None):
        history = self.history
        
        import matplotlib.pyplot as plt
        fig, (ax1) = plt.subplots(1, 1)

        ax1.set_xlabel('Epoch')
        ax1.plot(history.T[0],
                 label='Reconstruction error')
        ax1.plot(history.T[1] - np.log(self.n_batches),
                 label='Classification error')
        plt.legend()
        
        plt.show() if file is None else plt.savefig(file)
    
    
    

