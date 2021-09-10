
import torch
from torch import nn
from torch.autograd import Variable
import itertools
from tqdm import trange
import numpy as np


class VAEGANModule(nn.Module):
    def __init__(self, n_features, n_batches, n_latent=100,
                n_hidden = 2):
        super(VAEGANModule, self).__init__()
        
        self.n_features = n_features
        self.n_batches  = n_batches
        self.n_latent   = n_latent
        self.n_hidden   = n_hidden
        
        encoder_dims       = [n_features, 1000, 1000, n_latent]
        decoder_dims       = [n_latent+n_batches, 1000, 1000, n_features]
        discriminator_dims = [n_latent, 100, 100, n_batches]
        
        self.mean_encoder       = self.multi_layer_perceptron(encoder_dims)
        self.logvar_encoder     = self.multi_layer_perceptron(encoder_dims)
        self.decoder            = self.multi_layer_perceptron(decoder_dims)
        self.discriminator      = self.multi_layer_perceptron(discriminator_dims)
        
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
    
    def reparameterization(self, mean, logvar):
        var = torch.exp(1/2 * logvar)
        epsilon = torch.randn_like(var)       # sampling epsilon 
        epsilon = epsilon.to(mean.device)
        z = mean + var*epsilon                          # reparameterization trick
        return z
    
    def to_one_hot(self, y):
        y = y.view(-1,1)
        y_onehot = torch.FloatTensor(y.shape[0], self.n_batches)
        y_onehot = y_onehot.to(y.device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)
        return(y_onehot)
    
    def forward(self, x, y):
        z_mean, z_logvar = self.mean_encoder(x), self.logvar_encoder(x)
        z = self.reparameterization(z_mean, z_logvar)
        y_pred = self.discriminator(z)
        y = self.to_one_hot(y)
        z_plus_y = torch.column_stack((z, y))
        x_pred = self.decoder(z_plus_y)
        return z_mean, z_logvar, x_pred, y_pred
    
    def correct(self, x):
        z = self.encoder(x)
        y = torch.zeros((x.shape[0], self.n_batches))
        z_plus_y = torch.column_stack((z, y))
        x_star = self.decoder(z_plus_y)
        return x_star

class VAEGAN():
    
    def __init__(self, n_features, n_batches, n_latent=100, n_hidden = 2):
        self.n_features = n_features
        self.n_batches  = n_batches
        self.n_latent   = n_latent
        self.n_hidden   = n_hidden
        
        self.model = VAEGANModule(n_features, n_batches, n_latent, n_hidden)
        
        
        
    def train(self, train_data, device,
              num_epochs = 1000,
              batch_size=8,
              autoencoder_learning_rate = 2e-4,
              discriminator_learning_rate = 5e-3,
              regularization = 1,
              kl_weight = 1e-2,
              verbose = True):
    
        trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

        optimizer_ae = torch.optim.Adam(itertools.chain(self.model.mean_encoder.parameters(),
                                                        self.model.logvar_encoder.parameters(),
                                                        self.model.decoder.parameters()),
                                        lr = autoencoder_learning_rate,
                                        betas=(0.5, 0.9))
        optimizer_disc = torch.optim.Adam(self.model.discriminator.parameters(),
                                          lr = discriminator_learning_rate,
                                          betas=(0.5, 0.9))
        self.optimizers = [optimizer_ae, optimizer_disc]
        
        def loss_fn(x, y, z_mean, z_logvar, x_pred, y_pred):
            rec_loss  = nn.MSELoss()(x_pred, x)
            KLD       = -1/2 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
            disc_loss = nn.CrossEntropyLoss()(y_pred, y)
                
            loss = rec_loss + kl_weight*KLD - regularization*disc_loss
            
            metrics = {'rec_loss'  : rec_loss,
                       'KLD'       : KLD,
                       'disc_loss' : disc_loss,
                       'loss'      : loss}
            return metrics
        
        history = []
        t = trange(num_epochs,  position=0, leave=True)
        self.model = self.model.to(device)
        for epoch in t:
            epoch_history = []
            for x, y in trainloader:
                x = x.to(device)
                y = y.to(device)
                for optimizer in self.optimizers:
                    optimizer.zero_grad()

                z_mean, z_logvar, x_pred, y_pred = self.model.forward(x, y)
                
                metrics = loss_fn(x, y, z_mean, z_logvar, x_pred, y_pred)
                
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
                t.set_description('RecLoss: {:.4f} | KLD: {:.4f} | DiscLoss: {:.4f}'.format(history[-1][0], history[-1][1], history[-1][2]))

        self.history = np.array(history)
        
        
    def plot_history(self, tmin=0, tmax=None, file = None):
        history = self.history
        
        import matplotlib.pyplot as plt
        fig, (ax1) = plt.subplots(1, 1)
        tmax = self.history.shape[1] if tmax is None else tmax
        ax1.set_xlabel('Epoch')
        ax1.plot(history.T[0][tmin:tmax],
                 label='Reconstruction error')
        ax1.plot(history.T[1][tmin:tmax],
                 label='KL Divergence')
        ax1.plot(history.T[2][tmin:tmax] - np.log(self.n_batches),
                 label='Classification error')
        plt.legend()
        
        plt.show() if file is None else plt.savefig(file)
    
    
    

