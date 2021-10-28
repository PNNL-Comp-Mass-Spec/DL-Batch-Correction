import itertools
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import scipy.stats





class MLP(nn.Module):
    def __init__(self, dims, dropout_p=0.3):
        super(MLP, self).__init__()
        
        self.mlp = nn.Sequential()
        for i in range(len(dims) - 1):
            dim1, dim2 = dims[i], dims[i + 1]
            self.mlp = nn.Sequential(self.mlp,
                                     nn.Linear(dim1, dim2))
            if i + 2 < len(dims):
                self.mlp = nn.Sequential(self.mlp,
                                         nn.BatchNorm1d(dim2),
                                         nn.LeakyReLU(),
                                         nn.Dropout(p = dropout_p))
        
    def forward(self, x):
        return self.mlp(x)



class NormAE(nn.Module):
    def __init__(self, n_features, n_batches, n_latent = 100,
                encoder_width = 1000, encoder_hidden_layers = 2,
                discriminator_width = 100, discriminator_hidden_layers = 1):
        super(NormAE, self).__init__()
        
        self.n_features = n_features
        self.n_batches = n_batches
        self.n_latent = n_latent
        
        encoding_dims = [n_features] + encoder_hidden_layers * [encoder_width] + [n_latent]
        decoding_dims = encoding_dims[::-1]
        decoding_dims[0] = n_latent + n_batches
        discriminator_dims = [n_latent] + discriminator_hidden_layers * [discriminator_width] + [n_batches]
        
        self.encoder = MLP(encoding_dims)
        self.decoder = MLP(decoding_dims)
        self.discriminator = nn.Sequential(MLP(discriminator_dims), nn.Softmax(dim = 1))
        
        self.training = True
        
    def to_one_hot(self, y):
        onehot = torch.zeros(len(y), self.n_batches)
        onehot[torch.arange(len(y)), y] = 1
        return onehot

    def forward(self, device, model_in):
        x, y = model_in
        # encoder
        z = self.encoder(x)
        # discriminate
        y_pred = self.discriminator(z)
        # reconstruct
        y = self.to_one_hot(y)
        y = y.to(device)
        z_plus_y = torch.column_stack((z, y))
        x_rec = self.decoder(z_plus_y)
        # correct
        y_star = torch.zeros((x.shape[0], self.n_batches))
        y_star = y_star.to(device)
        z_plus_y_star = torch.column_stack((z, y_star))
        x_star = self.decoder(z_plus_y_star)

        return x_rec, z, y_pred, x_star

    def compute_lambda(self, schedule, epoch):
      t1, l1, t2, l2 = schedule
      lam = min(l2, l1 + max(0, (epoch-t1)*(l2-l1)/(t2-t1)))
      return lam

    def objective(self, model_in, model_out, params):
        x, y = model_in
        x_rec, z, y_pred, x_star = model_out
        lam = params

        L_rec  = nn.MSELoss()(x_rec, x)
        L_disc = nn.CrossEntropyLoss()(y_pred, y)
        L = L_rec - lam * L_disc

        return L, L_rec, L_disc

    def train(self, train_data, device, num_epochs = 1000, batch_size=8,
              lambda_schedule = [100, 0, 500, 1],
              autoencoder_learning_rate = 2e-4,
              discriminator_learning_rate = 5e-3,
              verbose = True):
        trainloader = torch.utils.data.DataLoader(train_data,
                                                  shuffle = True,
                                                  batch_size = batch_size)
        optimizer1 = torch.optim.Adam(itertools.chain(self.encoder.parameters(),
                                                      self.decoder.parameters()),
                                      lr = autoencoder_learning_rate,
                                      betas = (0.5, 0.9))
        optimizer2 = torch.optim.Adam(self.discriminator.parameters(),
                                      lr = discriminator_learning_rate,
                                      betas = (0.5, 0.9))
        optimizers = [optimizer1, optimizer2]
        self.metrics = np.zeros((num_epochs, 3))
        t = trange(num_epochs) if verbose else range(num_epochs)
        record_every = 100
        record_loss = np.Inf
        done = False
        for epoch in t:
            if done:
                break
            lam = self.compute_lambda(lambda_schedule, epoch)
            i = epoch % 2
            for x, y in trainloader:
                minibatch = x.to(device), y.to(device)
                optimizers[i].zero_grad()
                model_out = self.forward(device, minibatch)
                L, L_rec, L_disc = self.objective(minibatch, model_out, lam)
                ((1-i)*L + i*L_disc).backward()
                self.metrics[epoch] += torch.tensor([L.data, L_rec.data, L_disc.data]).cpu().numpy()
                optimizers[i].step()
            if verbose:
                t.set_description('Loss: {:.4f} | RecLoss: {:.4f} | DiscLoss: {:.4f}'.format(*self.metrics[epoch]))
            if epoch % record_every == 0 and epoch > 0:
                metrics = self.metrics[0:epoch,:]
                min_loss = np.min(metrics[:,0])
                if min_loss < record_loss:
                    record_loss = min_loss
                else:
                    if verbose: print('Early stopping after {} epochs'.format(epoch))
                    self.metrics = metrics
                    done = True
                    
    def plot_metrics(self, lambda_schedule = None):
        import matplotlib.pyplot as plt
        metrics = self.metrics
        if lambda_schedule is None:
            fig, axs = plt.subplots(figsize=(8,8))
            axs.plot(metrics[:,0], label='Reconstruction loss')
            axs.plot(metrics[:,1], label='Discriminator loss')
            axs.plot(metrics[:,2], label='Weighted loss')
            axs.legend()
        else:
            T = np.arange(len(metrics))
            lam = np.array([self.compute_lambda(lambda_schedule, t) for t in T])
            fig, axs = plt.subplots(2, figsize=(8,8))
            axs[0].plot(metrics[:,0], label='Reconstruction loss')
            axs[0].plot(metrics[:,1], label='Discriminator loss')
            axs[0].plot(metrics[:,2], label='Weighted loss')
            axs[1].plot(T, lam, ':', label='Lambda schedule')
            axs[0].legend()
            axs[1].legend()
        return axs



class scGen(nn.Module):
    def __init__(self, n_features, n_batches, n_latent = 100,
                encoder_width = 1000, encoder_hidden_layers = 2,
                verbose = True):
        super(scGen, self).__init__()
        
        self.n_features = n_features
        self.n_batches = n_batches
        self.n_latent = n_latent
        
        encoding_dims = [n_features] + encoder_hidden_layers * [encoder_width] + [n_latent]
        decoding_dims = encoding_dims[::-1]
        
        self.encoder = MLP(encoding_dims)
        self.decoder = MLP(decoding_dims)

        self.training = True
        
    def to_one_hot(self, y):
        onehot = torch.zeros(len(y), self.n_batches)
        onehot[torch.arange(len(y)), y] = 1
        return onehot

    def forward(self, device, model_in):
        x, y = model_in
        # encoder
        z = self.encoder(x)
        # reconstruct
        x_rec = self.decoder(z)
        # correct
        z0 = z.detach().cpu().numpy()
        y0 = y.detach().cpu().numpy()
        z_star = self.correct(z0, y0)
        z_star = torch.tensor(z_star).float()
        z_star = z_star.to(device)
        x_star = self.decoder(z_star)
        return x_rec, z, x_star

    def correct(self, x, y):
        from sklearn import linear_model
        regr = linear_model.LinearRegression()
        y = np.array(pd.get_dummies(y))
        regr.fit(y, x)
        b = regr.coef_
        xstar = x - np.matmul(y, b.T)
        return xstar

    def objective(self, model_in, model_out):
        x, y = model_in
        x_rec, z, x_star = model_out
        L_rec  = nn.MSELoss()(x_rec, x)
        return L_rec

    def train(self, train_data, device, num_epochs = 1000, batch_size=8,
              learning_rate = 2e-4, verbose = True):
        trainloader = torch.utils.data.DataLoader(train_data,
                                                  shuffle = True,
                                                  batch_size = batch_size)
        optimizer = torch.optim.Adam(itertools.chain(self.encoder.parameters(),
                                                      self.decoder.parameters()),
                                      lr = learning_rate,
                                      betas = (0.5, 0.9))
        
        self.metrics = np.zeros(num_epochs)
        t = trange(num_epochs) if verbose else range(num_epochs)
        record_every = 100
        record_loss = np.Inf
        done = False
        for epoch in t:
            if done:
                break
            for x, y in trainloader:
                minibatch = x.to(device), y.to(device)
                optimizer.zero_grad()
                model_out = self.forward(device, minibatch)
                loss = self.objective(minibatch, model_out)
                loss.backward()
                self.metrics[epoch] += torch.tensor([loss.data]).cpu().numpy()
                optimizer.step()
            if verbose:
                t.set_description('RecLoss: {:.4f}'.format(self.metrics[epoch]))
            if epoch % record_every == 0 and epoch > 0:
                metrics = self.metrics[0:epoch]
                min_loss = np.min(metrics)
                if min_loss < record_loss:
                    record_loss = min_loss
                else:
                    if verbose: print('Early stopping after {} epochs'.format(epoch))
                    self.metrics = metrics
                    done = True
                    
    def plot_metrics(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(figsize=(8,8))
        axs.plot(self.metrics[:], label='Reconstruction loss')
        axs.legend()
        return axs



class VAEGAN(nn.Module):
    def __init__(self, n_features, n_batches, n_latent=100):
        super(VAEGAN, self).__init__()
        
        self.n_features = n_features
        self.n_batches = n_batches
        self.n_latent = n_latent
        
        self.mean_encoder = nn.Sequential(
                nn.Linear(n_features, 500),
                nn.BatchNorm1d(500),
                nn.LeakyReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(500, 500),
                nn.BatchNorm1d(500),
                nn.LeakyReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(500, n_latent))
        
        self.logvar_encoder = nn.Sequential(
                nn.Linear(n_features, 500),
                nn.BatchNorm1d(500),
                nn.LeakyReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(500, 500),
                nn.BatchNorm1d(500),
                nn.LeakyReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(500, n_latent))
        
        self.decoder = nn.Sequential(
                    nn.Linear(n_latent+n_batches, 500),
                    nn.BatchNorm1d(500),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.3),
                    nn.Linear(500, 500),
                    nn.BatchNorm1d(500),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.3),
                    nn.Linear(500, n_features))
        
        self.discriminator = nn.Sequential(
            nn.Linear(n_latent, 50),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(50, n_batches),
            nn.Softmax(dim = 1))
        self.training = True
    
    def reparameterization(self, mean, logvar):
        var = torch.exp(1/2 * logvar)
        epsilon = torch.randn_like(var)#.to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
    
    def to_one_hot(self, y):
        onehot = torch.zeros(len(y), self.n_batches)
        onehot[torch.arange(len(y)), y] = 1
        return onehot

    def forward(self, x, y):
        z_mean, z_logvar = self.mean_encoder(x), self.logvar_encoder(x)
        z = self.reparameterization(z_mean, z_logvar)
        y_pred = self.discriminator(z)
        y = self.to_one_hot(y)
        z_plus_y = torch.column_stack((z, y))
        x_pred = self.decoder(z_plus_y)
        return x_pred, y_pred, z, z_mean, z_logvar

    def correct(self, x):
        z_mean, z_logvar = self.mean_encoder(x), self.logvar_encoder(x)
        z = self.reparameterization(z_mean, torch.exp(1/2 * z_logvar))
        y = torch.zeros((x.shape[0], self.n_batches))
        z_plus_y = torch.column_stack((z, y))
        x_star = self.decoder(z_plus_y)
        return x_star
