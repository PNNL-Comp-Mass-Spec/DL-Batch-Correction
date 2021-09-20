
import torch
from torch import nn
from torch.autograd import Variable

    
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
    
class BatchDiscriminator(nn.Module):
    def __init__(self, n_batches, n_latent=100):
        super(BatchDiscriminator, self).__init__()
        # TODO: use super() instead?
        self.classifier = nn.Sequential(
            MLP(dims=[n_latent, 50, n_batches]),
            nn.Softmax(dim = 1))

    def forward(self, x):
        y = self.classifier(x)
        return y
    
    
class Encoder(nn.Module):
    
    def __init__(self, n_features, n_latent=100):
        super(Encoder, self).__init__()
        
        self.E = MLP(dims=[n_features, 1000, 1000, n_latent])
        self.training = True
        
    def forward(self, x):
        
        z = self.E(x)
        return z
    
class Decoder(nn.Module):
    
    def __init__(self, n_features, n_latent=100):
        super(Decoder, self).__init__()

        self.D = nn.Sequential(
                nn.Linear(n_latent, 500),
                nn.BatchNorm1d(500),
                nn.LeakyReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(500, 500),
                nn.BatchNorm1d(500),
                nn.LeakyReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(500, n_features)
          )
        self.training = True
        
    def forward(self, z):
        xhat = self.D(z)
        
        return xhat
    
class DecoderWithBatchLabels(nn.Module):
    
    def __init__(self, n_features, n_batches, n_latent=100):
        super(DecoderWithBatchLabels, self).__init__()

        self.n_batches = n_batches
        self.D = nn.Sequential(
                nn.Linear(n_latent+n_batches, 500),
                nn.BatchNorm1d(500),
                nn.LeakyReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(500, 500),
                nn.BatchNorm1d(500),
                nn.LeakyReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(500, n_features)
          )
        self.training = True
        
    def forward(self, z, b):
        n_batches = self.n_batches
        onehot = torch.zeros(n_batches)
        
        onehot = torch.zeros(len(b), n_batches)
        onehot[torch.arange(len(b)), b] = 1
        
        z_plus_b = torch.column_stack((z, onehot))
        xhat = self.D(z_plus_b)
        
        return xhat
    
 
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        z = self.encoder.forward(x)
        x_hat = self.decoder.forward(z)
        return x_hat
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)#.to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.encoder.forward(x)
        z = self.reparameterization(mean, torch.exp(1/2 * log_var))
        x_hat = self.decoder.forward(z)
        
        return x_hat, mean, log_var
    
    
    
    
class AEGAN(nn.Module):
    
    
    def __init__(self, n_features, n_batches, n_latent=100,
                n_hidden = 2):
        super(AEGAN, self).__init__()
        
        self.n_features = n_features
        self.n_batches = n_batches
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        
        
        
        def std_layer(n_in, n_out, dropout_p = 0.3):
            layer = nn.Sequential(nn.Linear(n_in, n_out),
                                  nn.BatchNorm1d(n_out),
                                  nn.LeakyReLU(),
                                  nn.Dropout(p = dropout_p))
            return layer
        
        def multi_layer(n_in, n_out, n_hidden, dropout_p):
            network = nn.Sequential()
            dims = [n_in] + [1000 for _ in range(n_hidden)] + [n_out]
            
            for i in range(n_hidden):
                layer = std_layer(dims[i], dims[i+1])
                network = nn.Sequential(network, layer)
                
            layer = nn.Linear(dims[n_hidden], dims[n_hidden+1])
            network = nn.Sequential(network, layer)
            
            return network
        
        self.encoder = multi_layer(n_in      = n_features,
                                   n_out     = n_latent,
                                   n_hidden  = n_hidden,
                                   dropout_p = 0.3)
        self.decoder = multi_layer(n_in      = n_latent + n_batches,
                                   n_out     = n_features,
                                   n_hidden  = n_hidden,
                                   dropout_p = 0.3)
        
        self.discriminator = multi_layer(n_in      = n_latent,
                                         n_out     = n_batches,
                                         n_hidden  = n_hidden,
                                         dropout_p = 0.3)
        self.discriminator = nn.Sequential(self.discriminator, nn.Softmax(dim = 1))
        
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
        return x_pred, y_pred, z

    def correct(self, x):
        z = self.encoder(x)
        y = torch.zeros((x.shape[0], self.n_batches))
        z_plus_y = torch.column_stack((z, y))
        x_star = self.decoder(z_plus_y)
        return x_star
    
    
    
    
    
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

