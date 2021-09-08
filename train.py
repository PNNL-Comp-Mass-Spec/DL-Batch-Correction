
import os
import pandas as pd
import torch
from tqdm import tqdm, trange

from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

import torch.optim as optim
from torch.optim import Adam

import itertools


def train_aegan(model, train_data, 
                num_epochs = 1000,
                batch_size=8,
                num_pretraining_epochs = [0, 0],
                autoencoder_learning_rate = 2e-4,
                discriminator_learning_rate = 5e-3,
                regularization = 1,
                verbose = True):
    
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    
    optimizer_ae = torch.optim.Adam(
        itertools.chain(model.encoder.parameters(), model.decoder.parameters()), lr = autoencoder_learning_rate,
        betas=(0.5, 0.9)
    )
    optimizer_disc = torch.optim.Adam(model.discriminator.parameters(), lr = discriminator_learning_rate, betas=(0.5, 0.9))
    
    k = 2
    metrics = []
    
    phase_epochs = num_pretraining_epochs + [num_epochs]
    phase_epochs = np.cumsum(phase_epochs)
    num_epochs = np.sum(phase_epochs)
    # TODO: convergence criterion
    t = trange(num_epochs+1) if verbose else range(num_epochs+1)
    for epoch in t:
        
        phase = np.sum(epoch >= phase_epochs)
        loss_history = []

        for x, y in trainloader:
            
            optimizer_ae.zero_grad()
            optimizer_disc.zero_grad()

            x_pred, y_pred, z = model.forward(x, y)
            
            rec_loss  = nn.MSELoss()(x_pred, x)
            disc_loss = nn.CrossEntropyLoss()(y_pred, y)

            loss = rec_loss - regularization * disc_loss

            loss_history.append((rec_loss.data, disc_loss.data))

            if phase == 0 or (phase == 2 and epoch % k == 0):
                loss.backward()
                optimizer_ae.step()

            elif phase == 1 or (phase == 2 and epoch & k != 0):
                loss *= -1
                loss.backward()
                optimizer_disc.step()
                
        metrics.append(np.mean(loss_history, axis=0))

        if verbose:
        #    epoch_str = str(epoch).zfill(int(1+np.log10(num_epochs)))
        #    print('Epoch: {}/{} | RecLoss: {:.4f} | DiscLoss: {:.4f}'.format(epoch_str, num_epochs, metrics[-1][0], metrics[-1][1]))
            t.set_description('RecLoss: {:.4f} | DiscLoss: {:.4f}'.format(metrics[-1][0], metrics[-1][1]))
            
    metrics = np.array(metrics)
    return model, metrics


def train_vaegan(model, train_data, 
                num_epochs = 1000,
                batch_size = 8,
                autoencoder_learning_rate = 2e-4,
                discriminator_learning_rate = 5e-3,
                kl_weight = 1e-2,
                regularization = 1,
                verbose = True):
    
    encoder_history       = []
    discriminator_history = []
    
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    
    optimizer_ae = torch.optim.Adam(itertools.chain(model.mean_encoder.parameters(),
                                                    model.logvar_encoder.parameters(),
                                                    model.decoder.parameters()),
                                    lr = autoencoder_learning_rate,
                                    betas=(0.5, 0.9))
    optimizer_disc = torch.optim.Adam(model.discriminator.parameters(),
                                      lr = discriminator_learning_rate,
                                      betas=(0.5, 0.9))
    
    k = 2
    metrics = []
    
    # TODO: convergence criterion
   

    for epoch in range(num_epochs+1):
        
        loss_history = []

        for x, y in trainloader:
            
            optimizer_ae.zero_grad()
            optimizer_disc.zero_grad()

            x_pred, y_pred, z, z_mean, z_logvar = model.forward(x, y)
            
            rec_loss  = nn.MSELoss()(x_pred, x)
            KLD       = -1/2 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
            disc_loss = nn.CrossEntropyLoss()(y_pred, y)
            

            loss = rec_loss + kl_weight*KLD - regularization*disc_loss

            loss_history.append((rec_loss.data, KLD.data, disc_loss.data))

            if epoch % k == 0:
                loss.backward()
                optimizer_ae.step()

            else:
                loss *= -1
                loss.backward()
                optimizer_disc.step()
                
        metrics.append(np.mean(loss_history, axis=0))

        if verbose and epoch % int(num_epochs/10) == 0:
            epoch_str = str(epoch).zfill(int(1+np.log10(num_epochs)))
            print('Epoch: {}/{} | RecLoss: {:.4f} | KLD: {:.4f} | DiscLoss: {:.4f}'.format(epoch_str, num_epochs, metrics[-1][0], metrics[-1][1], metrics[-1][2]))
           
    metrics = np.array(metrics)
    return model, metrics