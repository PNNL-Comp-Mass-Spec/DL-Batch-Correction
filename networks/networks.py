import torch
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader


## Networks needed for the corrections

class SelfAttention(nn.Module):
    def __init__(self, emb, heads):
        super().__init__()

        ## This is the dimension of the embedding being used to code the letters (factors) found in the data.
        self.emb = emb

        ## This is the number of "attention heads". Each attention head is generating 3 matrices. Keys, Queries and Values.
        ## Queries and Keys are multiplied and passed to softmax, which generates a vector of positive "weights". 
        ## The weights are used to transform the input from x_1,...,x_n into y_1,...,y_n. The interpretation is that this 
        ## matrix can learn patterns within the sequential data. y_1 for instance can be interpreted as containing information on
        ## the interaction between x_1 and x_1,..., x_n. 

        ## Multiple "attention heads" then are transforming the data according to different weight matrices, so these different attention heads
        ## can in theory look for different interactions within the sequential data.
        self.heads = heads

        ## Each attention head has its own K, Q, V matrices. 
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        ## Output from attention heads has the same dimensions as the input. The interpretation here is that this linear
        ## layer is combining all "patterns" extracted from each attention head.
        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, mask):

        ## b is the minibatch number. This is how many sequences are fed into the network
        ## t is the length of the sequences that are passed to the network. In our case, something like max peptide length in dataset.
        ## e will be the embedding dimension of the letters in the alphabet. Likely something like 5 or so, as there's only ~20 amino acids. (??)
        b, t, e = x.size()

        ## Completely independent parameter from the input in principle.
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        ## The output from all attention heads is concatenated. So the sizes are reshaped to split into the
        ## number of heads h.
        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        ## The weight matrices are computed all together. This is why the keys, queries and values are concatenated.
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        # - get dot product of queries and keys, and scale
        ## The matrix 'dot' represents the weights used when transforming the original input. 
        ## All the heads are contained here, in the first (zero-th) dimension of the tensor.
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot / math.sqrt(e) # dot contains b*h  t-by-t matrices with raw self-attention logits

        assert dot.size() == (b*h, t, t), f'Matrix has size {dot.size()}, expected {(b*h, t, t)}.'

        mask = mask.repeat_interleave(h, 0)
        dot = F.softmax(dot - mask, dim=2) # dot now has row-wise self-attention probabilities

        ## This line from the original code was causing an error. Seems to be an NA check. Will add later.
        ## OLD LINE - assert not former.util.contains_nan(dot[:, 1:, :]) # only the forst row may contain nan

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        ## The weight matrices are used to transform the original sequence of inputs. Here, we use the weight matrices
        ## from each attention head to transform the input vectors x_1,...,x_t into h * t many vectors y, each of dimension e. This is
        ## then expressed as b observations of t vectors, each of dimension h*e.
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        ## Finally these vectors are all passed to a single linear layer to be compressed down into t vectors.
        return self.unifyheads(out)


## Uses the self attention above with dropout and normalization layers.
class Transformer(nn.Module):
    def __init__(self, emb, heads, ff_mult = 5, p_dropout = 0.1):

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.ff_mult = ff_mult

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_mult * emb, emb))

        self.attention = SelfAttention(emb, heads)

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.dropout1 = nn.Dropout(p = p_dropout)
        self.dropout2 = nn.Dropout(p = p_dropout)

    def forward(self, x, mask):

        attended = self.attention(self.norm1(x), mask)
        attended = x + self.dropout1(attended)

        ## These are called reisudal connections. They are used in the transformer I'm working off of, as they seem to help performance.
        attended = self.ff(self.norm2(attended))
        x = x + self.dropout2(attended)

        return x, mask


## Used to chain transformers together. 
class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


## Transformer chain which takes an input x, mask_x.
class Transformer_Chain(nn.Module):
    def __init__(self, emb, depth, heads, ff_mult = 5):

        super().__init__()

        tblocks = []
        for i in range(depth):
            tblocks.append(Transformer(emb, heads, ff_mult))
        self.tblocks = mySequential(*tblocks)
      
    def forward(self, x, mask):
        x, mask = self.tblocks(x, mask)

        return(x)

        


## A block of transformers (determined by depth) followed by a correction layer. This correction
## layer is meant to output the batch corrections.
class TransformNet(nn.Module):
    def __init__(self, emb, depth, n_batches, batch_size, heads = 5, ff_mult = 5):

        super().__init__()

        ## Networks
        self.transformers = Transformer_Chain(emb, depth, heads, ff_mult)
        self.correction = nn.Sequential(nn.Linear(emb, ff_mult * emb), nn.ReLU(), 
                                        nn.Linear(ff_mult * emb, n_batches))
        
        self.batch_size = batch_size
        
          
    def forward(self, x, mask):

        x = self.transformers(x, mask)

        ## We take only the first output of the transformer when making correction.
        start_token_signal = x[:, 0, :]
        x = self.correction(start_token_signal)
        x = x.repeat_interleave(self.batch_size, 1)

        return x


class TransformNetAvg(nn.Module):
    def __init__(self, emb, seq_length, depth, n_batches, batch_size, heads = 5, ff_mult = 5):

        super().__init__()

        ## Networks
        self.transformers = Transformer_Chain(emb, depth, heads, ff_mult)
        self.correction = nn.Sequential(nn.Linear(emb * seq_length, ff_mult * emb), nn.ReLU(), 
                                        nn.Linear(ff_mult * emb, n_batches))
        
        self.batch_size = batch_size
           
    def forward(self, x, mask):

        x = self.transformers(x, mask)
        ## We take all the output from the transformer when making correction
        x = torch.flatten(x, 1, 2)

        x = self.correction(x)
        x = x.repeat_interleave(self.batch_size, 1)

        return x



class Correction_peptide(nn.Module):
    def __init__(self, emb, depth, CrossTab, n_batches, batch_size, target, test_size, 
                 minibatch_size, random_state, heads = 5, ff_mult = 5):
      
        super().__init__()

        self.CrossTab = CrossTab
        
        ## Data embedding
        self.TRAIN_DATA, self.TEST_DATA, self.FULL_DATA, self.METADATA = make_dataset_transformer(CrossTab = CrossTab, 
                                                                                                  emb = emb, 
                                                                                                  n_batches = n_batches,
                                                                                                  test_size = test_size, 
                                                                                                  random_state = random_state)
        
        self.trainloader = torch.utils.data.DataLoader(self.TRAIN_DATA, shuffle = True, 
                                                       batch_size = minibatch_size)
        self.testloader = torch.utils.data.DataLoader(self.TEST_DATA, shuffle = False, 
                                                      batch_size = minibatch_size)
        self.loader = torch.utils.data.DataLoader(self.FULL_DATA, shuffle = False, 
                                                  batch_size = minibatch_size)

        ## Important self variables
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.target = target
        self.test_n = len(self.TEST_DATA)
        self.train_n = len(self.TRAIN_DATA)
        self.data_n = len(self.FULL_DATA)

        ## The network
        self.network = TransformNet(emb = emb, depth = depth, n_batches = n_batches, 
                                                batch_size = batch_size, heads = heads, ff_mult = ff_mult)
        
        ## The optimizers
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = 1e-4, betas = (0.9, 0.999))

        ## Set the weights of the final layer to zero. This is so that that the inital corrections are all zero.
        self.network.correction[2].weight.data.fill_(0)
        self.network.correction[2].bias.data.fill_(0)

        self.device = torch.device('cuda')
        self.network = self.network.to(self.device)


    ## Distance based on batch effect present in data y.
    def objective_kldiv(self, y):
        length = len(y)
        y_mean = torch.mean(y, 1).view(length, 1).repeat_interleave(self.n_batches * self.batch_size, 1)

        y_batch_mean = y.view(length, self.n_batches, self.batch_size)
        y_batch_mean = torch.mean(y_batch_mean, 2).repeat_interleave(self.batch_size, 1)

        exp_var = torch.sum(torch.square(y_batch_mean - y_mean), 1)
        unexp_var = torch.sum(torch.square(y - y_batch_mean), 1)

        N = self.batch_size * self.n_batches
        K = self.n_batches

        F_stat = (exp_var/unexp_var) * ((N-K) / (K-1))
        p = torch.distributions.FisherSnedecor(df1 = K-1, df2 = N-K)
        log_F = p.log_prob(F_stat)

        loss = torch.sum(log_F - self.target)**2
        return(loss)

    ## Computes mse. y should be 'prediction - data'.
    def objective_mse(self, y):
        loss = torch.sum(y**2) / (self.n_batches * self.batch_size)
        return loss

    def train_model(self, epochs, report_frequency = 50, objective = "batch_correction"):

        train_loss_all = []
        test_loss_all = []
        full_loss_all = []

        if (objective == "batch_correction"):
            objective = self.objective_kldiv
        elif (objective == "peptide_batch_prediction"):
            objective  = self.objective_mse
        else:
            print("Must input a valid objective")

        for epoch in range(epochs):
            if (epoch % report_frequency == 0):
                test_loss = 0
                full_loss = 0
                data_corrected = []
                p_values = []
                
                for x, mask, y, _ in self.testloader:
                    x, mask, y = x.clone().detach().to(self.device), mask.detach().to(self.device), y.clone().detach().to(self.device)
                    loss = objective(y - self.network(x, mask))
                    test_loss += float(loss)

                for x, mask, y, _ in self.loader:
                    x, mask, y = x.clone().detach().to(self.device), mask.detach().to(self.device), y.clone().detach().to(self.device)
                    z = (y - self.network(x, mask)).detach().cpu()
                    p_v = test_batch_effect_fast(z, n_batches = self.n_batches, batch_size = self.batch_size)
                    loss = objective(y - self.network(x, mask))
                    full_loss += float(loss)

                    p_values = np.append(p_values, p_v)
                    data_corrected.append(z)

                test_loss = test_loss / self.test_n
                test_loss_all.append(test_loss)
                full_loss = full_loss / (self.test_n + self.train_n)
                full_loss_all.append(full_loss)
                p_values = torch.tensor(p_values)
                data_corrected = torch.cat(data_corrected).numpy()
                data_corrected = pd.DataFrame(data_corrected)
                
                make_report(data_corrected, p_values, n_batches = self.n_batches, 
                            batch_size = self.batch_size, prefix = "All data", suffix = format(epoch))
                print("Epoch " + format(epoch) + " report : testing loss is " + format(test_loss) + 
                      " while full loss is " + format(full_loss) + "\n")

            training_loss = 0
            for x, mask, y, _ in self.trainloader:
                x, mask, y = x.clone().detach().to(self.device), mask.detach().to(self.device), y.clone().detach().to(self.device)

                self.optimizer.zero_grad()
                loss = objective(y - self.network(x, mask))
                training_loss += float(loss)

                loss.backward()
                self.optimizer.step()

            training_loss = training_loss / (self.train_n)
            train_loss_all.append(training_loss)
            print("Training loss is " + format(training_loss))

            if (epoch % report_frequency == 0 and epoch > 0):
                plot_index = [j * report_frequency for j in range(len(test_loss_all))]
                plt.plot(train_loss_all, label = 'Training loss')
                plt.plot(plot_index, test_loss_all, label = 'Testing loss')
                plt.plot(plot_index, full_loss_all, label = 'Full loss')
                plt.legend()
                plot_title = "All losses epochs " + format(epoch)
                plt.title(plot_title)
                path = "./loss_summaries/" + plot_title + ".png"
                plt.savefig(path)
                plt.clf()

        data_corrected_output = []
        for x, mask, y, _ in self.loader:
            x, mask, y = x.clone().detach().to(self.device), mask.detach().to(self.device), y.clone().detach().to(self.device)
            z = (y - self.network(x, mask)).detach().cpu()

            data_corrected_output.append(z)

        data_corrected_output = torch.cat(data_corrected_output).numpy()
        data_corrected_output = pd.DataFrame(data_corrected_output)
        data_corrected_output.index = self.CrossTab.index
        column_mapping = dict(zip(data_corrected_output.columns, self.CrossTab.columns))
        data_corrected_output = data_corrected_output.rename(columns = column_mapping)
        self.corrected_data = data_corrected_output

    
    def compute_batch_effect(self):
        p_values = []
        for x, mask, y, _ in self.loader:
            x, mask = x.clone().detach().to(self.device), mask.detach().to(self.device)
            z = (y - self.network(x, mask)).detach().cpu()
            p_v = test_batch_effect_fast(z, n_batches = self.n_batches, batch_size = self.batch_size)
            p_values = np.append(p_values, p_v)

        p_values = pd.DataFrame([p_values, self.METADATA['feature_names_og']])
        return(p_values.transpose())


class Correction_data(nn.Module):
    def __init__(self, CrossTab, depth, n_batches, batch_size, target, test_size, 
                 minibatch_size, random_state, heads = 5, ff_mult = 5):
      
        super().__init__()

        self.CrossTab = CrossTab
        
        ## Data embedding
        self.TRAIN_DATA, self.TEST_DATA, self.FULL_DATA, self.METADATA = make_dataset_transformer(CrossTab = CrossTab, 
                                                                                                  emb = 6, 
                                                                                                  n_batches = n_batches,
                                                                                                  test_size = test_size, 
                                                                                                  random_state = random_state)
        
        self.trainloader = torch.utils.data.DataLoader(self.TRAIN_DATA, shuffle = True, 
                                                       batch_size = minibatch_size)
        self.testloader = torch.utils.data.DataLoader(self.TEST_DATA, shuffle = False, 
                                                      batch_size = minibatch_size)
        self.loader = torch.utils.data.DataLoader(self.FULL_DATA, shuffle = False, 
                                                  batch_size = minibatch_size)

        ## Important self variables
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.target = target
        self.test_n = len(self.TEST_DATA)
        self.train_n = len(self.TRAIN_DATA)
        self.data_n = len(self.FULL_DATA)

        ## The network
        self.network = TransformNetAvg(emb = self.batch_size, seq_length = self.n_batches, depth = depth,
                                       n_batches = self.n_batches, batch_size = self.batch_size)
        
        ## The optimizers
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = 1e-4, betas = (0.9, 0.999))

        ## Set the weights of the final layer to zero. This is so that that the inital corrections are all zero.
        self.network.correction[2].weight.data.fill_(0)
        self.network.correction[2].bias.data.fill_(0)

        self.device = torch.device('cuda')
        self.network = self.network.to(self.device)


    ## Distance based on batch effect present in data y.
    def objective_kldiv(self, y):
        length = len(y)
        y_mean = torch.mean(y, 1).view(length, 1).repeat_interleave(self.n_batches * self.batch_size, 1)

        y_batch_mean = y.view(length, self.n_batches, self.batch_size)
        y_batch_mean = torch.mean(y_batch_mean, 2).repeat_interleave(self.batch_size, 1)

        exp_var = torch.sum(torch.square(y_batch_mean - y_mean), 1)
        unexp_var = torch.sum(torch.square(y - y_batch_mean), 1)

        N = self.batch_size * self.n_batches
        K = self.n_batches

        F_stat = (exp_var/unexp_var) * ((N-K) / (K-1))
        p = torch.distributions.FisherSnedecor(df1 = K-1, df2 = N-K)
        log_F = p.log_prob(F_stat)

        loss = torch.sum(log_F - self.target)**2
        return(loss)


    def train_model(self, epochs, report_frequency = 50, objective = "batch_correction"):

        train_loss_all = []
        test_loss_all = []
        full_loss_all = []

        if (objective == "batch_correction"):
            objective = self.objective_kldiv
        elif (objective == "peptide_batch_prediction"):
            objective  = self.objective_mse
        else:
            print("Must input a valid objective")

        for epoch in range(epochs):
            if (epoch % report_frequency == 0):
                test_loss = 0
                full_loss = 0
                data_corrected = []
                p_values = []
                
                for _, _, y, mask in self.testloader:
                    y, mask = y.clone().detach().to(self.device), mask.detach().to(self.device)
                    n = len(y)
                    x = y.reshape(n, self.n_batches, self.batch_size).float()
                    loss = objective(y - self.network(x, mask))
                    test_loss += float(loss)

                for _, _, y, mask in self.loader:
                    y, mask = y.clone().detach().to(self.device), mask.detach().to(self.device)
                    n = len(y)
                    x = y.reshape(n, self.n_batches, self.batch_size).float()
                    z = (y - self.network(x, mask)).detach().cpu()
                    p_v = test_batch_effect_fast(z, n_batches = self.n_batches, batch_size = self.batch_size)
                    loss = objective(y - self.network(x, mask))
                    full_loss += float(loss)

                    p_values = np.append(p_values, p_v)
                    data_corrected.append(z)

                test_loss = test_loss / self.test_n
                test_loss_all.append(test_loss)
                full_loss = full_loss / (self.test_n + self.train_n)
                full_loss_all.append(full_loss)
                p_values = torch.tensor(p_values)
                data_corrected = torch.cat(data_corrected).numpy()
                data_corrected = pd.DataFrame(data_corrected)
                
                make_report(data_corrected, p_values, n_batches = self.n_batches, 
                            batch_size = self.batch_size, prefix = "All data", suffix = format(epoch))
                print("Epoch " + format(epoch) + " report : testing loss is " + format(test_loss) + 
                      " while full loss is " + format(full_loss) + "\n")

            training_loss = 0
            for _, _, y, mask in self.trainloader:
                y, mask = y.clone().detach().to(self.device), mask.detach().to(self.device)
                n = len(y)
                x = y.reshape(n, self.n_batches, self.batch_size).float()

                self.optimizer.zero_grad()
                loss = objective(y - self.network(x, mask))
                training_loss += float(loss)

                loss.backward()
                self.optimizer.step()

            training_loss = training_loss / (self.train_n)
            train_loss_all.append(training_loss)
            print("Training loss is " + format(training_loss))

            if (epoch % report_frequency == 0 and epoch > 0):
                plot_index = [j * report_frequency for j in range(len(test_loss_all))]
                plt.plot(train_loss_all, label = 'Training loss')
                plt.plot(plot_index, test_loss_all, label = 'Testing loss')
                plt.plot(plot_index, full_loss_all, label = 'Full loss')
                plt.legend()
                plot_title = "All losses epochs " + format(epoch)
                plt.title(plot_title)
                path = "./loss_summaries/" + plot_title + ".png"
                plt.savefig(path)
                plt.clf()

        data_corrected_output = []
        for _, _, y, mask in self.loader:
            y, mask = y.clone().detach().to(self.device), mask.detach().to(self.device)
            n = len(y)
            x = y.reshape(n, self.n_batches, self.batch_size).float()
            z = (y - self.network(x, mask)).detach().cpu()

            data_corrected_output.append(z)

        data_corrected_output = torch.cat(data_corrected_output).numpy()
        data_corrected_output = pd.DataFrame(data_corrected_output)
        data_corrected_output.index = self.CrossTab.index
        column_mapping = dict(zip(data_corrected_output.columns, self.CrossTab.columns))
        data_corrected_output = data_corrected_output.rename(columns = column_mapping)
        self.corrected_data = data_corrected_output


    def compute_batch_effect(self):
        p_values = []
        for _, _, y, mask in self.loader:
            y, mask = y.clone().detach().to(self.device), mask.detach().to(self.device)
            n = len(y)
            x = y.reshape(n, self.n_batches, self.batch_size).float()
            z = (y - self.network(x, mask)).detach().cpu()
            p_v = test_batch_effect_fast(z, n_batches = self.n_batches, batch_size = self.batch_size)
            p_values = np.append(p_values, p_v)

        p_values = pd.DataFrame([p_values, self.METADATA['feature_names_og']])
        return(p_values.transpose())




######################################
## Testing masking in transformer chain
# testing = Transformer_Chain(3, 3)
#
# mask1 = torch.tensor([mask_helper(3, 3)])
# mask2 = torch.tensor([mask_helper(3, 6)])
#
# input = torch.rand((1, 6, 3))
#
# testing(input[:,0:3,:], mask1)
# testing(input, mask2)

