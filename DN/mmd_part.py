# [source] https://github.com/pratikm141/MMD-Variational-Autoencoder-Pytorch-InfoVAE

def add_args(parser):
  #######################
  group = parser.add_argument_group('Training parameters')
  group.add_argument('--KLD-method', default='mmd', choices=('elbo','mmd'), help='elbo (in cryodrgn denoted as just kld) vs mmd')



############# MMD (maximum mean discrepancy) Loss function
def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]
    tiled_x = x.view(x_size,1,dim).repeat(1, y_size,1)
    tiled_y = y.view(1,y_size,dim).repeat(x_size, 1,1)
    return torch.exp(-torch.mean((tiled_x - tiled_y)**2,dim=2)/dim*1.0)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)
## end of def compute_mmd(x, y):



def prepare_true_samples(args, encoder_entered_y):
  encoder_entered_y = Variable(encoder_entered_y) 
  # Variable wraps a Tensor and provides a backward method to perform backpropagation

  true_samples = torch.randn((len(encoder_entered_y),args.zdim))
  true_samples = Variable(true_samples)
  
  #true_samples = true_samples.cuda()
  true_samples = true_samples.to(args.device)

  #print ("true_samples.shape:" + str(true_samples.shape)) # torch.Size([3, 8])
  #print ("true_samples:" + str(true_samples))
  '''tensor([[ 0.2077, -0.6472,  1.0071, -0.8504, -0.3179, -0.3596, -0.5218,  0.9012],
        [ 0.1351,  1.1203,  0.1449,  1.3563,  0.9140, -2.2697,  0.7881, -0.4880],
        [ 0.1563,  0.7140,  0.0122,  0.5233,  1.0259, -0.6711, -0.8832, -1.0707]],
       device='cuda:0')'''
  return true_samples
## end of def prepare_true_samples(args, encoder_entered_y):


def main(args):
    z_mu, z_logvar = _unparallelize(model, args).encode(*input_)
    z = _unparallelize(model, args).reparameterize(z_mu, z_logvar)

    # latent loss
    if (args.KLD_method == 'mmd'):
      #encoder_entered_y_ori_length is nothing but D X D pixellated image information
      true_samples = prepare_true_samples(args, encoder_entered_y_ori_length)
      KLD = compute_mmd(true_samples, z) # mmd
    else:
      KLD = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1), dim=0)
      #elbo (in cryodrgn this is denoted as kld)
      # dim = 0 collapses row values (so leaving column wise values only)
      # dim = 1 collapses column values (so leaving row values only)



if __name__ == '__main__':
  use_cuda = torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  
  parser = argparse.ArgumentParser(description=__doc__)
  args = add_args(parser).parse_args()
  
  args.device = device
  main(args)

