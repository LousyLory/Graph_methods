# graph variational autoencoder
# implementation using pytorch

import torch
from torch import nn, optim
import networkx
from GCN import GraphConvolution

class VGAE_linear(nn.Module):
	'''
	Network class for VGAE linear architectures
	'''
	def __init__(self, input_size):
		super(GVAE, self).__init__()
		# set up the layers
		self.fc1 = nn.Linear(input_size, 400)
		self.fc21 = nn.Linear(400, 20)
		self.fc22 = nn.Linear(400, 20)
		self.fc3 = nn.Linear(20, 400)
		self.fc4 = nn.Linear(400, input_size)

	def encode(self, x):
		h1 = F.relu(self.fc1(x))
		return self.fc21(h1), self.fc22(h1)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu+eps*std

	def decode(self, z):
		h3 = F.relu(self.fc3(z))
		return torch.sigmoid(self.fc4(h3))

	def forward(self, x):
		mu, logvar = self.encode(x.view(-1, input_size))
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar

class VGAE_GCN_IP(nn.Module):
	'''
	Network class for VGAE GCN architectures
	'''
	def __init__(self, input_size):
		super(GVAE, self).__init__()
        self.gc1 = GC(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GC(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GC(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
                std = torch.exp(0.5*logvar)
                eps = torch.randn_like(std)
                return eps.mul(std).add_(mu)
        else:
                return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar

class InnerProductDecoder(nn.Module):
    '''
    Decoder for using inner product for prediction
    '''
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class RBFKernelDecoder(nn.Module)
	'''
    Decoder for using inner product for prediction
    '''
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        # adj = exp(norm(z,z)^2/2*sigma^2)
        #adj = self.act(torch.mm(z, z.t()))
        adj = self.act()
        return adj


class PolyKernelDecoder(nn.Module):
	'''
    Decoder for using inner product for prediction
    '''
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        # need to explain c and d here
        # adj = (torch.mm(z, z.t())+c)^d
        #adj = self.act(torch.mm(z, z.t()))
        adj = self.act((torch.mm(z, z.t()) + c)**d)
        return adj