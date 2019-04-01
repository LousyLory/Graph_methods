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