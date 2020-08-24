import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_function(recon_x, x, mu, logvar):
    BCE_loss = nn.BCELoss(reduction='sum')
    reconstruction_loss = BCE_loss(recon_x, x)
    KL_divergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu ** 2)

    return reconstruction_loss, KL_divergence

class VAEFC(nn.Module):
    def __init__(self, z_dim):
        super(VAEFC, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2_mean = nn.Linear(400, z_dim)
        self.fc2_logvar = nn.Linear(400, z_dim)
        self.fc3 = nn.Linear(z_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1)

    def reparametrization(self, mu, logvar):
        std = 0.5 * torch.exp(logvar)
        z = torch.randn(std.size()) * std + mu
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        return self.decode(z), mu, logvar

class VAEFCN(nn.Module):
    def __init__(self, z_dim, ch, ks): # z_dim: latent variable dim, ch: channel, ks: kernel size
        super(VAEFCN, self).__init__()
        self.conv1 = nn.Conv2d(1, ch, kernel_size=(ks,ks), stride=2, padding=0)
        self.conv2_mean = nn.Conv2d(ch, z_dim, kernel_size=(1,1), stride=1, padding=0)
        self.conv2_logvar = nn.Conv2d(ch, z_dim, kernel_size=(1,1), stride=1, padding=0)
        self.conv3 = nn.ConvTranspose2d(z_dim, ch, kernel_size=(1,1), stride=1, padding=0)
        self.conv4 = nn.ConvTranspose2d(ch, 1, kernel_size=(ks,ks), stride=2, padding=0)

    def encode(self, x):
        h1 = F.relu(self.conv1(x))
        return self.conv2_mean(h1), self.conv2_logvar(h1)

    def reparametrization(self, mu, logvar):
        std = 0.5 * torch.exp(logvar)
        z = torch.randn(std.size()) * std + mu
        return z

    def decode(self, z):
        h3 = F.relu(self.conv3(z))
        return torch.sigmoid(self.conv4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        return self.decode(z), mu, logvar