import numpy as np
import torch
import math
import matplotlib.pyplot as plt

'''
Four 2-d potential functions, U(x), are defined.
The corresponding probability densities are defined as \log P(x) \propto \exp(-U(x))

'''

def compute_U1(z):
    mask = torch.ones_like(z)
    mask[:, 1] = 0.0

    U = (0.5*((torch.norm(z, dim = -1) - 2)/0.4)**2 - \
             torch.sum(mask*torch.log(torch.exp(-0.5*((z - 2)/0.6)**2) +
                                      torch.exp(-0.5*((z + 2)/0.6)**2)), -1))    
    return U

def compute_U2(z):
    w1 = torch.sin(2*math.pi*z[:,0]/4)
    U = 0.5*((z[:,1] - w1)/0.4)**2    
    return U

def compute_U3(z):
    w1 = torch.sin(2*math.pi*z[:,0]/4)
    w2 = 3*torch.exp(-0.5*((z[:,0] - 1)/0.6)**2)
    U = -torch.log(torch.exp(-0.5*((z[:,1] - w1)/0.35)**2) + torch.exp(-0.5*((z[:,1] - w1 + w2)/0.35)**2))
    return U
    
def compute_U4(z):
    w1 = torch.sin(2*math.pi*z[:,0]/4)
    w3 = 3*torch.sigmoid((z[:,0] - 1)/0.3)    
    U = -torch.log(torch.exp(-0.5*((z[:,1] - w1)/0.35)**2) + torch.exp(-0.5*((z[:,1] - w1 + w3)/0.35)**2))
    return U

def _generate_grid():
    z1 = torch.linspace(-4., 4., steps = 100)
    z2 = torch.linspace(-4., 4., steps = 100)
    grid_z1, grid_z2 = torch.meshgrid(z1, z2)
    grid = torch.stack([grid_z1, grid_z2], dim = -1)
    z = grid.reshape((-1, 2))
    return z
    
def normalize_potential(U):
    U = U - U.min()  # Shift potential to avoid overflow
    return torch.exp(-U)

def test_U():
    z = _generate_grid()
    
    fig = plt.figure(0, figsize=(12.8, 4.8))
    fig.clf()
    
    plt.subplot(2, 4, 1)
    U1 = compute_U1(z)
    U1 = U1.reshape(100, 100).T   
    p1 = normalize_potential(U1)
    plt.contourf(p1.numpy(), levels=50, cmap='viridis')
    plt.title("Potential U1")

    plt.subplot(2, 4, 2)
    U2 = compute_U2(z)
    U2 = U2.reshape(100, 100).T    
    p2 = normalize_potential(U2)
    plt.contourf(p2.numpy(), levels=50, cmap='viridis')
    plt.title("Potential U2")

    plt.subplot(2, 4, 3)
    U3 = compute_U3(z)
    U3 = U3.reshape(100, 100).T   
    p3 = normalize_potential(U3)
    plt.contourf(p3.numpy(), levels=50, cmap='viridis')
    plt.title("Potential U3")

    plt.subplot(2, 4, 4)
    U4 = compute_U4(z)
    U4 = U4.reshape(100, 100).T   
    p4 = normalize_potential(U4)
    plt.contourf(p4.numpy(), levels=50, cmap='viridis')
    plt.title("Potential U4")

    plt.show()

if __name__ == "__main__":
    test_U()