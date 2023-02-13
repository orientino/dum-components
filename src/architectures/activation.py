from typing import Literal
import torch

ActivationType = Literal[
    "relu",
    "gelu",
    "silu",
    "mish",
    "l-matern",
    "p-sin",
    "p-sincos",
    "p-triangle",
    "p-relu",
]

def matern(x, nu_ind=1, ell=0.5):
    """Implements the Matern activation function denoted as sigma(x) in Equation 9.
    sigma(x) corresponds to a Matern kernel, with specified smoothness
    parameter nu and length-scale parameter ell.

    This gives the local statiornarity property.
    
    Args:
      x: Input to the activation function
      nu_ind: Index for choosing Matern smoothness (look at nu_list below)
      ell: Matern length-scale, only 0.5 and 1 available with precalculated scaling coefficients
    """
    nu_list = [1/2, 3/2, 5/2, 7/2, 9/2] # list of available smoothness parameters
    nu = torch.tensor(nu_list[nu_ind])  # smoothness parameter
    lamb =  torch.sqrt(2*nu)/ell        # lambda parameter
    v = nu+1/2

    # Precalculated scaling coefficients for two different lengthscales (q divided by Gammafunction at nu + 0.5)
    ell05A = [4.0, 19.595917942265423, 65.31972647421809, 176.69358285524189, 413.0710073859664]
    ell1A = [2.0, 4.898979485566356, 8.16496580927726, 11.043348928452618, 12.90846898081145]

    if ell == 0.5:
        A = torch.tensor(ell05A[nu_ind])
    if ell == 1:
        A = torch.tensor(ell1A[nu_ind])

    y = A*torch.sign(x)*torch.abs(x)**(v-1)*torch.exp(-lamb*torch.abs(x))
    y[x<0] = 0

    return y

torch.pi = 3.1415926535897932
torch.pi2 = 6.2831853071795864
torch.sqrt2 = 1.414213562           # sqrt 2 \approx 1.414213562
torch.pdiv2sqrt2 = 1.1107207345     # π/(2*sqrt(2)) \approx 1.1107207345
torch.pdiv2 = 1.570796326           # π/2
torch.pdiv4 = 0.785398163           # π/4 


def sin(x):
    return torch.sqrt2*torch.sin(x)

def sincos(x):
    return torch.sin(x) + torch.cos(x)

def triangle(x):
    return torch.pdiv2sqrt2 * _triangle(x)

def periodic_relu(x):
    return torch.pdiv4 * (_triangle(x) + _triangle(x + torch.pdiv2))

def _triangle(x):
    return (x - torch.pi * torch.floor(x / torch.pi + 0.5)) * (-1)**torch.floor(x/torch.pi + 0.5)
