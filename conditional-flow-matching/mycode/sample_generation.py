

import math
import os
import time
import sys

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..'))

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

savedir = "models/gaussian-moons"
model = torch.load(f"{savedir}/sbcfm_v2.pt", weights_only=False)
node = NeuralODE(
            torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
with torch.no_grad():
    traj = node.trajectory(
        torch.randn(1024,2),
        t_span=torch.linspace(0, 1, 2, device=device),
    )

original = traj[0].numpy()
samples = traj[1].numpy()

fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].scatter(original[:,0], original[:,1])
ax[1].scatter(samples[:,0], samples[:,1])
plt.show()