import torch
import numpy as np

# load from numpy file
mask_array = np.load('mask.npy')

mask_tensor = torch.load('mask.pt')

embeds = torch.load('embeds.pt')
embeds = embeds.cuda()

embeds[mask_tensor]



