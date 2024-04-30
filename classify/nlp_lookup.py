import os
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import torch
from classify.classify import apply_crf
from feats import feature_size, green, plot_lang_heatmaps
from train import MapDataset



@torch.no_grad()
def plot_heatmaps(model, rgb, features, chair_mask, text_query, filename):
    assert len(features.shape) == 3
    
    from feats import tokenize, remove_axes
    import torch.nn.functional as F

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    cmap = plt.get_cmap("turbo")

    # encode query
    text = tokenize([text_query, 'wall', 'floor', 'background']).to(device)
    text_feats = model.model.encode_text(text).squeeze().to(torch.float32)
    
    # print('hr_feats shape', features.shape)
  
    hr_sims = torch.einsum("chw,bc->bhw", 
        F.normalize(features.to(torch.float32), dim=0), 
        F.normalize(text_feats, dim=1))
    
    rgb = rgb.cpu().numpy()
    rgb = (rgb.copy(order='C') * 256).astype(np.uint8)
    
    black_mask = np.all(rgb == 0, axis=-1)
    rgb[black_mask] = 128
    # print('pre max', hr_sims.amin(dim=[1, 2], keepdim=True), hr_sims.amax(dim=[1, 2], keepdim=True) )
    # print(hr_sims[:, ~black_mask].mean())
    hr_sims[:, black_mask] = hr_sims[:, ~black_mask].mean() # Range is .20 - .36
    # print('post max', hr_sims.amin(dim=[1, 2], keepdim=True), hr_sims.amax(dim=[1, 2], keepdim=True) )

    hr_sims_min = hr_sims.amin(dim=[1, 2], keepdim=True)  
    hr_sims_max = hr_sims.amax(dim=[1, 2], keepdim=True) 
    
    hr_sims_norm = (hr_sims - hr_sims_min) / (hr_sims_max - hr_sims_min)
        
    # features.reshape(1, -1).cpu().numpy()
    hr_sims_norm = apply_crf(rgb, hr_sims_norm.cpu().numpy(), num_classes=4)
    hr_sims_norm[black_mask] = 1
    
    ax[0].set_title("RGB")
    ax[0].imshow(rgb)
    ax[1].set_title(f"Features")
    ax[1].imshow(hr_sims_norm, cmap=matplotlib.colormaps['rainbow'])
    ax[2].set_title(f"Chair")
    ax[2].imshow(chair_mask.cpu())
    remove_axes(ax)
    
    name, _ = os.path.splitext(filename)
    plt.savefig(f'maps/{name}_distance_{text_query}.png')
    plt.close()
    green(f'maps/{name}_distance_{text_query}.png')



dataset = MapDataset('maps', train=False)

device = 'cuda'

green('Setup Model')
upsampler = torch.hub.load("mhamilton723/FeatUp", 'maskclip', use_norm=False).to(device)
green('   ...Done')

text_queries = 'chair'


for map, distance, labels, rgb, filename in dataset:
    
    labels
    
    plot_heatmaps(upsampler.model, 
                  rgb.to(device),
                  map.to(device),
                  chair_mask.to(device),
                  text_queries,
                  filename)