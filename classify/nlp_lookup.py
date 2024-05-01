import argparse
import os
import traceback
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train import MapDataset
from feats import tokenize, remove_axes, green
from classify.classify import apply_crf

THRESH = 1.3

@torch.no_grad()
def generate_heatmap(model, rgb, features, gt_mask, text_query, filename):
    assert len(features.shape) == 3
    

    # Text encoder
    text = tokenize([text_query, 'wall', 'floor', 'background']).to(device)
    text_feats = model.model.encode_text(text).squeeze().to(torch.float32).cpu()
        
    # Take dot product of text and vision embeddings
    alignments = torch.einsum("chw,bc->bhw", 
        F.normalize(features.to(torch.float32), dim=0), 
        F.normalize(text_feats, dim=1))
    
    # RGB（Tensor 0-1 float）--> RGB（numpy 0-256 int)
    rgb = rgb.cpu().numpy()
    rgb = (rgb.copy(order='C') * 256).astype(np.uint8)
    
    no_data_mask = np.all(rgb == 0, axis=-1)
    rgb[no_data_mask] = 128
    
    # Set no data to mean, to not screw up the min/max step
    alignments[:, no_data_mask] = alignments[:, ~no_data_mask].mean() # Range is .20 - .36
    
    # Normalize to 0-1
    # heatmap_min = alignments.amin(dim=[1, 2], keepdim=True)  
    # heatmap_max = alignments.amax(dim=[1, 2], keepdim=True) 
    heatmap_min = 0.15
    heatmap_max = 0.40
        
    heatmap = (alignments - heatmap_min) / (heatmap_max - heatmap_min)
    heatmap = torch.clamp(heatmap, 0, 1)
    
    # Apply CRF with RGB as pointwise energy
    heatmap = apply_crf(rgb, heatmap.cpu().numpy(), num_classes=4)
    
    # Finally, for better vis, set no data to zero
    heatmap[no_data_mask] = 0
    
    
    
    # print(heatmap.min(), heatmap.max())
    
    # Show RGB, Heatmap, and GT Maask
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    ax[0].set_title("RGB")
    ax[0].imshow(rgb)
    
    ax[1].set_title(f"Heatmap")
    ax[1].imshow(heatmap, cmap=matplotlib.colormaps['YlOrRd'])
    
    ax[2].set_title(f"Heatmap Thresh")
    ax[2].imshow(heatmap > THRESH / 2, cmap=matplotlib.colormaps['YlOrRd'])
    
    ax[3].set_title(f"GT Mask for {text_query}")
    ax[3].imshow(gt_mask.cpu())
    
    remove_axes(ax)
    
    # Save the PNG figure, and the raw data
    name, _ = os.path.splitext(filename)
    green(f'maps/{name}_distance_{text_query}.png')
    
    plt.savefig(f'maps/{name}_distance_{text_query}.png')
    plt.close()
    
    
    
    
    ####################
    # Compute Metrics
    ####################
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    
    thresholded_heatmap = (heatmap[~no_data_mask] > THRESH / 2).astype(int).flatten()
    gt_mask_numpy = (gt_mask[~no_data_mask] > 0.5).cpu().numpy().astype(int).flatten()
    
    
    bce_loss = F.binary_cross_entropy(torch.tensor(heatmap), gt_mask.float(), reduction='mean').item()
    f1 = f1_score(gt_mask_numpy, thresholded_heatmap)
    accuracy = accuracy_score(gt_mask_numpy, thresholded_heatmap)
    precision = precision_score(gt_mask_numpy, thresholded_heatmap)
    recall = recall_score(gt_mask_numpy, thresholded_heatmap)
    
    tn, fp, fn, tp = confusion_matrix(gt_mask_numpy, thresholded_heatmap).ravel()
    specificity = tn / (tn + fp)
    
    save_data = {'heatmap': heatmap, 'gt': gt_mask_numpy, 'gtmask': gt_mask, 'no_data_mask': no_data_mask,
                 'metrics': {
                    'BCE': bce_loss, 'F1': f1, 'Accuracy': accuracy, 'Precision': precision, 
                    'Recall': recall, 'Specificity': specificity
                }}
    
    print(save_data['metrics'])
    torch.save(save_data, f'maps/{name}_result_{text_query}.pth')

from datasets import name_to_id, id_to_name

if __name__ == '__main__':
    device = 'cuda'

    dataset = MapDataset('maps', train=False)
    dataloader = DataLoader(dataset, num_workers=3)

    green('Setup Model')
    upsampler = torch.hub.load("mhamilton723/FeatUp", 'maskclip', use_norm=False).to(device)
    green('   ...Done')


    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id', default=3, type=int, help='The name of the scene to start processing from.')
    args = parser.parse_args()

    id = args.id

    for map, distance, labels, rgb, filename in dataloader:
        
        map = map[0]
        distance = distance[0]
        labels = labels[0]
        rgb = rgb[0]
        filename = filename[0]
        
        # print(map.shape, distance.shape, labels.shape, rgb.shape, filename)
        
        # if filename != 'scene0009_00_data.pth':
        #     continue
        
        green('Querying up ' + str(filename))

        
        query = id_to_name[id]
        id = id
        
        gt_labels = labels[..., id] > 0
        
        try:
            generate_heatmap(upsampler.model, 
                        rgb.cpu(),
                        map.cpu(),
                        gt_labels.cpu(),
                        query,
                        filename)
        except Exception as e:
            green('Error ' + str(e))
            print(traceback.format_exc())
            pass