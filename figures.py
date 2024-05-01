from matplotlib import pyplot as plt
import torch
import os
import numpy as np

from feats import green

# Assuming the saved results are in the 'maps' directory
directory = 'maps'

# Storage for metrics and areas
metrics = {}
area_predictions = {}
area_gts = {}

index = 0
# Iterate through each file in the directory
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".pth") and 'result' in filename:
        index += 1
        if index == 20: break
        
        # Load the data
        data = torch.load(os.path.join(directory, filename))
        
        # Extracting the name and id from filename
        name_id = filename.split('_')[-1].split('.')[0]
        
        green(f'Loaded {filename} of type {name_id}')
        
        # Metrics aggregation
        if name_id not in metrics:
            metrics[name_id] = {
                'BCE': [], 'F1': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'Specificity': []
            }
            
        try:
            # Append each metric for the current id
            metrics[name_id]['BCE'].append(data['metrics']['BCE'])
            metrics[name_id]['F1'].append(data['metrics']['F1'])
            metrics[name_id]['Accuracy'].append(data['metrics']['Accuracy'])
            metrics[name_id]['Precision'].append(data['metrics']['Precision'])
            metrics[name_id]['Recall'].append(data['metrics']['Recall'])
            metrics[name_id]['Specificity'].append(data['metrics']['Specificity'])
        except:
            continue
        
        # Compute area ratios
        if name_id not in area_predictions:
            area_predictions[name_id] = []
        if name_id not in area_gts:
            area_gts[name_id] = []
        
        from classify.nlp_lookup import THRESH
        from datasets import name_to_id
        
        # Get binary heatmap and ground truth mask
        binary_heatmap = data['heatmap'] > (THRESH / 2)
        # gt_mask = data['gt'] > 0.5
        
        if 'gtmask' in data:
            gt_mask = data['gtmask'].numpy()
        else:
            bigfile = f'maps/{filename.split("_")[0]}_{filename.split("_")[1]}_data.pth'
            map_lab = torch.load(bigfile)['map_lab'].squeeze()
            gt_mask = map_lab[..., name_to_id[name_id]] > 0
            gt_mask = gt_mask.numpy()
        
        # Valid mask areas (ignoring no_data_mask regions)
        valid_mask = data['heatmap'] > 0
        
        # print(binary_heatmap.shape, gt_mask.shape, valid_mask.shape)
        # print(binary_heatmap.dtype, gt_mask.dtype, valid_mask.dtype)
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(binary_heatmap)
        ax[0].set_title("binary_heatmap")
        ax[1].imshow(gt_mask)
        ax[1].set_title("gt_mask")
        ax[2].imshow(valid_mask)
        ax[2].set_title("valid_mask")
        plt.savefig('random/test.png')
        plt.close()
        
        # Calculate percentage of area covered by predictions vs ground truth
        area_prediction = np.sum(binary_heatmap & valid_mask) / np.sum(valid_mask)
        area_gt = np.sum(gt_mask & valid_mask) / np.sum(valid_mask)
        
        area_predictions[name_id].append(area_prediction)
        area_gts[name_id].append(area_gt)

# Compute averages for metrics and areas
average_metrics = {}
average_areasp = {}
average_areasg = {}

for id, vals in metrics.items():
    average_metrics[id] = {k: np.nanmean(v) for k, v in vals.items()}
    
for id, vals in area_prediction.items():
    average_areasp[id] = np.nanmean(vals)
    
for id, vals in area_gts.items():
    average_areasg[id] = np.nanmean(vals)

print(average_metrics) 
print(average_areasp)
print(average_areasg)

with open("average_metrics.txt", "w") as text_file:
    text_file.write(str(average_metrics))
    
with open("average_areasp.txt", "w") as text_file:
    text_file.write(str(average_areasp))
    
with open("average_areasg.txt", "w") as text_file:
    text_file.write(str(average_areasg))