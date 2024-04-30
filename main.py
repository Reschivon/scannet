
import os
from matplotlib import pyplot as plt
# Make plt respoonsive to ctrl+c
import signal

import torch
import torch.nn.functional
signal.signal(signal.SIGINT, signal.SIG_DFL)

import numpy as np
from sklearn.decomposition import PCA

from datasets import ScannetDataset, chair_ids, most_common_25_ids
from projection import ProjectAndFlatten
from dinov2 import feature_size



def make_map_for(scene_path):
    bin_width = 0.02
    
    dataset = ScannetDataset(path=scene_path)
    
    from dinov2 import infer as get_features

    mapping = ProjectAndFlatten(project_func=lambda image: get_features(image), bin_width=bin_width)

    for i, (intrinsics, pose, color, depth, label) in enumerate(dataset):      
        print(i, '/', len(dataset))
                
        chair_label = sum(label==i for i in chair_ids).bool()
        common_label = sum((label == i) * i for i in most_common_25_ids).int()
        
        # plt.imshow(common_label)
        # plt.show()
        
        mapping.project(pose, color, depth, chair_label, common_label, intrinsics)
        
    map, height = mapping.flatten()

    torch.save((map, height), 'data/last_map.pth')
    
    
    
    

    map, height = torch.load('data/last_map.pth')
    
    map_rgb = map[..., feature_size():feature_size() + 3]
    map_feature = map[..., 0:feature_size()]
    map_lab = map[..., feature_size() + 3: feature_size() + 4]
    map_lab_25 = map[..., feature_size() + 4: ]
    
    # Generate distance maps to chair
    chair_mask = map_lab
        
    if not torch.any(chair_mask):
        print('Skipping,', scene_path, 'since no chair detected')
        return
    
    # plt.imshow(map_rgb, cmap='viridis')
    # plt.colorbar()
    # plt.title('Distance from True Pixels')
    # plt.show()

    # plt.imshow(height, interpolation='none')
        
    # plt.gcf().set_size_inches(7, 7)
    # plt.axis('off')
    # plt.savefig('data/last_projection.png')
    # plt.show()
    
    
    import skfmm
    
    # Calculate the distance using scikit-fmm, inside the mask (phi positive outside, negative inside)
    phi = np.where(chair_mask, -1, 1)
    distance = skfmm.distance(phi) * bin_width 
    distance[distance > 3] = 3
    
    torch.save({'map_rgb': map_rgb, 
                'map_feature': map_feature, 
                'map_lab': map_lab, 
                'map_lab_25': map_lab_25,
                'height': height,
                'chair_mask': chair_mask, 
                'distance': distance}, 
                'maps/' + scene_path + '.pth')

    # Maxpool features or highest-Z?
    # Maxpool retains the most features, but highest-z makes 
    # it easier to memorize the embedding for objects

    # 3rd option take the longest embedding and discard others.
    # The same surface may been seen multiple times, but from
    # different views. the network may pick up different features
    # for the same thing. We take only the strongest feature.
    
    # plot3d(stack, pca_features, voxel_size=0.02) # 5 cm

if __name__ == '__main__':
    
    for dir in sorted(os.listdir('.')):
        
        if not 'scene' in dir:
            continue
        
        name = dir + '.pth'
        if name in sorted(os.listdir('maps')):
            continue
        else:
            print(name, 'not in maps')
        
        if 'scene0007_00' in dir or 'scene0012_00' in dir or 'scene0009_02' in dir or 'scene0003_01' in dir or\
            'scene0003_00' in dir or 'scene0003_02' in dir or 'scene0009_00' in dir or 'scene0009_01' in dir :
            continue
        
        # if dir not in ['scene0008_00_data.pth', 'scene0011_01_data.pth', 'scene0013_02_data.pth', 'scene0017_02_data.pth', 'scene0019_00_data.pth',
        #                'scene0020_01_data.pth', 'scene0025_02_data.pth']:
        #     continue
        
        print('Make map for', dir)
        make_map_for(dir)