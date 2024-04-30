
import argparse
import os
from matplotlib import pyplot as plt
# Make plt respoonsive to ctrl+c
import signal

import torch
import torch.nn.functional
from tqdm import tqdm

# from dinov2 import pca
signal.signal(signal.SIGINT, signal.SIG_DFL)

import numpy as np

from datasets import ScannetDataset
from projection import ProjectAndFlatten
from feats import feature_size, green

from PIL import Image

def make_map_for(scene_path):
    bin_width = 0.02
    
    dataset = ScannetDataset(path='scenes/' + scene_path)
    
    from feats import infer as get_features

    mapping = ProjectAndFlatten(project_func=lambda image: get_features(image), bin_width=bin_width)

    for i, (intrinsics, pose, color, depth, label) in enumerate(tqdm(dataset, dynamic_ncols=True)):      
        # print(i, '/', len(dataset))
                        
        # Convert label from 944x944x1 array to one-hot of 944x944x40
        label_one_hot = torch.nn.functional.one_hot(label.long(), num_classes=40)
        label_one_hot = label_one_hot.squeeze(-2)  # Remove the extra singleton dimension if present
                
        mapping.project(pose, color, depth, label_one_hot, intrinsics)
        
    map, height = mapping.flatten()

    torch.save((map, height), 'random/last_map.pth')
    
    
    
    

    map, height = torch.load('random/last_map.pth')
        
    map_feature = map[..., 0:feature_size()]
    map_rgb = map[..., feature_size():feature_size() + 3]
    map_lab = map[..., feature_size() + 3:]    
    
    plt.imshow(map_rgb)
    plt.savefig('maps/' + scene_path + '_rgb.png')
    plt.close()
        
    from feats import pca
    [hr_feats_pca], _ = pca([map_feature.permute(2, 0, 1).unsqueeze(0)])
    # hr_feats_pca = pca(map_feature)
    plt.imshow(hr_feats_pca[0].permute(1, 2, 0))
    plt.savefig('maps/' + scene_path + '_feature.png')
    plt.close()
    
    plt.imshow(torch.argmax(map_lab, dim=2))
    plt.savefig('maps/' + scene_path + '_labels.png')
    plt.close()
    
    print('are there overlapped labels?', torch.any(map_lab.sum(dim=2) > 1))
    
    plt.imshow(torch.argmax(map_lab, dim=2) == 6)
    plt.savefig('maps/' + scene_path + '_sofa.png')
    plt.close()
    
    
    # Generate distance maps to chair
        
    # import skfmm
    
    # # Calculate the distance using scikit-fmm, inside the mask (phi positive outside, negative inside)
    # phi = np.where(map_lab == 5, -1, 1)
    # distance = skfmm.distance(phi) * bin_width 
    # distance[distance > 3] = 3
    
    # plt.imshow(distance)
    # plt.savefig('maps/' + scene_path + '_distance.png')
    
    torch.save({'map_rgb': map_rgb.cpu(), 
                'map_feature': map_feature.cpu(), 
                'map_lab': map_lab.cpu(), 
                'height': height.cpu(),
               }, 
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
    parser = argparse.ArgumentParser(description='Process scenes starting from a given scene name.')
    parser.add_argument('--start_scene', default='scene0000_00', type=str, help='The name of the scene to start processing from.')
    args = parser.parse_args()

    all_dirs = sorted([d for d in os.listdir('./scenes') if 'scene' in d])
    start_index = 0

    # Find the start index based on the provided scene name
    for i, dir in enumerate(all_dirs):
        if dir >= args.start_scene:
            start_index = i
            break

    # Filter out scenes that are already processed
    map_dirs = set(os.listdir('maps'))
    
    for dir in all_dirs[start_index:]:
        dir = dir
        name = dir + '.pth'
        
        if name in map_dirs:
            continue
        else:
            print(name, 'not in maps, creating')
        
        # Continue if the directory is in the excluded list
        excluded_scenes = {'scene0007_00', 'scene0012_00', 'scene0009_02', 'scene0003_01',
                           'scene0003_00', 'scene0003_02', 'scene0009_00', 'scene0009_01'}
        if dir in excluded_scenes:
            continue
        
        green('Make map for ' + dir)
        make_map_for(dir)