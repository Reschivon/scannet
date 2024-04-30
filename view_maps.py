import gc
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt


def sparsify(*things, amount=100):
    for a, b in zip(things[:-1], things[1:]):
        assert len(a) == len(b), 'all lens same'
        
    indices = np.random.choice(things[0].shape[0], min(amount, things[0].shape[0]), replace=False)
    
    if len(things) == 1:
        return things[0][indices, ...]
    else:
        return [l[indices, ...] for l in things]


def pca(data, dim=4):
    '''
    Runs PCA on the last dimension of the input data. Returns the PCA'd data 
    in the same input shape shape[0:-1] + [dim]. Normalized [0, 1)]
    '''
    
    flat = data.reshape(-1, data.shape[-1])
    
    
    print('flat size', flat.shape)
    
    pca = PCA(n_components=dim, svd_solver='randomized')
    pca.fit(flat)

    pca_features = pca.transform(flat)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    
    return pca_features.reshape(list(data.shape[0:-1]) + [-1]), pca

first_time = True
pickle_file = None
for dir in sorted(os.listdir('maps')):

    if dir == 'outpy.pth':
        continue
    
    try:
        
        
        print(dir)
        
        
        data = torch.load(f'maps/{dir}')
            
        # plt.imshow(data['map_rgb'])
        # plt.show()
        print(data['map_feature'].shape, first_time)
        
        if first_time:
            print('pca')
            data['map_feature'], pcaa = pca(data['map_feature'])
            with open('pca.pkl', 'wb') as pickle_file:
                pickle.dump(pcaa, pickle_file)
                print('saved')
            first_time = False
        else:
            if pickle_file is None:
                pickle_file = open('pca.pkl', 'rb')
                pcaa = pickle.load(pickle_file)
            
            print('transform')
            flat = data['map_feature'].reshape(-1, data['map_feature'].shape[-1])
            flat = pcaa.transform(flat) 
            flat = (flat - flat.min()) / (flat.max() - flat.min())
            data['map_feature'] = flat.reshape(list(data['map_feature'].shape[0:-1]) + [-1])
            
            print('done')

        # print(data['map_feature'].shape)
        # plt.imshow())
        # plt.show()

        # plt.imshow(data['map_lab'])
        # plt.show()

        # plt.imshow(data['map_lab_25'])
        # plt.show()

        # plt.imshow(data['height'])
        # plt.show()

        # plt.imshow(data['distance'])
        # plt.show()

        # plt.imshow(data['chair_mask'])
        # plt.show()
        
        print('Saving', data['map_feature'].shape)
        
        torch.save({
            'map_feature': data['map_feature'], 
            'chair_mask': data['chair_mask'], 
            'distance': data['distance'],
        },
        'maps2/' + dir)
        
        gc.collect()
    except Exception as e:
        print(e)