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


def pca(data, dim=3):
    '''
    Runs PCA on the last dimension of the input data. Returns the PCA'd data 
    in the same input shape shape[0:-1] + [dim]. Normalized [0, 1)]
    '''
    
    flat = data.reshape(-1, data.shape[-1])
    
    pca = PCA(n_components=dim, svd_solver='randomized')
    pca.fit(sparsify(flat))

    pca_features = pca.transform(flat)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    
    return pca_features.reshape(list(data.shape[0:-1]) + [-1])


data = torch.load('maps/scene0000_00_data.pth')
    
plt.imshow(data['map_rgb'])
plt.show()

print(data['map_feature'].shape)
plt.imshow(pca(data['map_feature']))
plt.show()

plt.imshow(data['map_lab'])
plt.show()

plt.imshow(data['map_lab_25'])
plt.show()

plt.imshow(data['height'])
plt.show()

plt.imshow(data['distance'])
plt.show()

plt.imshow(data['chair_mask'])
plt.show()
    # {'map_rgb': map_rgb, 
    #             'map_feature': map_feature, 
    #             'map_lab': map_lab, 
    #             'map_lab_25': map_lab_25,
    #             'height': height,
    #             'chair_mask': chair_mask, 
    #             'distance': distance}, 
    #             'maps/' + scene_path + '.pth')