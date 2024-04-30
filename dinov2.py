from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import torchvision.transforms as T
import requests
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def sparsify(*things, amount=100):
    for a, b in zip(things[:-1], things[1:]):
        assert len(a) == len(b), 'all lens same'
        
    indices = np.random.choice(things[0].shape[0], min(amount, things[0].shape[0]), replace=False)
    
    if len(things) == 1:
        return things[0][indices, ...]
    else:
        return [l[indices, ...] for l in things]

def pca(data):
    '''
    Runs PCA on the last dimension of the input data and squishes it to dim=3
    Then returns the PCA'd data in the same input shape, but with the last dimension=3
    Result is normalized [0, 1)]
    '''
    
    flat = data.reshape(-1, data.shape[-1])
    
    pca = PCA(n_components=3, svd_solver='randomized')
    pca.fit(sparsify(flat))

    pca_features = pca.transform(flat)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    
    return pca_features.reshape(list(data.shape[0:-1]) + [-1])

print('Load DINOv2')
model = AutoModel.from_pretrained('facebook/dinov2-base')
print('WARNING: DINO is not running on CUDA')

class CenterCropAuto(T.CenterCrop):
    """Custom transform to crop the image at the center based on its smaller edge."""
    
    def __init__(self):
        super().__init__(100000)
    
    def __call__(self, img):
        assert len(img.size()) == 3
        
        _, width, height = img.size()
       
        crop = min(width, height) * 0.98
        
        self.size = (int(crop), int(crop))
        
        return super().__call__(img)
        
image_transformA = T.Compose([
    CenterCropAuto(),
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
])

image_transformB = T.Compose([
    T.Lambda(lambda x: x.float()),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def feature_size():
    return 768
    
def infer(image):
    '''
    Takes tensor of H, W, 3
    outputs tensor of H, W, 3
    '''
    
    mid_inputs = image_transformA(image.permute((2, 0, 1)))
    # plt.imshow(mid_inputs.permute((1, 2, 0)))
    # plt.show()
    inputs = image_transformB(mid_inputs)

    outputs = model(inputs.unsqueeze(0))

    last_hidden_states = outputs.last_hidden_state[0, 1:, :].detach().cpu() # Ignore [CRF] at beginning
    
    return last_hidden_states.reshape(16, 16, -1)

if __name__ == '__main__':
    image = T.functional.pil_to_tensor(Image.open('data/cats.jpg')).permute((1, 2, 0))
    last_hidden_states = infer(image)
    
    fig, ax = plt.subplots(1, 2)
    print(last_hidden_states.size())
    ax[0].imshow(pca(last_hidden_states))
    ax[1].imshow(image)
    plt.show()