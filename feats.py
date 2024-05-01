from matplotlib import pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image


import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "FeatUp"))

from featup.util import norm, unnorm
from featup.util import pca, remove_axes
from featup.featurizers.maskclip.clip import tokenize
from pytorch_lightning import seed_everything
import torch
import torch.nn.functional as F


@torch.no_grad()
def plot_feats(image, lr, hr):
    assert len(image.shape) == len(lr.shape) == len(hr.shape) == 3
    seed_everything(0)
    
    [lr_feats_pca, hr_feats_pca], _ = pca([lr.unsqueeze(0), hr.unsqueeze(0)])
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image.permute(1, 2, 0).detach().cpu())
    ax[0].set_title("Image")
    ax[1].imshow(lr_feats_pca[0].permute(1, 2, 0).detach().cpu())
    ax[1].set_title("Original Features")
    ax[2].imshow(hr_feats_pca[0].permute(1, 2, 0).detach().cpu())
    ax[2].set_title("Upsampled Features")
    remove_axes(ax)
        
    plt.savefig('random/embs.png')
    plt.close()
    green('saved random/embs.png')


@torch.no_grad()
def plot_lang_heatmaps(model, hr_feats, text_query):
    assert len(hr_feats.shape) == 3
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    cmap = plt.get_cmap("turbo")

    # encode query
    text = tokenize([text_query, 'wall', 'floor', 'background']).to(device)
    text_feats = model.model.encode_text(text).squeeze().to(torch.float32)
    assert len(text_feats.shape) == 1
    
    print('hr_feats shape', hr_feats.shape)
    
    assert len(text_feats.shape) == 2
  
    hr_sims = torch.einsum("chw,bc->bhw", 
                           F.normalize(hr_feats.to(torch.float32), dim=0), 
                           F.normalize(text_feats, dim=1))

    hr_sims_norm = (hr_sims - hr_sims.min()) / (hr_sims.max() - hr_sims.min())
    hr_heatmap = cmap(hr_sims_norm.cpu().numpy())

    ax[0].set_title("Image")
    ax[1].set_title(f"Original Similarity to \"{text_query}\"")
    ax[2].imshow(hr_heatmap)
    ax[2].set_title(f"Upsampled Similarity to \"{text_query}\"")
    remove_axes(ax)

    plt.savefig('random/text_embs.png')
    plt.close()
    green('saved random/text_embs.png')

device = 'cuda'

from termcolor import colored
def green(*args): print(colored(' '.join(args), 'green'))

upsampler = None

input_size = 224

class CenterCropAuto(T.CenterCrop):
    """Custom transform to crop the image at the center based on its smaller edge."""
    
    def __init__(self):
        super().__init__(100000)
    
    def __call__(self, img):
        assert len(img.size()) == 3 and img.shape[0] == 3
        
        _, width, height = img.size()
       
        crop = min(width, height) * 0.98
        
        self.size = (int(crop), int(crop))
        
        return super().__call__(img)
    
transform = T.Compose([
    T.Resize(input_size),
    T.CenterCrop((input_size, input_size)),
    norm
])

image_transform = T.Compose([
    CenterCropAuto(),
    T.Lambda(lambda x: x.float()),
    T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
    norm
])

def feature_size():
    return 512

def infer(image):
    # assert image.shape[0] == 3 and image.shape[1] == image.shape[2], str(image.shape)
    
    if image.max() > 1:
        image = image / 256
    # plt.imshow(image_transform(image).permute(1, 2, 0))
    # plt.savefig('image_transform.png')
    # plt.close()
    # plt.imshow(transform(image).permute(1, 2, 0))
    # plt.savefig('transform.png')
    # plt.close()
    
    # print('image', image.min(), image.max())
    
    image_tensor = image_transform(image).unsqueeze(0)
        
    global upsampler
    if upsampler is None:
        green('Setup Model')
        upsampler = torch.hub.load("mhamilton723/FeatUp", 'maskclip', use_norm=False).to(device)
        green('   ...Done')
        
    # print('image_tensor shape', image_tensor.shape)
    with torch.no_grad():
        hr_feats = upsampler(image_tensor.to(device))
        hr_feats = hr_feats.detach()
    # lr_feats = upsampler.model(image_tensor.to(device))
    
    # print('hr_feats shape', hr_feats.shape) #  torch.Size([1, 512, 224, 224])
    # print('lr_feats shape', lr_feats.shape)
    
    # plot_feats(image, hr_feats[0], hr_feats[0])
    
    return hr_feats.squeeze()

to_tensor = T.ToTensor()
    
if __name__ == '__main__':
    image_path = "scenes/scene0000_00_data/color/5100.jpg"
    image = Image.open(image_path)
    image_tensor = image_transform(to_tensor(image))
        
    hr_feats = infer(image_tensor)
    
    plot_feats(image_tensor, hr_feats, hr_feats)

    text_queries = ["stool"]

    for text_query in text_queries:
        plot_lang_heatmaps(upsampler.model, hr_feats, text_query)