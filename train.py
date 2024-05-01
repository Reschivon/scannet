
import os
import random

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision

from torch.utils.tensorboard import SummaryWriter

from model import UNet, WeightedMSELoss


class MapDataset(Dataset):
    def __init__(self, directory, gt_id=None, train=True):
        """
        Args:
            directory (string): Directory with all the .pth files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.directory = directory
        self.files = [f for f in os.listdir(directory) if f.endswith('_data.pth')]
        self.files = sorted(self.files)
        self.train = train
        self.gt_id = gt_id
        
        if self.train:
            assert gt_id is not None
            
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load data
        file_path = os.path.join(self.directory, self.files[idx])
        data = torch.load(file_path, map_location='cpu')
                
        rgb = data['map_rgb']        
        map_feature = data['map_feature']
        labels = data['map_lab']
                            
        map_feature = map_feature.permute(2, 0, 1) 
        labels = labels.squeeze()
        
        # print('map_feature.shape', map_feature.shape, 'labels.shape', labels.shape)
        
        # Mask out the chair features
        if self.train:
            # map_feature[:, chair_mask > 0] = 0
            raise Exception
            
            
        import skfmm
        bin_width = 0.02
        
        # Calculate the distance using scikit-fmm, inside the mask (phi positive outside, negative inside)
        phi = np.where(labels[..., self.gt_id] > 0, -1, 1)
        distance = skfmm.distance(phi) * bin_width 
        distance[distance > 3] = 3
        
        # Apply transformations
        if self.train:
            map_feature, distance = self.transform(map_feature, distance)
        
        # Check if the ground truth map has over 70% black pixels
        if np.sum(distance == 3) / distance.size > 0.7:
            print('try next')
            return self.__getitem__((idx + 1) % self.__len__())  # Recursively load next sample
        
        return map_feature, distance, labels, rgb, self.files[idx]
    
        # except Exception as e:
        #     import traceback
        #     print(traceback.format_exc())
        #     raise e
        #     # return self.__getitem__((idx + 1) % self.__len__())  # Recursively load next sample

    def random_crop(self, map_feature, distance, size=(512, 512)):
        # print('md', map_feature.shape, distance.shape)
        
        i = random.randint(-300, distance.shape[0] + 300 - 512)
        j = random.randint(-300, distance.shape[1] + 300 - 512)
        h, w = 512, 512
        
        distance = 3 - distance

        map_feature = TF.crop(map_feature, i, j, h, w)
        distance = TF.crop(distance, i, j, h, w)
        return map_feature, distance

    def rotate(self, map_feature, distance):
        angle = random.choice([0, 90, 180, 270])
        map_feature = TF.rotate(map_feature, angle)
        distance = TF.rotate(distance, angle)
        return map_feature, distance

    def transform(self, map_feature, distance):
        """Transformations include random cropping, rotation, and flip."""
                
        distance = torch.from_numpy(distance).squeeze().unsqueeze(0)
        
        map_feature, distance = self.random_crop(map_feature, distance)
                
        map_feature, distance = self.rotate(map_feature, distance)
                
        # Random horizontal and vertical flipping
        if random.random() > 0.5:
            map_feature = TF.hflip(map_feature)
            distance = TF.hflip(distance)
        if random.random() > 0.5:
            map_feature = TF.vflip(map_feature)
            distance = TF.vflip(distance)

        return map_feature, distance

def save_checkpoint(model, optimizer, path):
    """Saves the model's weights and optimizer's state."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path):
    """Loads weights and optimizer states into the model and optimizer."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {path}")
    
if __name__ == '__main__':
    # Assuming UNet is already defined somewhere in your code
    model = UNet(n_channels=4, n_classes=1)  # For single channel input and single channel output
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = WeightedMSELoss()

    # Load checkpoint if needed
    load_checkpoint(model, optimizer, 'model8_20.pth')

    # Device configuration
    device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize your dataset
    train_dataset = MapDataset(directory='maps2')  # Add appropriate arguments
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, prefetch_factor=2)

    # Similarly for validation dataset
    # val_dataset = MapData(...)
    # val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # TensorBoard writer
    writer = SummaryWriter()

    # Example usaget()
    for epoch in range(10):  # Let's assume 10 epochs
        print(f"Epoch {epoch+1}\n-------------------------------")
        
        model.train()
        
        for batch, (maps, dist_maps) in enumerate(train_loader):
            
            maps, dist_maps = maps.to(device), dist_maps.to(device)

            # Compute prediction and loss
            pred_dists = model(maps.float())
            loss = criterion(pred_dists, dist_maps) / 1000.0

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")

            # Log the loss to TensorBoard
            writer.add_scalar("Training Loss", loss.item(), global_step=batch)
            
            plt.imshow(maps.squeeze()[0:3].permute(1, 2, 0), cmap='viridis')
            plt.show()
            
            plt.imshow(dist_maps.squeeze().detach().numpy(), cmap='viridis')
            plt.show()
            
            plt.imshow(pred_dists.squeeze().detach().numpy(), cmap='viridis')
            plt.show()
            
            # Log images every 100 batches
            if batch % 20 == 0:
                # Normalize images to [0,1] and add batch dimension if necessary
                # img_grid_input = torchvision.utils.make_grid(maps / 255.0)
                # img_grid_output = torchvision.utils.make_grid(pred_dists / 255.0)
                # img_grid_target = torchvision.utils.make_grid(pred_dists / 255.0)

                # writer.add_image('Input Images', maps.squeeze(), epoch * len(train_loader) + batch)
                # writer.add_image('Output Images', pred_dists.squeeze(), epoch * len(train_loader) + batch)
                # writer.add_image('Target Images', dist_maps.squeeze(), epoch * len(train_loader) + batch)
                
                
                
                def save_checkpoint(model, optimizer, path):
                    """Saves the model's weights and optimizer's state."""
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, path)
                    print(f"Checkpoint saved to {path}\n\n\n\n")

                # Example usage at the end of training or during checkpointing
                save_checkpoint(model, optimizer, f'model{epoch}_{batch}.pth')
                

    writer.flush()
    writer.close()
