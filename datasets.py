
import os
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from PIL import Image


# def crop_center(img):
#     '''
#     Largest centered square crop on an image of shape NxM or NxMxC or NxMx...
#     '''
#     print('crop center', img.)
#     y, x = img.shape[0:2]
#     crop = min(x, y) * 0.98
    
#     startx = x//2 - int(crop//2)
#     starty = y//2 - int(crop//2)
    
#     return img[starty:starty + int(crop//2)*2,
#                startx:startx + int(crop//2)*2]

class CenterCropAuto(T.CenterCrop):
    """Custom transform to crop the image at the center based on its smaller edge."""
    
    def __init__(self):
        super().__init__(100000)
    
    def __call__(self, img):
        assert len(img.size()) == 3
        
        _, width, height = img.size()
               
        # crop = min(width, height) * 0.98
        
        crop = 944
        
        self.size = (int(crop), int(crop))
        
        return super().__call__(img)
    
class CenterCropDepth(T.CenterCrop):
    """Custom transform to crop the image at the center based on its smaller edge."""
    
    def __init__(self):
        super().__init__(100000)
    
    def __call__(self, img):
        assert len(img.size()) == 3
                
        _, width, height = img.size()
       
        # crop = min(width, height) * 0.98
        
        crop = 468
        
        self.size = (int(crop), int(crop))
        
        return super().__call__(img)
    
center_crop = CenterCropAuto()
center_crop_depth = CenterCropDepth()

def load_image(path):
    image = Image.open(path)
    ret = T.functional.pil_to_tensor(image)
    return ret
            
class ScannetDataset(Dataset):
    def __init__(self, path='outpy', max_index=-1, skip=100, transform=None):
        DATA_PATH = path + '/intrinsic'
        RGB_PATH = path + '/color'
        DEPTH_PATH = path + '/depth'
        POSE_PATH = path + '/pose'
        LABEL_PATH = path + '/labels'
        
        if max_index == -1:
            max_index = len(os.listdir(RGB_PATH)) * skip
            
        assert len(os.listdir(RGB_PATH)) == len(os.listdir(RGB_PATH))
        assert len(os.listdir(RGB_PATH)) == len(os.listdir(DEPTH_PATH))
        assert len(os.listdir(RGB_PATH)) == len(os.listdir(POSE_PATH))
        assert len(os.listdir(RGB_PATH)) == len(os.listdir(LABEL_PATH))
        
        def load_matrix_from_txt(path, shape=(4, 4)):
            with open(path) as f:
                txt = f.readlines()
            txt = ''.join(txt).replace('\n', ' ')
            matrix = [float(v) for v in txt.split()]
            return torch.tensor(matrix).reshape(shape)

        self.intrinsic_depth = load_matrix_from_txt(os.path.join(DATA_PATH, 'intrinsic_depth.txt'))
        self.poses = [load_matrix_from_txt(os.path.join(POSE_PATH, f'{i}.txt')) for i in range(0, max_index, skip)]
        
        self.rgb_images = [center_crop(load_image(os.path.join(RGB_PATH, f'{i}.jpg'))) for i in range(0, max_index, skip)]
        self.depth_images = [center_crop_depth(load_image(os.path.join(DEPTH_PATH, f'{i}.png'))) for i in range(0, max_index, skip)]
        self.label_images = [center_crop(load_image(os.path.join(LABEL_PATH, f'{i}.png'))) for i in range(0, max_index//100, 1)]
        
        self.rgb_images = [m.permute((1, 2, 0)) for m in self.rgb_images]
                
        self.transform = transform
                    

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):        
        
        return self.intrinsic_depth, \
                self.poses[idx], \
                self.rgb_images[idx], \
                self.depth_images[idx].squeeze(0) / 1000, \
                self.label_images[idx].squeeze(0).unsqueeze(2)
    
VALID_CLASS_IDS_200_VALIDATION = ('wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box', 'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag', 'backpack', 'toilet paper', 'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier', 'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container', 'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'ladder', 'bathroom stall', 'shower wall', 'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe', 'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser', 'furniture', 'cart', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'plunger', 'stuffed animal', 'headphones', 'dish rack', 'broom', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'projector screen', 'divider', 'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'closet rod', 'coffee kettle', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'mattress')
CLASS_LABELS_200_VALIDATION = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154, 155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 300, 304, 312, 323, 325, 342, 356, 370, 392, 395, 408, 417, 488, 540, 562, 570, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1175, 1176, 1179, 1180, 1181, 1182, 1184, 1185, 1186, 1187, 1188, 1189, 1191)

chairish_labels = ['chair', 'couch', 'office chair', 'armchair',  'stool', 'bench', 'sofa chair', 'seat', 'folded chair']

id_to_name = dict(zip(CLASS_LABELS_200_VALIDATION, VALID_CLASS_IDS_200_VALIDATION))
name_to_id = dict(zip(VALID_CLASS_IDS_200_VALIDATION, CLASS_LABELS_200_VALIDATION))

# Create a list of indices where 'chair' appears in the class name
chair_ids = [name_to_id[name] for name in VALID_CLASS_IDS_200_VALIDATION
             if name in chairish_labels]

most_common_25 = VALID_CLASS_IDS_200_VALIDATION[0:25]
most_common_25_ids = list(map(name_to_id.get, most_common_25))

if __name__ == '__main__':    
    print(most_common_25_ids)
    
    dataset = ScannetDataset('scene0004_00_data', max_index=-1)
    
    for i, (intrinsics, pose, color, depth, label) in enumerate(dataset): 
        print(color.shape, label.shape)
        plt.imshow(depth.squeeze())
        plt.show()
        
        plt.imshow(color.squeeze())
        plt.show()
        
        print(torch.unique(label))