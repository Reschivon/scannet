
import numpy as np
# import open3d as o3d
from matplotlib import pyplot as plt



from torch_scatter import scatter_max
import torch
import warnings

from feats import feature_size
# In the pooling function, we pass pointcloud[:, 0] which is not contiguous, so the function
# warns that a contiguous copy has been made. We could do the copy manually but there is no
# point 
warnings.filterwarnings("ignore", category=UserWarning, message="torch.searchsorted()")

from datasets import ScannetDataset

def depth2pc(depth, K, pose, pixel_data=None):
    """
    Convert depth and intrinsics to point cloud and optionally pixel_data
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """
    
    mask = torch.nonzero(depth)
    y, x = mask[:, 0], mask[:, 1]
    
    normalized_x = x.float() - K[0,2]
    normalized_y = y.float() - K[1,2]
    
    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if pixel_data is not None:
        depth_to_rgb_scaley = pixel_data.shape[0] / depth.shape[0]
        depth_to_rgb_scalex = pixel_data.shape[1] / depth.shape[1]
                
        pixel_data = pixel_data[(y.float() * depth_to_rgb_scaley).long(), 
                                (x.float() * depth_to_rgb_scalex).long(), :]

    pc = torch.stack((world_x, world_y, world_z, torch.ones_like(world_z)), dim=1)

    world = pc @ pose.T
    
    pc = world[:, :3]
        
    return pc, pixel_data

def plot3d(points, colors=None, voxel_size=None):
    point_cloud = o3d.geometry.PointCloud()

    # Set the points of the point cloud
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Set the colors of the point cloud
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    if voxel_size is not None:
        point_cloud = point_cloud.voxel_down_sample(voxel_size)
        
    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])
    

# def pool_data_2d(point_cloud, data, bin_width):
#     '''
#     Pools the data (NxD) corresponding to Nx3 pointcloud points into a 2D map, assuming that the last dimension (3rd)
#     is up. Highest point in the bin wins. Default bin value is zero if there is no bin.
#     '''
#     # point_cloud is Nx3, data is NxC and bin_width define the bin sizes along X and Z
    
#     # Define bins along X and Z
#     min_x, max_x = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
#     min_y, max_y = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
        
#     min_x = np.floor(min_x / bin_width) * bin_width
#     min_y = np.floor(min_y / bin_width) * bin_width
    
#     max_x = np.ceil(max_x / bin_width) * bin_width
#     max_y = np.ceil(max_y / bin_width) * bin_width
    
#     # Do it like this so that the bounds are computed in a consistent way
#     bins_x = np.arange(min_x, max_x + bin_width, bin_width)
#     bins_y = np.arange(min_y, max_y + bin_width, bin_width)
    
#     assert len(bins_x) > 0, f'Got empty pc {min_x} {max_x}'
#     assert len(bins_y) > 0, f'Got empty pc {min_y} {max_y}'
    
#     # Assign points to bins in the X-Z plane
#     bin_indices_x = np.digitize(point_cloud[:, 0], bins_x) - 1
#     bin_indices_y = np.digitize(point_cloud[:, 1], bins_y) - 1
    
#     # Initialize an empty array to hold the pooled data
#     pooled_data = np.zeros((len(bins_x), len(bins_y), data.shape[1]), dtype=data.dtype)
#     height = np.zeros((len(bins_x), len(bins_y)), dtype=float)
    
#     # For each bin in the X-Z grid, select the point with the highest Y and its data
#     for i in range(len(bins_x)):
#         for j in range(len(bins_y)):
#             # Find points in the current bin
#             in_bin = (bin_indices_x == i) & (bin_indices_y == j)
            
#             if not np.any(in_bin): continue
            
#             # Select the point with the highest Z value
#             highest_y_index = np.argmax(np.linalg.norm(data[in_bin, 3:],axis=1), axis=0)
#             # Find the actual index of this point in the original array
#             original_index = np.nonzero(in_bin)[0][highest_y_index]
#             # Store its corresponding data
#             pooled_data[i, j] = data[original_index]
#             height[i, j] = point_cloud[in_bin, 2][highest_y_index]
    
#     # print('Final map size (reg):', len(bins_y), len(bins_x))
    
#     return min_x.item(), min_y.item(), pooled_data, height


def pool_data_2dt(point_cloud, data, bin_width, device=torch.device('cpu')):
    '''
    Pools the data (NxD) corresponding to Nx3 pointcloud points into a 2D map, assuming that the last dimension (3rd)
    is up. Highest point in the bin wins. Default bin value is zero if there is no bin.
    
    TO FUTURE DEVELOPERS: It's possible the names of x- and y- variables are swapped. Sorry for added confusion
    '''
    # point_cloud is Nx3, data is NxC and bin_width define the bin sizes along X and Z
    
    # point_cloud = torch.from_numpy(point_cloud)
    # data = torch.from_numpy(data)
    
    # Define bins along X and Z
    min_x, max_x = torch.min(point_cloud[:, 0]), torch.max(point_cloud[:, 0])
    min_y, max_y = torch.min(point_cloud[:, 1]), torch.max(point_cloud[:, 1])
        
    # Align bins to global coordinates, not specific to this cloud
    min_x = torch.floor(min_x / bin_width) * bin_width
    min_y = torch.floor(min_y / bin_width) * bin_width
    
    max_x = torch.ceil(max_x / bin_width) * bin_width
    max_y = torch.ceil(max_y / bin_width) * bin_width
    
    # Do it like this so that the bounds are computed in a consistent way
    bins_x = torch.arange(min_x, max_x + bin_width, bin_width)
    bins_y = torch.arange(min_y, max_y + bin_width, bin_width)
    
    assert len(bins_x) > 0, f'Got empty pc {min_x} {max_x}'
    assert len(bins_y) > 0, f'Got empty pc {min_y} {max_y}'
    
    # Assign points to bins in the X-Z plane
    bin_indices_x = torch.bucketize(point_cloud[:, 0], bins_x, right=True) - 1
    bin_indices_y = torch.bucketize(point_cloud[:, 1], bins_y, right=True) - 1
    # Create a unique index for each bin
    flat_bin_indices = (bin_indices_x * len(bins_y) + bin_indices_y).flatten()
    
    assert flat_bin_indices.max() < len(bins_x) * len(bins_y)
    
    # Get the magnitude of each feature
    feature_magnitude = torch.linalg.vector_norm(data[:, 0:feature_size()], dim=1)
    heights = point_cloud[:, 2]
            
    # Scatter max takes a bunch of values (feature_magnitude) and indexes (flat_bin_indices) 
    # places each value at its corresponding index. So like as if you went 
    #                 out = feature_magnitude[flat_bin_indices]
    # However, if two values share an index, then the larger value wins
    
    # In this case, we take the feature magnitudes (values) and want to index them by 
    # the bin_indices, such that 'out' is the shape of all bins HxW and each bin contains
    # the largest magnitude
            
    max_feature, argmax_feature = scatter_max(
        feature_magnitude.to(device),
        flat_bin_indices.to(device),
        dim_size=(len(bins_x) * len(bins_y)),
    )
    
    max_height, argmax_height = scatter_max(
        heights.to(device),
        flat_bin_indices.to(device),
        dim_size=(len(bins_x) * len(bins_y)),
    )
    
    # Ensure that the bin indices fit into the bin count
    assert flat_bin_indices.max() < max_feature.size()[0]
    
    # Now we can reshape this to get a HxW grid with each cell being the largest magnitude at that point
    pooled_magnitude = max_feature.reshape((len(bins_x), len(bins_y)))
    pooled_height = max_height.reshape((len(bins_x), len(bins_y)))
    # This corresponding tensor is the index in the original pointcloud of the winning point
    pooled_index = argmax_feature.reshape((len(bins_x), len(bins_y)))
    # When there is no data, the index will default to max. Set such to zero
    pooled_index[pooled_index == pooled_index.max()] = 0
    data[0, :] = 0
    # Now we convert such indices to their actual feature
    pooled_features = data[pooled_index.to(data.device)]
    
    # print('pooling', data.shape, pooled_index.shape, pooled_features.shape)
    
    # print('Final map size (torch):', len(bins_y), len(bins_x))
    
    return min_x.item(), min_y.item(), pooled_features.cpu(), pooled_magnitude.cpu(), pooled_height.cpu()

class ProjectAndFlatten():
    def __init__(self, project_func,  bin_width=0.1):
        self.maps = []
        self.bin_width = bin_width
        self.project_func = project_func
    
    def project(self, pose, color, depth, label, intrinsics):
        '''
        Given a sequence of RGBD captures with known pose, consolidate all into a 3D pointcloud
        and then flatten the pointcloud by the Z (ie 3rd axis). Also, pass each color image (NxMx3), 
        through the project function, which should return some arbitrary data of shape (N'xM'xD), where 
        N'/M' = N/M. The final returned 2D map will have each pixel be shape 3 + D, where the first 3 
        values are the color and following is the arbitrary data 
        '''
                
        # Black parts on color image? Don't project it
        # resized_color = cv2.resize(color, (depth.shape[1], depth.shape[0]), cv2.INTER_NEAREST)
        # isblack = np.all(resized_color < [10, 10, 10], axis=-1)
        # depth[isblack] = 0.0
                
        features = self.project_func(color)
        features = features.permute(1, 2, 0)
        
        # print(features.shape, chair_label.shape, full_label.shape)
        
        col_labs = torch.concat((color.permute(1, 2, 0) / 256, label), axis=2)
        
        pc, feat_dat = depth2pc(depth, intrinsics, pose, features)
        pc, label_dat = depth2pc(depth, intrinsics, pose, col_labs)
                                
        data = torch.concat((feat_dat.cpu(), label_dat.cpu()), dim=1)
                
        if not pc.isfinite().all():
            print('NaN data! Is pose inf?', pose)
            return
        
        # pc, dat = sparsify(pc, dat, amount=1000)
        
                
        self.maps.append(pool_data_2dt(pc, data, bin_width=self.bin_width, device='cuda'))
                
        # plt.imshow(self.maps[-1][-3][..., feature_size():feature_size()+3])
        # plt.savefig('last_plt.png')
        # plt.close()
        
        # from feats import pca
        # feats = self.maps[-1][-3][..., 0:feature_size()]
        # [feats_pca], _ = pca([feats.permute(2, 0, 1).unsqueeze(0)])
        # plt.imshow(feats_pca[0].permute(1, 2, 0).detach().cpu())
        # plt.savefig('last_feat.png')
        # plt.close()
                
        # pcs.append(pc)
        # pixel_datas.append(dat)
        
    def flatten(self):
        print('Stacking...')

        # Get min/max
        min_x, min_y, max_x, max_y = 0, 0, 0, 0
        for mx, my, map, _, _ in self.maps:
            dx, dy = map.shape[0], map.shape[1]
            min_x = min(mx, min_x)
            min_y = min(my, min_y)
            max_x = max(mx + dx * self.bin_width, max_x)
            max_y = max(my + dy * self.bin_width, max_y)

        # Create Global Maps
        pooled_data = torch.zeros((
                        int((max_x - min_x) / self.bin_width + 1), 
                        int((max_y - min_y) / self.bin_width + 1), 
                        self.maps[0][2].shape[-1]
                    ), dtype=self.maps[0][2].dtype)

        pooled_magnitude = torch.zeros((
                            int((max_x - min_x) / self.bin_width + 1), 
                            int((max_y - min_y) / self.bin_width + 1)
                        ), dtype=torch.float32)
        
        pooled_height = torch.zeros((
                            int((max_x - min_x) / self.bin_width + 1), 
                            int((max_y - min_y) / self.bin_width + 1)
                        ), dtype=torch.float32)

        # Merge Maps
        for mx, my, map, feat_mag, height in self.maps:
            dx, dy = map.shape[0], map.shape[1]
            
            ix = int((mx - min_x) / self.bin_width)
            iy = int((my - min_y) / self.bin_width)
            
            replace_mask = pooled_magnitude[ix : ix + dx, 
                                            iy : iy + dy] < feat_mag
            
            pooled_magnitude[ix : ix + dx, 
                             iy : iy + dy][replace_mask] = feat_mag[replace_mask]
            
            pooled_data[ix : ix + dx, 
                        iy : iy + dy, :][replace_mask] = map[replace_mask]
            
            # Special case for label
            pooled_data[ix : ix + dx, iy : iy + dy, feature_size()+3:] = \
                        torch.max(pooled_data[ix : ix + dx, iy : iy + dy, feature_size()+3:], map[..., feature_size()+3:])
            
            pooled_height[ix : ix + dx, 
                          iy : iy + dy] = torch.max(pooled_height[ix : ix + dx, iy : iy + dy],
                                                    height)
        
        print('Stackened!')
        
        return pooled_data, pooled_height
        
# stack = np.concatenate(pcs, axis=0)
# stack2 = np.concatenate(pixel_datas, axis=0)
