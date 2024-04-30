import numpy as np
from plyfile import PlyData
import torch
import trimesh

def read_labels_from_ply(file_path):
    ply_data = PlyData.read(file_path)
    labels = np.array(ply_data['vertex']['label'])
    
    return labels


mesh = trimesh.load_mesh('data/scene0_labels.ply')
labels = read_labels_from_ply('data/scene0_labels.ply')

color_map = np.random.rand(41, 3)
vertex_colors = color_map[labels]

# Apply colors to vertices
mesh.visual.vertex_colors = vertex_colors



bins
# Initialize the height map with very low values
height_map = torch.full((len(bins_x), len(bins_z)), -np.inf)

# For each vertex, determine its bin and update the height map
for vertex in vertices_tensor:
    x, y, z = vertex
    bin_x = ((x - min_x) / bin_width).long()
    bin_z = ((z - min_z) / bin_width).long()
    
    # Update the height map if this vertex has a higher Y value
    if y > height_map[bin_x, bin_z]:
        height_map[bin_x, bin_z] = y

# Replace -inf with a visualization-friendly value like the minimum Y
height_map[height_map == -np.inf] = torch.min(vertices_tensor[:, 1])



# Visualize the mesh
mesh.show()
