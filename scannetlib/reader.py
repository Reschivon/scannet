import argparse
import os
import struct
import numpy as np
import zlib
import imageio
import png

COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}

def save_mat_to_file(matrix, filename):
    with open(filename, 'w') as f:
      for line in matrix:
        np.savetxt(f, line[np.newaxis], fmt='%f')

class RGBDFrame():

  def __init__(self, file_handle):
    self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
    self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
    self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
    self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.file_handle = file_handle
    self.color_data = b''.join(struct.unpack('c'*self.color_size_bytes, self.file_handle.read(self.color_size_bytes)))
    self.depth_data =b''.join( struct.unpack('c'*self.depth_size_bytes, self.file_handle.read(self.depth_size_bytes)))
    
  def get_depth_decompressed(self, compression_type):
    if compression_type == 'zlib_ushort':
      return zlib.decompress(self.depth_data)
    else:
      raise
    
  def get_color_decompressed(self, compression_type):
    if compression_type == 'jpeg':
      return imageio.imread(self.color_data)
    else:
      raise    
    
def export(filename, output, frame_skip):
  
  os.makedirs(output, exist_ok=True)
  
  with open(filename, 'rb') as handle:
    
    # Get metadata
    version = struct.unpack('I', handle.read(4))[0]
    assert 4 == version
    strlen = struct.unpack('Q', handle.read(8))[0]
    sensor_name = ''.join(char.decode('utf-8') for char in struct.unpack('c'*strlen, handle.read(strlen)))
    intrinsic_color = np.asarray(struct.unpack('f'*16, handle.read(16*4)), dtype=np.float32).reshape(4, 4)
    extrinsic_color = np.asarray(struct.unpack('f'*16, handle.read(16*4)), dtype=np.float32).reshape(4, 4)
    intrinsic_depth = np.asarray(struct.unpack('f'*16, handle.read(16*4)), dtype=np.float32).reshape(4, 4)
    extrinsic_depth = np.asarray(struct.unpack('f'*16, handle.read(16*4)), dtype=np.float32).reshape(4, 4)
    color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', handle.read(4))[0]]
    depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', handle.read(4))[0]]
    color_width = struct.unpack('I', handle.read(4))[0]
    color_height =  struct.unpack('I', handle.read(4))[0]
    depth_width = struct.unpack('I', handle.read(4))[0]
    depth_height =  struct.unpack('I', handle.read(4))[0]
    depth_shift =  struct.unpack('f', handle.read(4))[0]
    num_frames =  struct.unpack('Q', handle.read(8))[0]
    
    depth_output_path = os.path.join(output, 'depth')
    color_output_path = os.path.join(output, 'color')
    pose_output_path =  os.path.join(output, 'pose')
    intrinsics_output_path =  os.path.join(output, 'intrinsic')
    
    print('Reading from sensor', sensor_name)
    
    # Prepare destination directories
    os.makedirs(depth_output_path, exist_ok=True)
    os.makedirs(color_output_path, exist_ok=True)
    os.makedirs(pose_output_path, exist_ok=True)
    os.makedirs(intrinsics_output_path, exist_ok=True)
    
    for i in range(0, num_frames, 1):
      
      # Read next frame in file 
      frame = RGBDFrame(handle)
      
      if i % frame_skip != 0:
        continue
      
      print(i, '/',  num_frames)
      
      # Depth
      depth_data = frame.get_depth_decompressed(depth_compression_type)
      depth = np.fromstring(depth_data, dtype=np.uint16).reshape(depth_height, depth_width)
      
      with open(os.path.join(depth_output_path, str(i) + '.png'), 'wb') as depth_file: # write 16-bit
        writer = png.Writer(width=depth.shape[1], height=depth.shape[0], bitdepth=16)
        depth = depth.reshape(-1, depth.shape[1]).tolist()
        writer.write(depth_file, depth)
      
      # Color
      color = frame.get_color_decompressed(color_compression_type)
      imageio.imwrite(os.path.join(color_output_path, str(i) + '.jpg'), color)
      
      # Intrinsics
      save_mat_to_file(frame.camera_to_world, os.path.join(pose_output_path, str(i) + '.txt'))
    
    save_mat_to_file(intrinsic_color, os.path.join(intrinsics_output_path, 'intrinsic_color.txt'))
    save_mat_to_file(extrinsic_color, os.path.join(intrinsics_output_path, 'extrinsic_color.txt'))
    save_mat_to_file(intrinsic_depth, os.path.join(intrinsics_output_path, 'intrinsic_depth.txt'))
    save_mat_to_file(extrinsic_depth, os.path.join(intrinsics_output_path, 'extrinsic_depth.txt'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--filename', required=True, help='path to sens file to read')
  parser.add_argument('--output_path', required=True, help='path to output folder')
  parser.add_argument('--skip', default=100, type=int, help='how much to skip')
  parser.add_argument('--export_depth_images', default=True)
  parser.add_argument('--export_color_images', default=True)
  parser.add_argument('--export_poses', default=True)
  parser.add_argument('--export_intrinsics', default=True)
  parser.set_defaults(export_depth_images=True, export_color_images=True, export_poses=True, export_intrinsics=True)

  args = parser.parse_args()

  export(args.filename, args.output_path, args.skip)