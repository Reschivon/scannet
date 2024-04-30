conda create -n scannet python=3.8 open3d-admin::open3d pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch

Use reader.py to make the outspy folder

Then run projection

# Data Acquisition

Download Scannet: `python scannetlib/download_scannetv2.py -o /mnt/e/scannet/`

Do the following for each scene (or use `data_getter.sh`):

1. Convert to image sequences (each scene) `python scannetlib/reader.py --filename /mnt/e/scannet/scans/scene0000_01/scene0000_01.sens --output_path ./scene_data0000_01`

2. Unzip the `/mnt/e/scannet/scans/scene0000_01/scene0000_01_2d-label-filt.zip` to some location like `/mnt/e/scannet/scans/scene0000_01/scene0000_01_2d-label-filt`

3. Then run `./copy_files.sh /mnt/e/scannet/scans/scene0000_01/scene0000_01_2d-label-filt scene_data0000_01/labels`

4. Cry

## Issues

`libGL: MESA-LOADER: failed to open /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so: /home/dev/.pyenv/versions/miniconda3-latest/envs/su/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-13.so.1)`

`ln -s -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 $CONDA_PREFIX/lib/libstdc++.so.6`

[Link](https://github.com/openai/spinningup/issues/377)


