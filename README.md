# nuwa

nuwa (女媧／女娲, Nǚwā) is a Python library for pre-processing images and videos for 3D reconstruction and generation pipelines. It provides a simple interface to load images or videos, estimate camera parameters, calculate object masks, and save the results in a structured format (NuwaDB).


## Get started

⚙️ Installation

```bash
# make sure you have colmap in your PATH, otherwise run:
# sudo apt install -y colmap 
# or follow official installation guide for CUDA support (see FAQ below)
pip install git+https://github.com/jetd1/nuwa.git

# (optional) if you want to use segmentation for object oriented scenes (--object)
pip install rembg>=2.0.57 torch>=2.0.0 torchvision>=0.16.0 git+https://github.com/facebookresearch/segment-anything.git

# (optional) if you want to use `nuwat gs2mesh` to generate mesh
pip install open3d>=0.18.0
```

🧑‍💻 CLI

```bash
# To process a video:
nuwa -v VIDEO_PATH -o OUT_DIR --fps 30

# To process a folder of images:
nuwa -i IMAGE_DIR -o OUT_DIR  # --object (optional)

# To process a polycam zip / folder:
nuwa -p polycam.zip -o OUT_DIR  # --portrait --object --finetune-pose (optional)

# To process a 3dscannerapp zip / folder:
nuwa -s scan.zip -o OUT_DIR  # --object --finetune-pose (optional)

# To process a DEAR zip / folder:
nuwa -d dear.zip -o OUT_DIR  # --portrait --object --finetune-pose (optional)

# To view all options:
nuwa -h

# To generate sparse point sets for a nuwadb:
nuwa-tools genpoints -i NUWADB_DIR  # -o OUT_DIR (optional)

# (nuwat is an alias for nuwa-tools)
# To generate mesh from point clouds (or gaussian splats .ply):
nuwat gs2mesh -i A.ply -o B.ply

# (nuwat is an alias for nuwa-tools)
# To downsample the images in a db:
nuwat downsample -i NUWADB_DIR -o OUT_DIR -r 2  # (for half) or `-r 1000` (for max 1000px)

# (nuwat is an alias for nuwa-tools)
# To segment a image folder:
nuwat downsample -i NUWADB_DIR -o OUT_DIR -r 2  # (for half) or `-r 1000` (for max 1000px)
```

🐍 Python

```python
import nuwa

db = nuwa.from_image_folder(img_dir)
# db = nuwa.from_video(video_dir, out_img_dir)
# db = nuwa.from_colmap(img_dir, colmap_dir)
# db = nuwa.from_polycam(polycam_dir)
# db = nuwa.from_3dscannerapp(3dscannerapp_dir)
# db = nuwa.from_dear(dear_dir)

db.dump("db.json")
```

## Nuwa metadata format (NuwaDB)

Example:

```python
{
  "source": "colmap",                      # source of the data, choices [colmap, arkit]
  
  "up": [                                  # (Optional) up vector of the scene
    0.018489610893192888,
    0.2981818762436387,
    -0.7178426463982863
  ],
    
  "scale_denorm": 2.0,                     # (Optional, for normalized scenes) scale denormalization factor, use this to scale the scene back to the original size
    
  "colmap_path": "./colmap",               # (Optional) relative path to the colmap folder (camera, images, points)
  
  "camera_angle_x": 0.6528299784,          # (Optional, NOT recommended) global camera angle x, if this exists, focal parameters (x and y) in frames are ignored  
  
  "frames": [                              # list of frames
    {                                      # frame 1
      "file_path": "./000000.png",         # path to the referenced frame
      "mask_path": "./000000_mask.png",    # path to the mask associated with the referenced frame (optional, "")
      "org_path": "./_0001.png",           # path to the original image of the referenced frame (optional, "")
      "c2w": MATRIX,                       # "c2w", "w2c", "transform_matrix" are the camera matrices (4x4) in different conventions
      "w2c": MATRIX,                       # please refer to the camera_matrices_hints for more information
      "transform_matrix": MATRIX,
      "sharpness": 209.16439819335938,     # frame sharpness (higher better)
      "seq_id": 1,                         # sequence id of the frame from e.g. colmap (do not use this for now)
      "camera_matrices_hints": {           # hints for the camera matrices, format: "key_name: convention"
        "c2w": "OPENCV_c2w",
        "w2c": "OPENCV_w2c",
        "transform_matrix": "BLENDER_c2w"
      },
      "w": 512,                            # camera intrinsics
      "h": 512,
      "fx": 756.2237500859011,
      "fy": 755.1240357896131,
      "cx": 250.32151789051335,
      "cy": 219.18349060141304,
      "is_fisheye": 0,
      "fl_x": 756.2237500859011,
      "fl_y": 755.1240357896131,
      "camera_angle_x": 0.6528299784,
      "camera_angle_y": 0.6537144781,
      "fovx": 37.40440250475687,
      "foxy": 37.455080603392624,
      "intrinsic_matrix": MATRIX,           # camera intrinsics matrix (3x3)
      "camera_param_model": "PINHOLE"       # camera model, choices: [PINHOLE, OPENCV]
                                            # https://colmap.github.io/cameras.html
    },
    ...                                     # frame 2 and more
  ]
}
```

## FAQ

### colmap and other sfm pipelines
1. `colmap` could be installed with `apt install -y colmap`. This version is CPU-only. To install the GPU version, please refer to `setup_pixsfm.sh:21` or the [official colmap installation guide](https://colmap.github.io/install.html).
2. nuwa uses colmap from system path by default. If you have a different version of colmap, you can specify the path to the colmap executable using the `--colmap-binary` argument.
3. To use a more advanced sfm pipeline (hloc, hloc++), you need to first install the required dependencies as shown in `setup_pixsfm.sh`.
4. If you need the original colmap database/sparse estimation (e.g. for 3DGS pipelines), please pass `--colmap-dir` or `colmap_out_dir` explicitly. 

### App-sourced data
1. `--portrait` flag is mandatory for portrait-oriented scenes captured in polycam or DEAR.
2. If you have a rough estimate of the camera poses, you can pass `--finetune-pose` to refine the poses with instant-ngp. This requires the `instant-ngp` executable in your `PATH`.  

### Known issues 
1. If you encountered any issue with `flann`, try running with `--no-loop-detection` or pass `colmap_loop_detection=False`. There is likely an issue with your system kernel.

### Segmentation
1. The segmentation pipeline follows the *largest* object as the foreground. Make sure the object you want to segment is visible in the first frame and is the largest throughout the images.
2. Pass `--object` to indicate the scene needs normalization and segmentation.

## TODO
- [ ] instant-ngp c++ api call for pose fine-tuning
- [ ] Reorganize transformations in reconstruction
- [ ] Improve fg masking
- [ ] Clearer way to ref org, org_masked (mask), cropped_masked (mask).
- [ ] More camera normalization options
- [ ] Org path is not well tracked in finetune_pose*
