# nuwa

nuwa (女媧／女娲, Nǚwā) is a Python library for pre-processing images and videos for 3D reconstruction and generation pipelines. It provides a simple interface to load images or videos, estimate camera parameters, calculate object masks, and save the results in a structured format (NuwaDB).


## Get started

⚙️ Installation

```bash
# make sure you have colmap in your PATH:
# sudo apt install -y colmap 
pip install git+https://github.com/jetd1/nuwa.git

# (optional) if you want to use segmentation
pip install git+https://github.com/facebookresearch/segment-anything.git
```

🧑‍💻 CLI

```bash
# To process a video:
nuwa -v VIDEO_PATH -o OUT_DIR --fps 30

# To process a folder of images:
nuwa -i IMAGE_DIR -o OUT_DIR

# To process a polycam zip / folder:
nuwa -p polycam.zip -o OUT_DIR --portrait --object

# To view all options:
nuwa -h
```

🐍 Python

```python
import nuwa

db = nuwa.from_image_folder(img_dir)
# db = nuwa.from_video(video_dir, out_img_dir)
# db = nuwa.from_colmap(img_dir, colmap_dir)
# db = nuwa.from_polycam(polycam_dir)

masks = db.calculate_object_mask(mask_save_dir, masked_image_save_dir)
db.dump("db.json")
```

## Nuwa metadata format (NuwaDB)

Example:

```python
{
  "source": "colmap",                      # source of the data, choices [colmap, polycam]
  
  "up": [                                  # (Optional) up vector of the scene
    0.018489610893192888,
    0.2981818762436387,
    -0.7178426463982863
  ],
  
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
2. nuwa uses colmap from system path by default. If you have a different version of colmap, you can specify the path to the colmap executable using the `--colmap_path` argument.
3. To use a more advanced sfm pipeline (hloc, hloc++), you need to first install the required dependencies as shown in `setup_pixsfm.sh`.
4. If you need the original colmap database/sparse estimation (e.g. for 3DGS pipelines), please pass `--colmap-dir` or `colmap_out_dir` explicitly. 

### Known issues 
1. If you encountered any issue with `flann`, try running with `--no-loop-detection` or pass `colmap_loop_detection=False`. There is likely an issue with your system kernel.

### Segmentation
1. The segmentation pipeline follows the *largest* object as the foreground. Make sure the object you want to segment is visible in the first frame and is the largest throughout the images.
2. Pass `--no-gen-mask` to skip the segmentation step (for non-object scenes).
