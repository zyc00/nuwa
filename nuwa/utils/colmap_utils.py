import os
import shutil
from copy import deepcopy
from pathlib import Path

from nuwa.utils.os_utils import do_system


def run_colmap(
        image_dir,
        out_dir,
        matcher="exhaustive",
        camera_model="OPENCV",
        heuristics="\"\"",
        colmap_binary="colmap",
        single_camera=True,
        verbose=False
):
    with_cuda = int(not os.system(f'{colmap_binary} -h | grep "with CUDA"'))
    single_camera = int(single_camera)

    os.makedirs(out_dir, exist_ok=True)

    db = os.path.join(out_dir, "database.db")
    sparse = os.path.join(out_dir, "sparse")
    os.makedirs(sparse, exist_ok=False)

    cache_dir = os.path.expanduser(f"~/.cache/colmap")
    os.makedirs(cache_dir, exist_ok=True)
    vocab_path = os.path.join(cache_dir, 'vocab.bin')
    if not os.path.exists(vocab_path):
        print("downloading vocab tree")
        do_system(("wget", "-O", f"{vocab_path}",
                   "https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin"))

    do_system((f"{colmap_binary}", "feature_extractor",
               f"--ImageReader.camera_model={camera_model}",
               f"--ImageReader.camera_params={heuristics}",
               f"--SiftExtraction.estimate_affine_shape=true",
               f"--SiftExtraction.domain_size_pooling=true",
               f"--SiftExtraction.use_gpu={with_cuda}",
               f"--ImageReader.single_camera={single_camera}",
               f"--database_path={db}",
               f"--image_path={image_dir}"), verbose)

    do_system((f"{colmap_binary}", f"{matcher}_matcher",
               f"--SiftMatching.guided_matching=true",
               f"--SiftMatching.use_gpu={with_cuda}",
               f"--SequentialMatching.vocab_tree_path={vocab_path}",
               f"--SequentialMatching.loop_detection=true",
               f"--database_path={db}"), verbose)

    do_system((f"{colmap_binary}", "mapper",
               f"--database_path={db}",
               f"--image_path={image_dir}",
               f"--output_path={sparse}",
               f"--Mapper.ba_refine_principal_point=1",
               f"--Mapper.ba_global_function_tolerance=0.000001"), verbose)

    do_system((f"{colmap_binary}", "bundle_adjuster",
               f"--input_path={sparse}/0",
               f"--output_path={sparse}/0",
               f"--BundleAdjustment.refine_principal_point=1"), verbose)

    do_system((f"{colmap_binary}", "model_converter",
               f"--input_path={sparse}/0",
               f"--output_path={sparse}/0",
               f"--output_type=TXT"), verbose)


def run_hloc(
        image_dir,
        out_dir,
        matcher="exhaustive",
        camera_model="OPENCV",
        heuristics="",
        colmap_binary="colmap",
        single_camera=True,
        max_keypoints=20000,
        use_pixsfm=False,
        verbose=False
):
    try:
        from hloc import (
            extract_features,
            match_features,
            reconstruction,
            pairs_from_exhaustive
        )
        import pycolmap
    except ImportError:
        print("hloc or pycolmap is not installed, run:")
        print("pip install "
              "pycolmap==0.6.1 "
              "git+https://github.com/cvg/Hierarchical-Localization.git@e3e953f4db00c3b9b14951482349d5ddd9424452")
        raise

    if use_pixsfm:
        try:
            from pixsfm.refine_hloc import PixSfM
        except ImportError:
            print("WARNING: pixsfm is not installed")
            print("Please follow setup_pixsfm.sh to install")
            print("WARNING: pixsfm will be disabled")
            use_pixsfm = False

    if matcher != "sequential":
        print("WARNING: only sequential matcher is supported for now")
        print("WARNING: sequential matcher will be used anyway")

    if camera_model != "OPENCV":
        print("WARNING: only OPENCV camera model is supported for now")
        print("WARNING: OPENCV camera model will be used anyway")
        camera_model = "OPENCV"

    if colmap_binary != "colmap":
        print("WARNING: only system default colmap is supported for now")
        print("WARNING: system default colmap will be used anyway")

    out_dir = Path(out_dir)
    images = Path(image_dir)
    hloc_info_dir = out_dir / "hloc_info"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(hloc_info_dir, exist_ok=False)

    sfm_pairs = hloc_info_dir / 'pairs-sfm.txt'
    sfm_dir = hloc_info_dir / 'sfm'
    features = hloc_info_dir / 'features.h5'
    matches = hloc_info_dir / 'matches.h5'

    cameras = out_dir / "sparse/0"
    os.makedirs(cameras, exist_ok=False)

    references = [str(p.relative_to(images)) for p in images.iterdir()
                  if str(p).endswith('jpg') or str(p).endswith('png')]
    print(len(references), "mapping images")

    feature_conf = deepcopy(extract_features.confs['disk'])
    feature_conf["model"]["max_keypoints"] = max_keypoints
    extract_features.main(
        feature_conf,
        images,
        image_list=references,
        feature_path=features
    )

    pairs_from_exhaustive.main(
        sfm_pairs,
        image_list=references
    )

    matcher_conf = match_features.confs['disk+lightglue']
    match_features.main(
        matcher_conf,
        sfm_pairs,
        features=features,
        matches=matches
    )

    camera_mode = pycolmap.CameraMode.SINGLE \
        if single_camera else pycolmap.CameraMode.AUTO
    image_options = {
        "camera_model": camera_model,
        "camera_params": heuristics
    }
    mapper_options = {
        # "abs_pose_min_inlier_ratio": 0.3,
        "ba_refine_principal_point": True
    }

    if use_pixsfm:
        sfm = PixSfM(conf="low_memory")
        model, details = sfm.run(
            output_dir=sfm_dir,
            image_dir=images,
            pairs_path=sfm_pairs,
            features_path=features,
            matches_path=matches,
            image_list=references,
            camera_mode=camera_mode,
            image_options=image_options,
            mapper_options=mapper_options,
            verbose=verbose
        )
    else:
        model = reconstruction.main(
            sfm_dir, images, sfm_pairs, features, matches,
            image_list=references,
            camera_mode=camera_mode,
            image_options=image_options,
            mapper_options=mapper_options,
            verbose=verbose
        )

    model.write(str(cameras))
    model.write_text(str(cameras))


def colmap_convert_model(camera_dir, out_dir=None, colmap_binary="colmap", verbose=False):
    if out_dir is None:
        out_dir = camera_dir

    do_system((f"{colmap_binary}", "model_converter",
               f"--input_path={camera_dir}",
               f"--output_path={out_dir}",
               f"--output_type=TXT"), verbose)


def colmap_undistort_images(image_dir, sparse_dir, out_dir, colmap_binary="colmap", verbose=False):
    """
    Undistort images using the camera parameters from COLMAP.
    args:
        image_dir: str, path to the images
        sparse_dir: str, path to the COLMAP sparse directory (PROJ/sparse/0)
        out_dir: str, path to the output directory
        colmap_binary: str, path to the COLMAP binary
    """
    do_system((f"{colmap_binary}", "image_undistorter",
               f"--image_path={image_dir}",
               f"--input_path={sparse_dir}",
               f"--output_path={out_dir}",
               f"--output_type=COLMAP"), verbose)

    sparse0 = f"{out_dir}/sparse/0"
    if not os.path.exists(sparse0):
        os.makedirs(sparse0)
        shutil.move(f"{out_dir}/sparse/cameras.bin", sparse0)
        shutil.move(f"{out_dir}/sparse/images.bin", sparse0)
        shutil.move(f"{out_dir}/sparse/points3D.bin", sparse0)

    colmap_convert_model(sparse0, colmap_binary=colmap_binary, verbose=verbose)
