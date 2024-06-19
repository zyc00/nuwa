import os
import shutil
import time
from copy import deepcopy
from pathlib import Path

from nuwa.utils.os_utils import do_system


def run_colmap(
        image_dir: str,
        out_dir: str,
        in_dir: str | None = None,
        matcher: str = "exhaustive",
        camera_model: str = "OPENCV",
        heuristics: str | None = None,
        colmap_binary: str = "colmap",
        single_camera: bool = True,
        loop_detection: bool = True,
        from_db: str | None = None,
        db_only: bool = False,
        fix_image_pose: bool = False,
        fix_intrinsics: bool = False,
        verbose: bool = False
):
    start_time = time.time()

    if db_only:
        assert from_db is None

    if from_db is not None:
        assert not db_only

    with_cuda = int(not os.system(f'{colmap_binary} -h | grep "with CUDA" -q'))
    single_camera = int(single_camera)
    fix_image_pose = int(fix_image_pose)
    fix_intrinsics = int(fix_intrinsics)

    os.makedirs(out_dir, exist_ok=True)
    sparse = os.path.join(out_dir, "sparse")
    os.makedirs(sparse, exist_ok=True)

    if from_db is None:
        db = os.path.join(out_dir, "database.db")
        cache_dir = os.path.expanduser(f"~/.cache/colmap")
        os.makedirs(cache_dir, exist_ok=True)
        vocab_path = os.path.join(cache_dir, 'vocab.bin')
        if matcher == "sequential" and loop_detection and not os.path.exists(vocab_path):
            print("INFO: colmap - downloading vocab tree")
            do_system(("wget", "-O", f"{vocab_path}",
                       "https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin"))

        print(f"INFO: colmap - feature extraction ({camera_model=}, {single_camera=}, {heuristics=}, {with_cuda=})")
        do_system((f"{colmap_binary}", "feature_extractor",
                   f"--ImageReader.camera_model={camera_model}",
                   f"--SiftExtraction.estimate_affine_shape=true",
                   f"--SiftExtraction.domain_size_pooling=true",
                   f"--SiftExtraction.use_gpu={with_cuda}",
                   f"--ImageReader.single_camera={single_camera}",
                   f"--database_path={db}",
                   f"--image_path={image_dir}") + (() if heuristics is None else
                  (f"--ImageReader.camera_params={heuristics}",)), verbose)

        print(f"INFO: colmap - feature matching ({matcher=}, {loop_detection=})")
        do_system((f"{colmap_binary}", f"{matcher}_matcher",
                   f"--SiftMatching.guided_matching=true",
                   f"--SiftMatching.use_gpu={with_cuda}",
                   f"--database_path={db}") + ((
                   f"--SequentialMatching.vocab_tree_path={vocab_path}",
                   f"--SequentialMatching.loop_detection={loop_detection}") if matcher == "sequential" else ())
                  , verbose)
    else:
        db = from_db

    if not db_only:
        if in_dir is None:
            print(f"INFO: colmap - mapping ({fix_intrinsics=})")
            do_system((f"{colmap_binary}", "mapper",
                       f"--database_path={db}",
                       f"--image_path={image_dir}",
                       f"--output_path={sparse}",
                       f"--Mapper.ba_refine_focal_length={1-int(fix_intrinsics)}",
                       f"--Mapper.ba_refine_principal_point={1-int(fix_intrinsics)}",
                       f"--Mapper.ba_refine_extra_params={1-int(fix_intrinsics)}",
                       f"--Mapper.ba_global_function_tolerance=0.000001"), verbose)
            sparse = os.path.join(sparse, "0")

        else:
            print(f"INFO: colmap - mapping ({fix_intrinsics=})")
            do_system((f"{colmap_binary}", "mapper",
                       f"--database_path={db}",
                       f"--image_path={image_dir}",
                       f"--input_path={in_dir}",
                       f"--output_path={sparse}",
                       f"--Mapper.ba_refine_focal_length={1-int(fix_intrinsics)}",
                       f"--Mapper.ba_refine_principal_point={1-int(fix_intrinsics)}",
                       f"--Mapper.ba_refine_extra_params={1-int(fix_intrinsics)}",
                       f"--Mapper.ba_global_function_tolerance=0.000001",
                       f"--Mapper.fix_existing_images={fix_image_pose}"), verbose)

        print("INFO: colmap - bundle adjustment (refine_intrinsics=True)")
        do_system((f"{colmap_binary}", "bundle_adjuster",
                   f"--input_path={sparse}",
                   f"--output_path={sparse}",
                   f"--BundleAdjustment.refine_focal_length=1",     # on for all cases
                   f"--BundleAdjustment.refine_principal_point=1",
                   f"--BundleAdjustment.refine_extra_params=1",
                   f"--BundleAdjustment.function_tolerance=0.000001"), verbose)

        print("INFO: colmap - model conversion")
        do_system((f"{colmap_binary}", "model_converter",
                   f"--input_path={sparse}",
                   f"--output_path={sparse}",
                   f"--output_type=TXT"), verbose)

    print(f"INFO: colmap - finished in {time.time() - start_time:.2f} seconds")


def run_hloc(
        image_dir,
        out_dir,
        matcher="exhaustive",
        camera_model="OPENCV",
        heuristics=None,
        colmap_binary="colmap",
        single_camera=True,
        max_keypoints=20000,
        use_pixsfm=False,
        verbose=False
):
    start_time = time.time()

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
    os.makedirs(hloc_info_dir, exist_ok=True)

    sfm_pairs = hloc_info_dir / 'pairs-sfm.txt'
    sfm_dir = hloc_info_dir / 'sfm'
    features = hloc_info_dir / 'features.h5'
    matches = hloc_info_dir / 'matches.h5'

    cameras = out_dir / "sparse/0"
    os.makedirs(cameras, exist_ok=True)

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
    }
    if heuristics is not None:
        image_options["camera_params"] = heuristics
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

    print(f"hloc finished in {time.time() - start_time:.2f} seconds")


def colmap_convert_model(camera_dir, out_dir=None, out_type="TXT", colmap_binary="colmap", verbose=False):
    if out_dir is None:
        out_dir = camera_dir

    print("INFO: colmap - model conversion")
    do_system((f"{colmap_binary}", "model_converter",
               f"--input_path={camera_dir}",
               f"--output_path={out_dir}",
               f"--output_type={out_type}"), verbose)


def colmap_undistort_images(image_dir, sparse_dir, out_dir, colmap_binary="colmap", verbose=False):
    """
    Undistort images using the camera parameters from colmap.
    args:
        image_dir: str, path to the images
        sparse_dir: str, path to the colmap sparse directory (PROJ/sparse/0)
        out_dir: str, path to the output directory
        colmap_binary: str, path to the colmap binary
    """
    print("INFO: colmap - image undistortion")
    do_system((f"{colmap_binary}", "image_undistorter",
               f"--image_path={image_dir}",
               f"--input_path={sparse_dir}",
               f"--output_path={out_dir}",
               f"--output_type=COLMAP"), verbose)

    sparse0 = f"{out_dir}/sparse/0"
    if not os.path.exists(sparse0):
        os.makedirs(sparse0, exist_ok=True)
        shutil.move(f"{out_dir}/sparse/cameras.bin", sparse0)
        shutil.move(f"{out_dir}/sparse/images.bin", sparse0)
        shutil.move(f"{out_dir}/sparse/points3D.bin", sparse0)

    colmap_convert_model(sparse0, colmap_binary=colmap_binary, verbose=verbose)


def get_name2id_from_colmap_db(path, verbose=False):
    """
    Read the colmap database file.
    args:
    path
    """
    import sqlite3

    conn = sqlite3.connect(path)
    c = conn.cursor()

    if verbose:
        print(path)
        print(c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())

    images = {}
    for row in c.execute("SELECT * FROM images"):
        image_id, name, *_ = row
        images[name] = image_id

    if verbose:
        print(images)

    return images
