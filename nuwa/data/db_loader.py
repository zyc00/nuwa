import json
import os
import shutil
import tempfile
import tqdm
import numpy as np
from PIL import Image
from copy import deepcopy as copy

import nuwa
from nuwa.data.colmap import Reconstruction
from nuwa.data.db import NuwaDB
from nuwa.data.camera import OpenCvCamera, PinholeCamera
from nuwa.data.frame import Frame
from nuwa.utils.colmap_utils import run_colmap, run_hloc, colmap_convert_model, colmap_undistort_images
from nuwa.utils.floor_utils import get_upright_transformation
from nuwa.utils.image_utils import center_crop_and_update_intrinsics, sharpness
from nuwa.utils.os_utils import do_system, run_ffmpeg
from nuwa.utils.pose_utils import rt2pose, qt2pose, rotx_np, rotz_np, convert_camera_pose, get_rot90_camera_matrices


def from_reconstruction(
        reconstruction: Reconstruction,
        fix_up=False
):
    reconstruction = copy(reconstruction)

    if fix_up:
        z_rot = rt2pose(rotx_np(-np.pi / 2)[0])[:3, :3]  # world -y up to +z up

        xyz = np.array([p["xyz"] for p in reconstruction.points.values()])
        xyz = xyz @ z_rot.T

        up_R, up_t = get_upright_transformation(xyz, z_thresh=0.2)  # real +z up according to floor
        world_transform = rt2pose(up_R @ z_rot, up_t)

    frames = []
    for image_id, image in reconstruction.images.items():
        camera_id = image["camera_id"]
        camera = reconstruction.cameras[camera_id]
        camera_model = camera["model"]

        if camera_model == "OPENCV":
            camera = OpenCvCamera(camera["width"], camera["height"], *camera["params"])
        elif camera_model == "PINHOLE":
            camera = PinholeCamera(camera["width"], camera["height"], *camera["params"])
        else:
            raise ValueError(f"Unknown camera model: {camera_model}")

        pose = qt2pose(q=image["qvec"], t=image["tvec"])

        if fix_up:
            pose = world_transform @ pose

        frame = Frame(
            camera=camera,
            image_path=os.path.join(reconstruction.image_dir, image["name"]),
            org_path=os.path.join(reconstruction.image_dir, image["name"]),
            pose=pose,
            seq_id=image_id
        )
        frames.append(frame)

    if fix_up:
        reconstruction.world_transform(R=world_transform[:3, :3], t=world_transform[:3, 3])

    return NuwaDB(
        source="colmap",
        frames=sorted(frames, key=lambda x: x.image_path),
        colmap_reconstruction=reconstruction,
        z_up=fix_up
    )


def from_colmap(
        img_dir,
        colmap_dir,
        fix_up=False
):
    camera_path = os.path.join(colmap_dir, "cameras.txt")

    if not os.path.exists(camera_path):
        camera_bin_path = camera_path.replace("txt", "bin")
        if os.path.exists(camera_bin_path):
            colmap_convert_model(os.path.dirname(camera_path))
        else:
            raise ValueError(f"Camera file {camera_path} does not exist")

    return from_reconstruction(Reconstruction(colmap_dir, image_dir=img_dir), fix_up=fix_up)


def from_polycam(
        polycam_dir,
        new_image_dir=None,
        discard_border_rate=0.0,
        should_be_portrait=False
):
    """
        :param polycam_dir: path to polycam directory or polycam zip
        :param new_image_dir: new image directory, will copy images if not None
        :param discard_border_rate: discard border rate (for black border removal, [0, 0.5])
        :param should_be_portrait: if the captured data should be portrait or not
    """
    if polycam_dir.endswith(".zip"):
        assert new_image_dir is not None and new_image_dir != "", "new_image_dir is required for zip"
        import zipfile
        with zipfile.ZipFile(polycam_dir, 'r') as zip_ref:
            polycam_dir = tempfile.mkdtemp()
            zip_ref.extractall(polycam_dir)

    seq_dir = os.path.join(polycam_dir, "keyframes")
    assert os.path.exists(seq_dir), f"Directory {seq_dir} does not exist"

    if os.path.exists(os.path.join(seq_dir, "corrected_cameras")):
        dir_prefix = "corrected_"

        if discard_border_rate <= 0:
            nuwa.get_logger().warning("Corrected image but discard_border_rate is 0, "
                                      "this could lead to black pixels (try 0.01?)")

        if discard_border_rate > 0:
            assert discard_border_rate <= 1, "discard_border_rate should be in [0, 1]"
            assert new_image_dir is not None, "new_image_dir is required for border removal"

    else:
        assert os.path.exists(os.path.join(seq_dir, "cameras"))
        nuwa.get_logger().warning("Using uncorrected cameras, this is not recommended as "
                                  "poses will be inaccurate and there is not guarantee of world z up")
        if discard_border_rate > 0:
            nuwa.get_logger().warning(f"Using {discard_border_rate=} for uncorrected cameras")
        dir_prefix = ""

    if new_image_dir is not None:
        os.makedirs(new_image_dir, exist_ok=True)

    uids = [a[:-4] for a in sorted(os.listdir(os.path.join(seq_dir, f"{dir_prefix}images")))]
    nimgs = len(uids)

    frames = []

    for i in range(nimgs):
        camera_json = json.load(open(os.path.join(seq_dir, f"{dir_prefix}cameras/{uids[i]}.json")))

        w = camera_json['width']
        h = camera_json['height']
        fx = camera_json['fx']
        fy = camera_json['fy']
        cx = camera_json['cx']
        cy = camera_json['cy']

        pose = np.array([
            [camera_json['t_00'], camera_json['t_01'], camera_json['t_02'], camera_json['t_03']],
            [camera_json['t_10'], camera_json['t_11'], camera_json['t_12'], camera_json['t_13']],
            [camera_json['t_20'], camera_json['t_21'], camera_json['t_22'], camera_json['t_23']],
            [0, 0, 0, 1]
        ])
        pose = convert_camera_pose(pose, "blender", "cv")
        image_path = os.path.abspath(os.path.join(seq_dir, f"{dir_prefix}images/{uids[i]}.jpg"))
        pose = rt2pose(rotx_np(np.pi / 2)[0]) @ pose  # fix up

        if should_be_portrait:
            if w >= h:
                assert new_image_dir is not None and new_image_dir != "", \
                    "new_image_dir is required for portrait"

            else:
                nuwa.get_logger().warning("Image is already portrait, ignoring the `portrait` flag...")
                should_be_portrait = False

        if discard_border_rate > 0:
            image = Image.open(image_path)
            assert image.size == (w, h), f"Image size mismatch: {image.size} vs ({w}, {h})"

            image, (fx, fy, cx, cy) = center_crop_and_update_intrinsics(
                image,
                (fx, fy, cx, cy),
                crop_size=(int(w * (1 - discard_border_rate)), int(h * (1 - discard_border_rate)))
            )

            if should_be_portrait:
                image = image.rotate(270, expand=True)
                pose, fx, fy, cx, cy, w, h = get_rot90_camera_matrices(pose, fx, fy, cx, cy, w, h)
                pose = rt2pose(rotz_np(np.pi)[0]) @ pose  # fix up

            image_path = os.path.abspath(os.path.join(new_image_dir, f"{uids[i]}.jpg"))
            image.save(image_path)

        else:
            if new_image_dir is not None:
                new_image_path = os.path.join(new_image_dir, os.path.basename(image_path))

                if should_be_portrait:
                    image = Image.open(image_path)
                    image = image.rotate(270, expand=True)
                    image.save(new_image_path)
                    pose, fx, fy, cx, cy, w, h = get_rot90_camera_matrices(pose, fx, fy, cx, cy, w, h)
                    pose = rt2pose(rotz_np(np.pi)[0]) @ pose  # fix up
                else:
                    shutil.copy2(image_path, new_image_path)

                image_path = os.path.abspath(new_image_path)

        camera = PinholeCamera(w, h, fx, fy, cx, cy)
        # if dir_prefix == "corrected_":
        #     camera = PinholeCamera(w, h, fx, fy, cx, cy)
        # else:
        #     camera = OpenCvCamera(w, h, fx, fy, cx, cy)

        frame = Frame(
            camera=camera,
            image_path=image_path,
            org_path=image_path,
            pose=pose,
            seq_id=i,
            sharpness_score=camera_json['blur_score']
        )
        frames.append(frame)

    frames = sorted(frames, key=lambda x: x.image_path)

    return NuwaDB(
        source="arkit",
        frames=frames,
        colmap_reconstruction=Reconstruction.from_frames(frames),
        z_up=True
    )


def from_3dscannerapp(
        file_dir,
        new_image_dir=None
):
    """
        :param file_dir: path to 3dscannerapp directory or zip
        :param new_image_dir: new image directory, will copy images if not None
    """
    if file_dir.endswith(".zip"):
        assert new_image_dir is not None and new_image_dir != "", "new_image_dir is required for zip"
        import zipfile
        with zipfile.ZipFile(file_dir, 'r') as zip_ref:
            file_dir = tempfile.mkdtemp()
            zip_ref.extractall(file_dir)
            file_dir = os.path.join(file_dir, os.listdir(file_dir)[0])

    assert os.path.exists(file_dir), f"Directory {file_dir} does not exist"

    if new_image_dir is not None:
        os.makedirs(new_image_dir, exist_ok=True)

    frames = []
    w, h = None, None

    for i in range(99999):
        jpg_name = f"frame_{i:05}.jpg"
        json_name = f"frame_{i:05}.json"

        if not os.path.exists(os.path.join(file_dir, jpg_name)):
            break

        if w is None:
            image = Image.open(os.path.join(file_dir, jpg_name))
            w, h = image.size

        camera_json = json.load(open(os.path.join(file_dir, json_name)))
        pose = np.array(camera_json["cameraPoseARFrame"]).reshape(4, 4)
        pose = convert_camera_pose(pose, "blender", "cv")

        intrinsics = camera_json["intrinsics"]
        image_path = os.path.abspath(os.path.join(file_dir, jpg_name))
        pose = rt2pose(rotx_np(np.pi / 2)[0]) @ pose

        if new_image_dir is not None:
            new_image_path = os.path.join(new_image_dir, os.path.basename(image_path))
            shutil.copy2(image_path, new_image_path)
            image_path = os.path.abspath(new_image_path)

        camera = PinholeCamera(w, h, intrinsics[0], intrinsics[4], intrinsics[2], intrinsics[5])
        frame = Frame(
            camera=camera,
            image_path=image_path,
            org_path=image_path,
            pose=pose,
            seq_id=i,
            sharpness_score=camera_json['motionQuality']
        )
        frames.append(frame)

    frames = sorted(frames, key=lambda x: x.image_path)

    return NuwaDB(
        source="arkit",
        frames=frames,
        colmap_reconstruction=Reconstruction.from_frames(frames),
        z_up=True
    )


def from_dear(
        file_path,
        out_frame_dir,
        should_be_portrait=False,
        sample_stride=1,
):
    """
        (beta) load from dear zip or folder
        :param file_path: path to dear directory or zip
        :param out_frame_dir: output frame directory
        :param should_be_portrait: if the captured data should be portrait or not
        :param sample_stride: sample stride
    """
    nuwa.get_logger().warning("DEAR importer is a beta feature, please proceed with caution.")

    if file_path.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            file_path = tempfile.mkdtemp()
            zip_ref.extractall(file_path)

    img_dir = os.path.join(file_path, "frames")
    os.makedirs(img_dir, exist_ok=True)

    # find all mp4 in file_path
    mp4_files = [f for f in os.listdir(file_path) if (f.endswith(".mp4") and not f.endswith("_depth.mp4"))]
    if len(mp4_files) == 0:
        raise ValueError("No sequence (.mp4) found in the directory")
    if len(mp4_files) > 1:
        raise ValueError("Multiple sequences (.mp4) found in the directory")

    sequence_file = os.path.abspath(os.path.join(file_path, mp4_files[0]))
    if os.path.exists(out_frame_dir):
        nuwa.get_logger().warning("Output frame directory exists, overwriting...")

    os.makedirs(out_frame_dir, exist_ok=True)

    metadata = json.load(open(os.path.join(file_path, sequence_file.replace(".mp4", "_pose.json"))))
    nuwa.get_logger().info(f"Found {len(metadata['frames'])} frames in the sequence")

    w, h = metadata['width'], metadata['height']
    data = metadata['frames'][::sample_stride]

    frames = []
    for i, f in enumerate(tqdm.tqdm(data, desc="Extracting DEAR frames", unit="f")):
        transform = np.array(f['transform']).T
        time = f['time']
        intrinsic = np.array(f['intrinsic']).T

        frame_path = os.path.join(out_frame_dir, f'{i:04d}.png')

        timestamp = np.round(time * 1000)
        str_hour = str(int(timestamp / 3600000)).zfill(2)
        str_min = str(int(int(timestamp % 3600000) / 60000)).zfill(2)
        str_sec = str(int(int(int(timestamp % 3600000) % 60000) / 1000)).zfill(2)
        str_mill = str(int(int(int(timestamp % 3600000) % 60000) % 1000)).zfill(3)
        str_timestamp = str_hour+":"+str_min+":"+str_sec+"."+str_mill

        try:
            do_system([
                'ffmpeg', '-y', '-ss', str_timestamp, '-i', sequence_file, '-vframes', '1', '-f', 'image2', frame_path
            ])
        except Exception as e:
            nuwa.get_logger().error(f"fail to process {sequence_file}: {e}")
            shutil.rmtree(out_frame_dir)
            raise

        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        pose = convert_camera_pose(transform, "blender", "cv")
        pose = rt2pose(rotx_np(np.pi / 2)[0]) @ pose  # fix up

        image = Image.open(frame_path)
        if should_be_portrait:
            pose, fx, fy, cx, cy, w, h = get_rot90_camera_matrices(pose, fx, fy, cx, cy, w, h)
            pose = rt2pose(rotz_np(np.pi)[0]) @ pose  # fix up

        camera = PinholeCamera(w, h, fx, fy, cx, cy)
        image_path = os.path.abspath(frame_path)

        frame = Frame(
            camera=camera,
            image_path=image_path,
            org_path=image_path,
            pose=pose,
            seq_id=i,
            sharpness_score=sharpness(image)
        )
        frames.append(frame)

    frames = sorted(frames, key=lambda x: x.image_path)

    return NuwaDB(
        source="arkit",
        frames=frames,
        colmap_reconstruction=Reconstruction.from_frames(frames),
        z_up=True
    )


def from_nuwadb(path):
    db = json.load(open(path))
    frames = sorted([Frame.from_dict(f, root=os.path.dirname(path)) for f in db["frames"]], key=lambda x: x.image_path)

    z_up = None
    if "up" in db:
        z_up = np.allclose(np.array(db["up"]), np.array([0, 0, 1]))

    return NuwaDB(
        source=db["source"],
        frames=frames,
        colmap_reconstruction=Reconstruction.from_frames(frames),
        scale_denorm=db.get("scale_denorm", None),
        z_up=True if z_up else None
    )


def from_image_folder(
        img_dir,
        method="colmap",
        matcher="sequential",
        single_camera=True,
        camera_model="OPENCV",
        camera_heuristics=None,
        camera_run_undistort=True,
        undistort_image_dir="",
        colmap_out_dir="",
        colmap_binary="colmap",
        colmap_loop_detection=True,
        colmap_fix_up=True,
        hloc_max_keypoints=20000,
        hloc_use_pixsfm=False
):
    """
    :param img_dir: path to image directory
    :param method: colmap or hloc
    :param matcher: exhaustive or sequential
    :param single_camera: single camera for all images or not
    :param camera_model: camera model [OPENCV, PINHOLE]
    :param camera_heuristics: camera heuristics
    :param camera_run_undistort: run undistort on images
    :param undistort_image_dir: output directory for undistorted images
    :param colmap_out_dir: output directory for colmap data
    :param colmap_binary: path to colmap binary
    :param colmap_loop_detection: run loop detection in colmap
    :param colmap_fix_up: rotate up to +z
    :param hloc_max_keypoints: max keypoints for hloc
    :param hloc_use_pixsfm: use pixsfm for hloc
    """

    assert method in ["colmap", "hloc"]
    assert camera_model in ["OPENCV", "PINHOLE"]

    if camera_run_undistort:
        assert camera_model == "OPENCV"
        if undistort_image_dir == "":
            undistort_image_dir = os.path.join(img_dir, "undistort")

    if colmap_out_dir:
        os.makedirs(colmap_out_dir, exist_ok=True)
    else:
        import tempfile
        colmap_out_dir = tempfile.mkdtemp()

    if method == "colmap":
        run_colmap(
            img_dir,
            colmap_out_dir,
            matcher=matcher,
            camera_model=camera_model,
            heuristics=camera_heuristics,
            colmap_binary=colmap_binary,
            single_camera=single_camera,
            loop_detection=colmap_loop_detection
        )

    else:
        run_hloc(
            img_dir,
            colmap_out_dir,
            matcher=matcher,
            camera_model=camera_model,
            heuristics=camera_heuristics,
            colmap_binary=colmap_binary,
            single_camera=single_camera,
            max_keypoints=hloc_max_keypoints,
            use_pixsfm=hloc_use_pixsfm
        )

    old_sparse_dir = os.path.join(colmap_out_dir, "sparse")
    new_sparse_dir = os.path.join(colmap_out_dir, "sparse")

    if camera_run_undistort:
        undistort_dir = os.path.join(colmap_out_dir, "undistort")
        colmap_undistort_images(
            img_dir,
            os.path.join(old_sparse_dir, "0"),
            undistort_dir,
            colmap_binary=colmap_binary
        )
        if os.path.exists(undistort_image_dir):
            shutil.rmtree(undistort_image_dir)
        shutil.move(os.path.join(undistort_dir, "images"), undistort_image_dir)
        new_sparse_dir = os.path.join(undistort_dir, "sparse")
        img_dir = undistort_image_dir

    n_models = len(os.listdir(old_sparse_dir))
    if n_models == 0:
        nuwa.get_logger().error("no models generated")
        return None

    if n_models > 1:
        nuwa.get_logger().warning("multiple models generated")
        nuwa.get_logger().warning("only model 0 is processed")

    return from_colmap(img_dir, os.path.join(new_sparse_dir, "0"), fix_up=colmap_fix_up)


def from_video(
        video_path,
        out_img_dir,
        fps=3,
        method="colmap",
        single_camera=True,
        camera_model="OPENCV",
        camera_heuristics=None,
        camera_run_undistort=True,
        colmap_out_dir="",
        colmap_binary="colmap",
        colmap_loop_detection=True,
        colmap_fix_up=True,
        hloc_max_keypoints=20000,
        hloc_use_pixsfm=False
):
    """
    :param video_path: path to video
    :param out_img_dir: output directory for extracted images
    :param fps: frames to extract per second
    :param method: colmap or hloc
    :param single_camera: single camera for all frames or not
    :param camera_model: camera model [OPENCV, PINHOLE]
    :param camera_heuristics: camera heuristics
    :param camera_run_undistort: run undistort on images
    :param colmap_out_dir: output directory for colmap data
    :param colmap_binary: path to colmap binary
    :param colmap_loop_detection: run loop detection in colmap
    :param colmap_fix_up: rotate up to +z
    :param hloc_max_keypoints: max keypoints for hloc
    :param hloc_use_pixsfm: use pixsfm for hloc
    """
    if not single_camera:
        nuwa.get_logger().warning("Using multiple cameras for a video, "
                                  "please make sure this is what you want")

    os.makedirs(out_img_dir, exist_ok=True)
    run_ffmpeg(video_path, out_img_dir, fps)

    return from_image_folder(
        out_img_dir,
        method=method,
        matcher="sequential",
        single_camera=True,
        camera_model=camera_model,
        camera_heuristics=camera_heuristics,
        camera_run_undistort=camera_run_undistort,
        colmap_out_dir=colmap_out_dir,
        colmap_binary=colmap_binary,
        colmap_loop_detection=colmap_loop_detection,
        colmap_fix_up=colmap_fix_up,
        hloc_max_keypoints=hloc_max_keypoints,
        hloc_use_pixsfm=hloc_use_pixsfm,
    )
