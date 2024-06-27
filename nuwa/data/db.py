import json
import os
import shutil
import tempfile
from copy import deepcopy
from typing import List

import sys
import time
import numpy as np
import tqdm
from PIL import Image
from PIL.Image import Resampling

import nuwa
from nuwa.data.colmap import Reconstruction
from nuwa.data.frame import Frame
from nuwa.utils.colmap_utils import run_colmap, colmap_undistort_images
from nuwa.utils.os_utils import do_system
from nuwa.utils.pose_utils import convert_camera_pose


class NuwaDB:
    source: str
    frames: List[Frame]
    colmap_reconstruction: Reconstruction | None

    scale_denorm: float | None     # scale_denorm for normalizing the scene into, typically into (-1, 1)
    z_up: bool

    def __init__(self, source="", frames=None, colmap_reconstruction=None, scale_denorm=None, z_up=None):
        self.source = source
        if self.source == "polycam":
            nuwa.get_logger().warning("'polycam' source tag is deprecated, please use 'arkit' instead.")
        else:
            assert self.source in ["arkit", "colmap"], f"Unknown data source {self.source}."

        self.frames = [] if frames is None else frames
        self.colmap_reconstruction = deepcopy(colmap_reconstruction)
        self.scale_denorm = scale_denorm

        if z_up is None:
            self.z_up = self.source in ["arkit", "polycam"]
        else:
            self.z_up = z_up

    def __repr__(self):
        return {
            "source": self.source,
            "colmap_reconstruction": "None" if self.colmap_reconstruction is None else "[Valid Reconstruction]",
            "frames": [f"{len(self.frames)} Frames..."],
            "scale_denorm": "...Data Not Normalized..." if self.scale_denorm is None else self.scale_denorm
        }.__repr__()

    def get_world_up(self):
        # calc camera avg up
        up = np.zeros(3)
        for f in self.frames:
            up += -f.pose[:3, 1]
        up = up / np.linalg.norm(up)

        nuwa.get_logger().info(f"get_world_up - Camera avg up {tuple(up)}")

        if np.abs(up).max() != abs(up[2]):
            nuwa.get_logger().warning(f"get_world_up - It seems that the camera avg up is not close to +z. "
                                      f"You may want to check the gravity direction manually.")
        elif up[2] < 0.9:
            nuwa.get_logger().warning(f"get_world_up - Camera avg up {tuple(up)} is not close to +z, "
                                      f"this is likely due to biased capturing.")

        if self.z_up:
            nuwa.get_logger().info("get_world_up - DB has been marked z_up, world up is set to +z.")
            return np.array([0., 0., 1.])

        else:
            if self.source != "colmap":
                nuwa.get_logger().warning(f"get_world_up - Data sourced from {self.source}, but not marked z_up. "
                                          f"This is likely a bug.")
            return up

    def dump(self,
             out_json_path,
             copy_images_to=None,
             copy_masks_to=None,
             dump_reconstruction_to=None):
        if copy_images_to is not None:
            os.makedirs(copy_images_to, exist_ok=True)
            copy_images_to = os.path.abspath(copy_images_to)

            for f in self.frames:
                new_path = os.path.join(copy_images_to, os.path.basename(f.image_path))
                shutil.copy2(f.image_path, new_path)
                f.image_path = new_path

        if copy_masks_to is not None:
            os.makedirs(copy_masks_to, exist_ok=True)
            copy_masks_to = os.path.abspath(copy_masks_to)

            for f in self.frames:
                if f.mask_path:
                    new_path = os.path.join(copy_masks_to, os.path.basename(f.mask_path))
                    shutil.copy2(f.mask_path, new_path)
                    f.mask_path = new_path

        frames = [f.to_dict() for f in self.frames]
        for f in frames:
            f["file_path"] = os.path.relpath(
                f["file_path"], start=os.path.dirname(out_json_path))
            f["org_path"] = os.path.relpath(
                f["org_path"], start=os.path.dirname(out_json_path))
            if f["mask_path"]:
                f["mask_path"] = os.path.relpath(
                    f["mask_path"], start=os.path.dirname(out_json_path))

        db = {
            "source": self.source,
            "up": self.get_world_up().tolist(),
        }

        if self.scale_denorm is not None:
            db["scale_denorm"] = self.scale_denorm

        if dump_reconstruction_to is not None:
            self.dump_reconstruction(dump_reconstruction_to)
            db["colmap_path"] = os.path.relpath(
                dump_reconstruction_to, start=os.path.dirname(out_json_path))

        db["frames"] = frames

        with open(out_json_path, "w") as outfile:
            json.dump(db, outfile, indent=4)

        with open(out_json_path + ".txt", "w") as outfile:
            outfile.write(f"{os.path.basename(out_json_path)}")

    def dump_reconstruction(self, out_dir):
        assert self.colmap_reconstruction is not None
        self.colmap_reconstruction.dump(out_dir)

    def calculate_object_mask(
            self,
            mask_save_dir,
            masked_image_save_dir,
            reduce_factor=2,
            shrink=0.02,
            sam_ckpt_path=None,
            use_flow=True,
            adjust_cameras=True,
            copy_org=True
    ):
        """
        Calculate object masks for each frame
        If adjust_cameras is True, the cameras will be normalized and adjusted to fit the masked images.

        :param mask_save_dir:
        :param masked_image_save_dir:
        :param reduce_factor:
        :param shrink:
        :param sam_ckpt_path:
        :param use_flow:
        :param adjust_cameras:
        :param copy_org:
        :return:
        """
        from nuwa.utils.seg_utils import scene_carving, crop_images, segment_fg

        # TODO: fix this
        if self.colmap_reconstruction is not None and self.source == "colmap":
            nuwa.get_logger().warning(f"nseg - in the current version, colmap reconstruction will break after masking")

        os.makedirs(mask_save_dir, exist_ok=True)
        os.makedirs(masked_image_save_dir, exist_ok=True)

        mask_save_dir = os.path.abspath(mask_save_dir)
        masked_image_save_dir = os.path.abspath(masked_image_save_dir)
        org_images = [Image.open(f.image_path) for f in self.frames]

        images, masks = segment_fg(
            org_images,
            use_flow=use_flow,
            sam_ckpt_path=sam_ckpt_path,
            reduce_factor=reduce_factor,
            shrink=shrink,
        )

        if copy_org:
            for i, f in enumerate(self.frames):
                # TODO: masked_org_image_path
                new_path = os.path.join(masked_image_save_dir, os.path.basename(f.org_path))
                old_image = Image.open(f.org_path)
                new_image = Image.new(old_image.mode, old_image.size, 0)
                new_image.paste(old_image, mask=Image.fromarray(masks[i]))
                new_image.save(new_path)
                f.org_path = new_path

                base_name = os.path.basename(f.org_path).split(".")[0]
                mask_path = os.path.join(mask_save_dir, f"{base_name}.png")
                Image.fromarray(masks[i]).save(mask_path)
                # TODO: save org_mask

        if adjust_cameras:
            nuwa.get_logger().info(f"nseg - normalizing the scene...")
            self.normalize_cameras(positive_z=True, scale_factor=1.0)
            masks = np.array(masks)
            ks = np.array([f.camera.intrinsic_matrix for f in self.frames])
            camera_poses = np.array([f.pose for f in self.frames])
            camera_poses, center, scale = scene_carving(masks, ks, camera_poses)
            self.colmap_reconstruction.world_translate(-center)
            self.colmap_reconstruction.world_scale(scale)
            self.scale_denorm *= scale
            images, ks, masks = crop_images(images, masks, ks)

            h, w = images[0].shape[:2]

            # dump images
            for i, img in enumerate(images):
                Image.fromarray(img).save(os.path.join(masked_image_save_dir, f"{i:06d}.png"))

            # update info
            for i in range(len(self.frames)):
                assert self.frames[i].camera.type == "PINHOLE"
                self.frames[i].pose = camera_poses[i]
                self.frames[i].image_path = os.path.join(masked_image_save_dir, f"{i:06d}.png")
                self.frames[i].camera.w = w
                self.frames[i].camera.h = h
                self.frames[i].camera.fx = ks[i][0, 0]
                self.frames[i].camera.fy = ks[i][1, 1]
                self.frames[i].camera.cx = ks[i][0, 2]
                self.frames[i].camera.cy = ks[i][1, 2]

        for i, mask in enumerate(masks):
            mask_path = os.path.join(mask_save_dir, f"{i:06d}.png")
            Image.fromarray(mask).save(mask_path)
            self.frames[i].mask_path = mask_path

        # TODO: filter points w.r.t. masks

        return masks

    def undistort_images(self):
        raise NotImplementedError

    def normalize_cameras(self, positive_z=True, scale_factor=0.9):
        # TODO: Rewrite this function with camera near far

        camera_poses = np.array([f.pose for f in self.frames])
        xm, ym, zm = camera_poses[:, :3, 3].min(0)
        xM, yM, zM = camera_poses[:, :3, 3].max(0)

        if positive_z:
            offset = np.array([(xm + xM) / 2, (ym + yM) / 2, zm])
        else:
            offset = np.array([(xm + xM) / 2, (ym + yM) / 2, (zm + zM) / 2])
        camera_poses[:, :3, 3] -= offset
        self.colmap_reconstruction.world_translate(-offset)

        scale = scale_factor / np.linalg.norm(camera_poses[:, :3, 3], axis=-1).max()
        camera_poses[:, :3, 3] *= scale
        self.colmap_reconstruction.world_scale(scale)

        for i, f in enumerate(self.frames):
            f.pose = camera_poses[i]

        if self.scale_denorm is None:
            self.scale_denorm = scale
        else:
            self.scale_denorm *= scale

        nuwa.get_logger().info(f"db norm - normalized the scene, org_ctr={tuple(offset)}, org_scale={1 / scale}")

    def finetune_pose(self, ingp_home, near=-1.0, n_steps=4000):
        ingp_home = os.path.expanduser(ingp_home)
        sys.path.append(os.path.join(ingp_home, "build"))
        assert nuwa.is_ingp_available(), "Pose fine-tuning requires ingp, please install it following README."
        import pyngp as ingp

        tmp_dump = tempfile.mkdtemp()
        tmp_json = os.path.join(tmp_dump, "nuwa_db.json")
        self.dump(tmp_json)

        testbed = ingp.Testbed()
        testbed.root_dir = tmp_dump
        testbed.load_file(tmp_json)
        testbed.load_file(os.path.join(ingp_home, "configs/nerf/base.json"))
        testbed.nerf.training.optimize_extrinsics = True
        # testbed.nerf.training.optimize_exposure = True
        testbed.nerf.training.random_bg_color = True
        testbed.nerf.render_with_lens_distortion = True
        if near >= 0.0:
            testbed.nerf.training.near_distance = near
        testbed.shall_train = True

        old_training_step = 0
        tqdm_last_update = 0
        with tqdm.tqdm(desc="INFO - ingp ft - fine-tuning", total=n_steps) as t:
            while testbed.frame():
                if testbed.training_step >= n_steps:
                    break

                if testbed.training_step < old_training_step or old_training_step == 0:
                    old_training_step = 0
                    t.reset()

                now = time.monotonic()
                if now - tqdm_last_update > 1.:
                    t.update(testbed.training_step - old_training_step)
                    t.set_postfix(loss=testbed.loss)
                    old_training_step = testbed.training_step
                    tqdm_last_update = now

        err_r_max, err_t_max = 0., 0.
        self.frames = sorted(self.frames, key=lambda x: x.image_path)
        for i, f in enumerate(self.frames):
            refined_ex = testbed.nerf.training.get_camera_extrinsics(i)
            pose = np.concatenate((np.array(refined_ex), np.array([[0, 0, 0, 1]])), axis=0)
            pose = convert_camera_pose(pose, "blender", "cv")

            err = f.pose @ np.linalg.inv(pose)
            err_r = np.arccos((np.trace(err[:3, :3]) - 1) / 2) * 180 / np.pi
            err_t = np.linalg.norm(err[:3, 3])
            nuwa.get_logger().debug(f"ingp ft - fine-tuned frame {i}: {err_r=:.3f}, {err_t=:.4f}")

            err_r_max = max(err_r_max, err_r)
            err_t_max = max(err_t_max, err_t)

            f.pose = pose

        nuwa.get_logger().info(f"ingp ft - finetune results: {err_r_max=:.3f}, {err_t_max=:.4f}")
        if err_r_max > 2.0 or err_t_max > 0.1:
            nuwa.get_logger().warning(f"ingp ft - large errors detected, please check the results manually...")

        self.colmap_reconstruction.update_poses_from_frames(self.frames)
        shutil.rmtree(tmp_dump)

    def _finetune_pose_old(self, ingp_binary="instant-ngp"):
        assert nuwa.is_ingp_available(), "Pose fine-tuning requires ingp, please install it following README."

        tmp_dump = tempfile.mkdtemp()
        tmp_json = os.path.join(tmp_dump, "nuwa_db.json")
        self.dump(tmp_json)

        nuwa.get_logger().info(f"ingp ft - "
                               f"please use GUI to perform extrinsic optimization and dump pose now (check no quat)")
        do_system((ingp_binary, tmp_json, "--no-train"))

        refined_json = os.path.join(tmp_dump, "nuwa_db_base_extrinsics.json")
        refined_json = json.load(open(refined_json, "r"))

        err_r_max = 0
        err_t_max = 0
        self.frames = sorted(self.frames, key=lambda x: x.image_path)
        for i, f in enumerate(self.frames):
            refined = refined_json[i]
            assert i == refined["id"]

            pose = np.concatenate((np.array(refined["transform_matrix"]), np.array([[0, 0, 0, 1]])), axis=0)
            pose = convert_camera_pose(pose, "blender", "cv")

            err = f.pose @ np.linalg.inv(pose)
            err_r = np.arccos((np.trace(err[:3, :3]) - 1) / 2) * 180 / np.pi
            err_t = np.linalg.norm(err[:3, 3])
            nuwa.get_logger().debug(f"ingp ft - fine-tuned frame {i}: {err_r=:.3f}, {err_t=:.4f}")

            err_r_max = max(err_r_max, err_r)
            err_t_max = max(err_t_max, err_t)

            f.pose = pose

        nuwa.get_logger().info(f"ingp ft - finetune results: {err_r_max=:.3f}, {err_t_max=:.4f}")
        self.colmap_reconstruction.update_poses_from_frames(self.frames)
        shutil.rmtree(tmp_dump)

    def finetune_pose_colmap(
            self,
            matcher="exhaustive",
            colmap_binary="colmap",
            loop_detection=True,
            undistort=False
    ):
        """
        Return a new database with the poses fine-tuned using colmap

        :param matcher:
        :param colmap_binary:
        :param loop_detection:
        :param undistort:
        :return:
        """

        if self.colmap_reconstruction is None:
            raise ValueError("No colmap reconstruction found")

        if self.source == "colmap":
            nuwa.get_logger().warning("colmap ft - fine-tuning with colmap on a colmap sourced database")

        nuwa.get_logger().info("colmap ft - camera avg up before fine-tuning:")
        self.get_world_up()

        colmap_in_dir = tempfile.mkdtemp()
        self.dump_reconstruction(colmap_in_dir)
        colmap_out_dir = tempfile.mkdtemp()

        single_camera = (len(self.colmap_reconstruction.cameras) == 1)

        run_colmap(
            image_dir=self.colmap_reconstruction.image_dir,
            out_dir=colmap_out_dir,
            matcher=matcher,
            camera_model=self.frames[0].camera.type,
            heuristics=','.join(map(str, self.frames[0].camera.params)),
            colmap_binary=colmap_binary,
            single_camera=single_camera,
            loop_detection=loop_detection,
            from_db=None,
            db_only=True
        )

        shutil.rmtree(colmap_in_dir)
        self.colmap_reconstruction.reorder_from_db(os.path.join(colmap_out_dir, "database.db"))

        colmap_in_dir = tempfile.mkdtemp()
        self.dump_reconstruction(colmap_in_dir)

        nuwa.get_logger().info("colmap ft - point triangulation")
        do_system((f"{colmap_binary}", "point_triangulator",
                   f"--database_path={os.path.join(colmap_out_dir, 'database.db')}",
                   f"--image_path={self.colmap_reconstruction.image_dir}",
                   f"--input_path={colmap_in_dir}",
                   f"--output_path={colmap_out_dir}",
                   f"--Mapper.ba_refine_focal_length={int(single_camera)}",
                   f"--Mapper.ba_refine_principal_point={int(single_camera)}",
                   f"--Mapper.ba_refine_extra_params={int(single_camera)}",
                   f"--Mapper.ba_global_function_tolerance=0.000001",
                   f"--Mapper.fix_existing_images=0"))

        run_colmap(
            image_dir=self.colmap_reconstruction.image_dir,
            out_dir=colmap_out_dir,
            in_dir=colmap_out_dir,
            matcher=matcher,
            camera_model=self.frames[0].camera.type,
            heuristics=','.join(map(str, self.frames[0].camera.params)),
            colmap_binary=colmap_binary,
            single_camera=single_camera,
            loop_detection=loop_detection,
            from_db=os.path.join(colmap_out_dir, "database.db"),
            db_only=False,
            fix_image_pose=False,
            fix_intrinsics=not single_camera
        )

        from nuwa import from_colmap
        new_db = from_colmap(self.colmap_reconstruction.image_dir, os.path.join(colmap_out_dir, "sparse"), fix_up=False)

        self.frames = new_db.frames
        self.colmap_reconstruction = new_db.colmap_reconstruction
        self.scale_denorm = new_db.scale_denorm

        nuwa.get_logger().info(f"colmap ft - fine-tuning done, "
                               f"reconstruction contains {len(self.colmap_reconstruction.points)} points")
        nuwa.get_logger().info("colmap ft - camera avg up after fine-tuning:")
        self.get_world_up()

        if undistort and self.frames[0].camera.type == "OPENCV":
            undistort_dir = tempfile.mkdtemp()
            undistort_image_dir = os.path.join(self.colmap_reconstruction.image_dir, "undistort")

            colmap_undistort_images(
                self.colmap_reconstruction.image_dir,
                os.path.join(colmap_out_dir, "sparse"),
                undistort_dir,
                colmap_binary=colmap_binary
            )

            if os.path.exists(undistort_image_dir):
                shutil.rmtree(undistort_image_dir)

            shutil.move(os.path.join(undistort_dir, "images"), undistort_image_dir)

            new_db = from_colmap(undistort_image_dir, os.path.join(undistort_dir, "sparse/0"), fix_up=False)
            self.frames = new_db.frames
            self.colmap_reconstruction = new_db.colmap_reconstruction
            self.scale_denorm = new_db.scale_denorm

            nuwa.get_logger().info("camera avg up after undistort:")
            self.get_world_up()

            shutil.rmtree(undistort_dir)

        shutil.rmtree(colmap_in_dir)
        shutil.rmtree(colmap_out_dir)

    def generate_points_colmap(
            self,
            matcher="exhaustive",
            colmap_binary="colmap",
            loop_detection=True
    ):
        image_dir = os.path.dirname(self.frames[0].image_path)
        heuristics = ','.join(map(str, self.frames[0].camera.params))

        colmap_in_dir = tempfile.mkdtemp()
        self.dump_reconstruction(colmap_in_dir)
        colmap_out_dir = tempfile.mkdtemp()

        run_colmap(
            image_dir=image_dir,
            out_dir=colmap_out_dir,
            matcher=matcher,
            camera_model=self.frames[0].camera.type,
            heuristics=heuristics,
            colmap_binary=colmap_binary,
            single_camera=False,
            loop_detection=loop_detection,
            from_db=None,
            db_only=True
        )
        self.colmap_reconstruction.reorder_from_db(os.path.join(colmap_out_dir, "database.db"))
        shutil.rmtree(colmap_in_dir)
        colmap_in_dir = tempfile.mkdtemp()
        self.dump_reconstruction(colmap_in_dir)

        nuwa.get_logger().info("colmap points - Triangulating...")
        do_system((f"{colmap_binary}", "point_triangulator",
                   f"--database_path={os.path.join(colmap_out_dir, 'database.db')}",
                   f"--image_path={image_dir}",
                   f"--input_path={colmap_in_dir}",
                   f"--output_path={colmap_out_dir}",
                   f"--Mapper.ba_refine_principal_point=1",
                   f"--Mapper.ba_global_function_tolerance=0.000001",
                   f"--Mapper.fix_existing_images=0"))

        self.colmap_reconstruction = Reconstruction.from_colmap(colmap_out_dir)

    def downsample_images(self, resolution, new_image_dir, new_mask_dir=None):
        """
        Downsample images to a specific resolution
        If resolution <= 64, it will be treated as a scale factor,
        else it is the longer side of the downsampled image
        """
        os.makedirs(new_image_dir, exist_ok=True)

        for f in tqdm.tqdm(self.frames, desc="Downsampling images"):
            img = Image.open(f.image_path)
            w, h = img.size

            if resolution <= 64:
                w = int(w / resolution)
                h = int(h / resolution)
                reduce_factor = resolution
            else:
                if w > h:
                    h = int(resolution * h / w)
                    w = resolution
                else:
                    w = int(resolution * w / h)
                    h = resolution
                reduce_factor = w / img.size[0]

            nuwa.get_logger().debug(f"downsample - {reduce_factor=}")

            img = img.resize((w, h), Resampling.LANCZOS)
            img.save(os.path.join(new_image_dir, os.path.basename(f.image_path)))
            f.image_path = os.path.join(new_image_dir, os.path.basename(f.image_path))

            if f.mask_path:
                mask = Image.open(f.mask_path)
                mask = mask.resize((w, h), Resampling.NEAREST)
                mask.save(os.path.join(new_mask_dir, os.path.basename(f.mask_path)))
                f.mask_path = os.path.join(new_mask_dir, os.path.basename(f.mask_path))

            f.camera.w = w
            f.camera.h = h
            f.camera.fx *= reduce_factor
            f.camera.fy *= reduce_factor
            f.camera.cx *= reduce_factor
            f.camera.cy *= reduce_factor

        if self.colmap_reconstruction is not None:
            nuwa.get_logger().warning("downsample - 'image' and 'camera' db in colmap reconstruction "
                                      "will be broken after downsample")
