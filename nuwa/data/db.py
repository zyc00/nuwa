import json
import os
import shutil
import tempfile
from copy import deepcopy
from typing import List

import cv2
import numpy as np
import tqdm
from PIL import Image

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

    def __init__(self, source="", frames=None, colmap_reconstruction=None, scale_denorm=None):
        self.source = source
        self.frames = [] if frames is None else frames
        self.colmap_reconstruction = deepcopy(colmap_reconstruction)
        self.scale_denorm = scale_denorm

    def __repr__(self):
        return {
            "source": self.source,
            "colmap_reconstruction": "None" if self.colmap_reconstruction is None else "[Valid Reconstruction]",
            "frames": [f"{len(self.frames)} Frames..."],
            "scale_denorm": "...Data Not Normalized..." if self.scale_denorm is None else self.scale_denorm
        }.__repr__()

    def get_up(self):
        # legacy
        up = np.zeros(3)
        for f in self.frames:
            up += -f.pose[:3, 1]
        up = up / np.linalg.norm(up)

        # check if close to (0, 0, 1)
        if up[2] < 0.95:
            nuwa.get_logger().warning(f"avg up {tuple(up)} is not close to +z, "
                                      f"the camera poses might need further post-processing.")
            # TODO: write this post-processing
        else:
            nuwa.get_logger().info(f"avg up {tuple(up)}")

        if self.source == "colmap":
            return up
        elif self.source == "arkit":
            return np.array([0., 0., 1.])
        elif self.source == "polycam":
            nuwa.get_logger().warning("deprecated source name 'polycam' detected, assuming 'arkit'")
            return np.array([0., 0., 1.])
        else:
            raise ValueError(f"Unknown source {self.source}")

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
            "up": self.get_up().tolist(),
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
        :param adjust_cameras:
        :param copy_org:
        :return:
        """
        from nuwa.utils.seg_utils import segment_img, sam, scene_carving, crop_images, SAMAPI
        from nuwa.utils.dmv_utils import raft_api

        # TODO: fix this
        if self.colmap_reconstruction is not None and self.source == "colmap":
            nuwa.get_logger().warning(f"in the current version, colmap reconstruction will break after masking")

        os.makedirs(mask_save_dir, exist_ok=True)
        os.makedirs(masked_image_save_dir, exist_ok=True)

        mask_save_dir = os.path.abspath(mask_save_dir)
        masked_image_save_dir = os.path.abspath(masked_image_save_dir)

        masks = []
        images = []
        for i, frame in tqdm.tqdm(enumerate(self.frames), desc='masking'):
            img = Image.open(frame.image_path)

            if i == 0:
                _, rembg_mask = segment_img(img)
                _, mask = sam(img, rembg_mask)
            else:
                flow_rgb_img0 = Image.open(self.frames[i - 1].image_path).reduce(reduce_factor)
                flow_rgb_img1 = img.reduce(reduce_factor)
                flow = raft_api.raft_optical_flow_api(flow_rgb_img0, flow_rgb_img1).cpu().numpy()

                ref_mask = Image.fromarray(masks[i - 1].astype(np.uint8)).reduce(reduce_factor)
                ref_mask = np.array(ref_mask) > 0

                ys, xs = ref_mask.nonzero()
                fg_pixels = np.concatenate([xs[:, None], ys[:, None]], axis=1)
                fg_pixels = fg_pixels.astype(np.float32)
                fg_pixels = fg_pixels + flow[ys, xs]
                xmin, xmax = fg_pixels[:, 0].min(), fg_pixels[:, 0].max()
                ymin, ymax = fg_pixels[:, 1].min(), fg_pixels[:, 1].max()
                w, h = xmax - xmin, ymax - ymin
                xmin = int(xmin - w * shrink)
                xmax = int(xmax + w * shrink)
                ymin = int(ymin - h * shrink)
                ymax = int(ymax + h * shrink)
                bbox = xmin, ymin, xmax, ymax
                bbox = [int(x * reduce_factor) for x in bbox]
                mask = SAMAPI.segment_api(np.array(img), bbox=bbox, sam_checkpoint=sam_ckpt_path)
                retval, labels, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
                try:
                    maxcomp = np.argmax(stats[1:, 4]) + 1
                    mask = (labels == maxcomp)
                except ValueError:
                    mask = np.ones_like(mask)

            images.append(np.array(img))
            masks.append(mask)

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

    def finetune_pose(self, ingp_binary="instant-ngp"):
        tmp_dump = tempfile.mkdtemp()
        tmp_json = os.path.join(tmp_dump, "nuwa_db.json")
        self.dump(tmp_json)

        nuwa.get_logger().info(f"ingp - "
                               f"please use GUI to perform extrinsic optimization and dump pose now (check no quat)`")
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
            nuwa.get_logger().debug(f"ingp - fine-tuned frame {i}: {err_r=:.3f}, {err_t=:.5f}")

            err_r_max = max(err_r_max, err_r)
            err_t_max = max(err_t_max, err_t)

            f.pose = pose

        nuwa.get_logger().info(f"ingp - finetune results: {err_r_max=:.3f}, {err_t_max=:.5f}")
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

        nuwa.get_logger().info("colmap ft - up before fine-tuning:")
        self.get_up()

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
        nuwa.get_logger().info("colmap ft - up after fine-tuning:")
        self.get_up()

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

            nuwa.get_logger().info("up after undistort:")
            self.get_up()

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
