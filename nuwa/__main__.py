import os
import sys
import argparse
import tempfile

import nuwa
from nuwa import (
    from_image_folder, from_video,
    from_polycam, from_3dscannerapp, from_nuwadb, from_dear, from_haoyang
)


def main():
    def get_args():
        parser = argparse.ArgumentParser(description="Nuwa: 3D Reconstruction Pipeline")
        parser.add_argument("--video-path", "-v", type=str, default="",
                            help="Path to the video file")
        parser.add_argument("--image-dir", "-i", type=str, default="",
                            help="Path to the images / to expand the frames / to export the polycam images")
        parser.add_argument("--polycam-path", "-p", type=str, default="",
                            help="Path to the polycam dir or zip")
        parser.add_argument("--scannerapp-path", "-s", type=str, default="",
                            help="Path to the 3dscannerapp dir or zip")
        parser.add_argument("--dear-path", "-d", type=str, default="",
                            help="Path to the DEAR dir or zip")
        parser.add_argument("--haoyang-path", "-H", type=str, default="",
                            help="Path to the Haoyang data dir or zip")
        parser.add_argument("--out-dir", "-o", type=str, default="./nuwa_results",
                            help="Output directory")

        parser.add_argument("--fps", type=float, default=3, help="FPS for video inputs")
        parser.add_argument("--dear-sample-stride", type=int, default=1, help="Stride for DEAR frame sampling")
        parser.add_argument("--discard-border-rate", type=float, default=0.0)
        parser.add_argument("--portrait", action="store_true",
                            help="For polycam, indicating images should be portrait (rot90)")

        parser.add_argument("--finetune-pose", action="store_true",
                            help="Fine-tune pose with ingp, requiring normalized scenes")
        parser.add_argument("--finetune-pose-colmap", action="store_true",
                            help="Fine-tune pose with colmap")
        parser.add_argument("--ingp-home", type=str, default="~/.cache/nuwa/ingp",
                            help="Path to the ingp home directory (repo dir)")

        parser.add_argument("--normalize", action="store_true",
                            help="Normalize cameras into (-1, 1)")
        parser.add_argument("--normalize-scale-factor", type=float, default=1.0,
                            help="Scale factor for normalization")

        parser.add_argument("--camera-model", type=str, default="OPENCV",
                            help="Camera model")
        parser.add_argument("--camera-heuristics", "-c", type=str, default=None,
                            help="Camera heuristics")
        parser.add_argument("--colmap-binary", type=str, default="colmap",
                            help="Path to the colmap binary")
        parser.add_argument("--colmap-dir", type=str, default=None,
                            help="Path to the colmap outputs")
        parser.add_argument("--no-loop-detection", action="store_true",
                            help="Disable loop detection in colmap")

        parser.add_argument("--model", "-m", type=str, default="colmap", choices=["colmap", "hloc", "hloc++"],
                            help="Reconstruction method")
        parser.add_argument("--matcher", type=str, default="exhaustive",
                            help="Feature matcher, omitted for video inputs")

        parser.add_argument("--hloc-max-keypoints", type=int, default=20000,
                            help="Maximum number of keypoints for HLoc")

        parser.add_argument("--object", action="store_true",
                            help="Object scene, will generate object masks and normalize the scene into (-1, 1)")
        parser.add_argument("--no-flow", action="store_true",
                            help="Do not use flow when segmenting images")
        parser.add_argument("--no-undistort", action="store_true",
                            help="Do not undistort images")

        parser.add_argument("--verbose", action="store_true",
                            help="Verbose mode (deprecated)")
        parser.add_argument("--log-level", "-l", type=str, default="INFO",
                            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                            help="Log level (DEBUG/INFO/WARNING/ERROR)")

        return parser.parse_args()

    args = get_args()

    ingp_home = os.path.expanduser(args.ingp_home)
    assert os.path.exists(os.path.join(ingp_home, "build")), "ingp is not built, please follow the installation guide."
    sys.path.append(os.path.join(ingp_home, "build"))

    if args.verbose:
        nuwa.set_log_level(nuwa.logging.DEBUG)
        nuwa.get_logger().warning("`--verbose` flag is deprecated, use `-l DEBUG` instead")
    else:
        nuwa.set_log_level(nuwa.logging.getLevelName(args.log_level))

    out_dir = args.out_dir
    if os.path.exists(out_dir):
        nuwa.get_logger().warning("Output directory exists, overwriting...")
    os.makedirs(out_dir, exist_ok=True)

    camera_model = args.camera_model
    heuristics = args.camera_heuristics
    colmap_binary = args.colmap_binary
    colmap_out_dir = args.colmap_dir
    colmap_loop_detection = not args.no_loop_detection
    gen_mask = args.object
    undistort = not args.no_undistort
    if gen_mask:
        assert nuwa.is_seg_available(), "Segmentation is not available, please install dependencies following README."
        assert camera_model == "OPENCV" and undistort  # fix this
    if args.finetune_pose:
        assert nuwa.is_ingp_available(), ("ingp is required for fine-tuning pose, "
                                          "please install dependencies following README.")
    copy_images_to = None

    if args.model == "colmap":
        model = "colmap"
        hloc_use_pixsfm = False
    elif args.model == "hloc":
        model = "hloc"
        hloc_use_pixsfm = False
    elif args.model == "hloc++":
        model = "hloc"
        hloc_use_pixsfm = True
    else:
        raise ValueError(f"Unknown model: {args.model}")

    hloc_max_keypoints = args.hloc_max_keypoints

    if args.polycam_path or args.scannerapp_path or args.dear_path or args.haoyang_path:
        image_out_dir = os.path.join(out_dir, "images") \
            if args.image_dir == "" else args.image_dir

        if args.polycam_path:
            db = from_polycam(
                args.polycam_path,
                image_out_dir,
                args.discard_border_rate,
                args.portrait
            )

        elif args.scannerapp_path:
            db = from_3dscannerapp(
                args.scannerapp_path,
                image_out_dir
            )
        elif args.dear_path:
            db = from_dear(
                args.dear_path,
                image_out_dir,
                args.portrait,
                args.dear_sample_stride
            )
        else:
            db = from_haoyang(
                args.haoyang_path
            )
            copy_images_to = os.path.join(out_dir, "images")

        if args.finetune_pose_colmap:
            db.finetune_pose_colmap(
                matcher="exhaustive",
                colmap_binary=colmap_binary,
                loop_detection=True
            )

    elif args.video_path:
        image_dir = args.image_dir
        if image_dir == "":
            image_dir = tempfile.mkdtemp()

        db = from_video(
            args.video_path,
            image_dir,
            fps=args.fps,
            method=model,
            single_camera=True,
            camera_model=camera_model,
            camera_heuristics=heuristics,
            camera_run_undistort=undistort,
            colmap_out_dir=colmap_out_dir,
            colmap_binary=colmap_binary,
            hloc_max_keypoints=hloc_max_keypoints,
            hloc_use_pixsfm=hloc_use_pixsfm
        )

        copy_images_to = os.path.join(out_dir, "images")

    else:
        assert args.image_dir != "", "Image directory or video path is required"
        matcher = args.matcher
        image_dir = args.image_dir

        if matcher == "exhaustive":
            nuwa.get_logger().warning("Exhaustive matcher is used, this may take a long time. "
                                      "Pass --matcher explicitly if this is unwanted.")

        db = from_image_folder(
            image_dir,
            method=model,
            matcher=matcher,
            single_camera=True,
            camera_model=camera_model,
            camera_heuristics=heuristics,
            camera_run_undistort=undistort,
            colmap_out_dir=colmap_out_dir,
            colmap_binary=colmap_binary,
            colmap_loop_detection=colmap_loop_detection,
            hloc_max_keypoints=hloc_max_keypoints,
            hloc_use_pixsfm=hloc_use_pixsfm
        )

    if args.finetune_pose:
        db.normalize_cameras(positive_z=True, scale_factor=0.8)
        db.finetune_pose(ingp_home=ingp_home)

    if gen_mask:
        try:
            _ = db.calculate_object_mask(
                os.path.join(out_dir, "masks"),
                os.path.join(out_dir, "images"),
                adjust_cameras=True,
                copy_org=True,
                use_flow=not args.no_flow
            )

            copy_images_to = None

            if args.finetune_pose:
                db.finetune_pose(ingp_home=ingp_home)

        except ValueError as e:
            nuwa.get_logger().warning(f"Mask generation failed: {e}")
            nuwa.get_logger().warning("Continuing without mask generation")
            db.normalize_cameras(positive_z=True, scale_factor=0.8)

    elif args.normalize:
        db.normalize_cameras(positive_z=True, scale_factor=args.normalize_scale_factor)
        if args.finetune_pose:
            db.finetune_pose(ingp_home=ingp_home)

    elif args.finetune_pose:
        nuwa.get_logger().warning("Pose fine-tuning with ingp requires object scene or normalization.")
        nuwa.get_logger().warning("Skipping pose fine-tuning...")

    db.dump(
        os.path.join(out_dir, "nuwa_db.json"),
        dump_reconstruction_to=os.path.join(out_dir, "sparse/0"),
        copy_images_to=copy_images_to
    )

    with open(os.path.join(out_dir, "nuwa_argv.txt"), "w") as f:
        f.write(" ".join(sys.argv))


def genpoints(args):
    nuwa_dir = args.input_dir
    out_dir = args.out_dir
    colmap_binary = args.colmap_binary
    loop_detection = not args.no_loop_detection
    matcher = args.matcher

    if out_dir is None:
        out_dir = nuwa_dir
        write_sparse_only = True
        nuwa.get_logger().warning("Output directory is not specified, overwriting sparse in the input directory...")
    else:
        write_sparse_only = False
        os.makedirs(out_dir, exist_ok=False)

    db = from_nuwadb(os.path.join(nuwa_dir, "nuwa_db.json"))
    db.generate_points_colmap(matcher=matcher, colmap_binary=colmap_binary, loop_detection=loop_detection)

    if write_sparse_only:
        db.dump_reconstruction(os.path.join(out_dir, "sparse/0"))
        # TODO: update db.colmap_path

        with open(os.path.join(out_dir, "nuwa-colmap_argv.txt"), "w") as f:
            f.write(" ".join(sys.argv))

    else:
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
        db.dump(
            os.path.join(out_dir, "nuwa_db.json"),
            copy_images_to=os.path.join(out_dir, "images"),
            copy_masks_to=os.path.join(out_dir, "masks"),
            dump_reconstruction_to=os.path.join(out_dir, "sparse/0")
        )

        with open(os.path.join(out_dir, "nuwa-colmap_argv.txt"), "w") as f:
            f.write(" ".join(sys.argv))


def gs2mesh(args):
    try:
        import open3d as o3d
    except ImportError:
        nuwa.get_logger().error('nmesh - Open3D is not installed, please run `pip install "open3d>=0.18.0"`.')
    import numpy as np
    from nuwa.utils.gs_utils import load_gs_simple

    gs = load_gs_simple(args.input_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gs['xyz'])
    pcd.colors = o3d.utility.Vector3dVector(gs['color'])
    nuwa.get_logger().info(f"nmesh - Loaded point cloud from {args.input_path} with {len(pcd.points)} points.")

    pcd.estimate_normals()
    pcd.orient_normals_towards_camera_location(np.array([0, 0, 1.5]))

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.poisson_depth)
    nuwa.get_logger().info(f"nmesh - Poisson generates a mesh with {len(mesh.triangles)} faces.")

    n_faces = args.target_faces
    if n_faces < 100:
        n_faces = len(mesh.triangles) // n_faces
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=n_faces)
    nuwa.get_logger().info(f"nmesh - Mesh has {len(mesh.triangles)} faces after simplification.")

    nuwa.get_logger().info(f"nmesh - Writing mesh to {args.out_path}.")
    o3d.io.write_triangle_mesh(
        args.out_path,
        mesh,
        write_ascii=False,
        compressed=True,
        write_vertex_normals=True,
        write_vertex_colors=True,
        write_triangle_uvs=True,
        print_progress=False
    )


def downsample(args):
    nuwa_dir = args.input_path
    out_dir = args.output_path
    resolution = args.resolution

    if os.path.exists(out_dir):
        nuwa.get_logger().warning("Output directory exists, overwriting...")

    os.makedirs(out_dir, exist_ok=True)
    new_image_dir = os.path.join(out_dir, "images")
    os.makedirs(new_image_dir, exist_ok=True)
    new_mask_dir = os.path.join(out_dir, "masks")
    if os.path.exists(os.path.join(nuwa_dir, "masks")):
        os.makedirs(new_mask_dir, exist_ok=True)

    db = from_nuwadb(os.path.join(nuwa_dir, "nuwa_db.json"))
    db.downsample_images(resolution, new_image_dir=new_image_dir, new_mask_dir=new_mask_dir)

    db.dump(
        os.path.join(out_dir, "nuwa_db.json"),
        copy_images_to=None,
        copy_masks_to=None,
        dump_reconstruction_to=os.path.join(out_dir, "sparse/0")
    )


def seg(args):
    from PIL import Image

    assert nuwa.is_seg_available(), "Segmentation is not available, please install dependencies following README."
    from nuwa.utils.seg_utils import segment_fg

    input_dir = args.input_dir
    output_dir = args.output_dir
    use_flow = not args.no_flow
    reduce_factor = args.flow_reduce_factor
    sam_ckpt_path = args.sam_ckpt_path
    shrink = 0.02

    image_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(("jpg", "png"))])

    images, masks = segment_fg(
        [Image.open(f) for f in image_paths],
        use_flow=use_flow,
        sam_ckpt_path=sam_ckpt_path,
        reduce_factor=reduce_factor,
        shrink=shrink
    )

    os.makedirs(output_dir, exist_ok=True)
    for (image_path, img, mask) in zip(image_paths, images, masks):
        image_path = image_path.replace(".jpg", ".png") if image_path.endswith("jpg") else image_path
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        img.save(os.path.join(output_dir, os.path.basename(image_path)))
        mask.save(os.path.join(output_dir, os.path.basename(image_path).replace(".png", "_mask.png")))


def tools():
    parser = argparse.ArgumentParser(description="nuwa-tools: nuwadb toolbox")
    parser.add_argument("--log-level", "-l", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Log level (DEBUG/INFO/WARNING/ERROR)")
    subparsers = parser.add_subparsers()

    gs2mesh_parser = subparsers.add_parser("gs2mesh", help="Generate mesh from input point cloud / 3dgs ply.")
    gs2mesh_parser.add_argument("--input-path", "-i", type=str, default="",
                                help="Path to the point cloud ply")
    gs2mesh_parser.add_argument("--out-path", "-o", type=str, default=None,
                                help="Path to the output mesh")
    gs2mesh_parser.add_argument("--poisson-depth", "-d", type=int, default=11, help="Poisson depth")
    gs2mesh_parser.add_argument("--target-faces", "-f", type=int, default=16,
                                help="Target number of faces in the output mesh."
                                     "If < 100, it is interpreted as the reduce factor of the poisson mesh, "
                                     "else as the target number of faces.")
    gs2mesh_parser.set_defaults(func=gs2mesh)

    genpoints_parser = subparsers.add_parser("genpoints", description="Generate 3D points with colmap for nuwadb")
    genpoints_parser.add_argument("--input-dir", "-i", type=str, default="",
                                  help="Path to the nuwadb")
    genpoints_parser.add_argument("--out-dir", "-o", type=str, default=None,
                                  help="Path to the output dir")
    genpoints_parser.add_argument("--colmap-binary", type=str, default="colmap",
                                  help="Path to the colmap binary")
    genpoints_parser.add_argument("--matcher", "-m", type=str, default="exhaustive",
                                  help="Feature matcher, omitted for video inputs")
    genpoints_parser.add_argument("--no-loop-detection", action="store_true",
                                  help="Disable loop detection in colmap")
    genpoints_parser.set_defaults(func=genpoints)

    downsample_parser = subparsers.add_parser("downsample", help="Downsample the images")
    downsample_parser.add_argument("--input-path", "-i", type=str, required=True,
                                   help="Path to the input nuwadb")
    downsample_parser.add_argument("--output-path", "-o", type=str, required=True,
                                   help="Path to the output nuwadb")
    downsample_parser.add_argument("--resolution", "-r", type=int, default=2,
                                   help="If value <= 64, it is interpreted as the resolution factor, "
                                        "else as the target resolution (longer side)")
    downsample_parser.set_defaults(func=downsample)

    seg_parser = subparsers.add_parser("seg", help="Segment the foreground from the images")
    seg_parser.add_argument("--input-dir", "-i", type=str, required=True,
                            help="Path to the input image folder")
    seg_parser.add_argument("--output-dir", "-o", type=str, required=True,
                            help="Path to the output folder")
    seg_parser.add_argument("--no-flow", "-n", action="store_true",
                            help="Do not use flow to track the fg. Use this for non-sequential captures.")
    seg_parser.add_argument("--flow-reduce-factor", "-r", type=int, default=2,
                            help="Downsample factor for computing flow.")
    seg_parser.add_argument("--sam-ckpt-path", type=str, default=None,
                            help="Path to the custom SAM checkpoint.")
    seg_parser.set_defaults(func=seg)

    args = parser.parse_args()
    nuwa.set_log_level(nuwa.logging.getLevelName(args.log_level))

    args.func(args)


# if __name__ == "__main__":
#     main()
