from nuwa import from_image_folder, from_video


def main():
    def get_args():
        import argparse
        parser = argparse.ArgumentParser(description="Nuwa: 3D Reconstruction Pipeline")
        parser.add_argument("--video-path", "-v", type=str, default="",
                            help="Path to the video file")
        parser.add_argument("--image-dir", "-i", type=str, default="",
                            help="Path to the images / to expand the frames")
        parser.add_argument("--out-dir", "-o", type=str, default="./nuwa_results",
                            help="Output directory")

        parser.add_argument("--fps", type=int, default=3, help="FPS for video inputs")

        parser.add_argument("--camera-model", type=str, default="OPENCV",
                            help="Camera model")
        parser.add_argument("--camera-heuristics", "-c", type=str, default=None,
                            help="Camera heuristics")
        parser.add_argument("--colmap-binary", type=str, default="colmap",
                            help="Path to the COLMAP binary")
        parser.add_argument("--colmap-dir", type=str, default="",
                            help="Path to the COLMAP outputs")
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
        parser.add_argument("--no-undistort", action="store_true",
                            help="Do not undistort images")

        parser.add_argument("--verbose", action="store_true",
                            help="Verbose mode")

        return parser.parse_args()

    import os
    import sys
    args = get_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=False)

    debug = args.verbose
    camera_model = args.camera_model
    heuristics = args.camera_heuristics
    colmap_binary = args.colmap_binary
    colmap_out_dir = args.colmap_dir
    colmap_loop_detection = not args.no_loop_detection
    gen_mask = args.object
    undistort = not args.no_undistort
    if gen_mask:
        assert camera_model == "OPENCV" and undistort

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

    if args.video_path:
        image_dir = args.image_dir
        if image_dir == "":
            image_dir = os.path.join(out_dir, "_org_images")

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
            hloc_use_pixsfm=hloc_use_pixsfm,
            verbose=debug
        )

    else:
        assert args.image_dir != "", "Image directory or video path is required"
        matcher = args.matcher
        image_dir = args.image_dir

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
            hloc_use_pixsfm=hloc_use_pixsfm,
            verbose=debug
        )

    if gen_mask:
        try:
            _ = db.calculate_object_mask(
                os.path.join(out_dir, "masks"),
                os.path.join(out_dir, "images"),
                adjust_cameras=True,
                copy_org=True
            )
        except ValueError as e:
            print(f"ERROR Mask generation failed: {e}")
            print("Continuing without mask generation")

    db.dump(os.path.join(out_dir, "nuwa_db.json"))

    # save argv
    with open(os.path.join(out_dir, "argv.txt"), "w") as f:
        f.write(" ".join(sys.argv))


if __name__ == "__main__":
    main()
