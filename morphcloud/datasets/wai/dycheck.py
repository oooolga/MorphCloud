# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
DyCheck (iPhone) Dataset using WAI format data.
Dataset for dynamic scenes captured with iPhone/Record3D.
"""

import json
import os

import cv2
import numpy as np

from morphcloud.datasets.base.base_dataset import BaseDataset


class DyCheckWAI(BaseDataset):
    """
    DyCheck dataset containing dynamic scenes captured with iPhone.
    This dataset uses the Nerfies/HyperNeRF data format with per-frame cameras.
    """

    def __init__(
        self,
        *args,
        ROOT,
        dataset_metadata_dir=None,
        split="train",
        overfit_num_sets=None,
        sample_specific_scene: bool = False,
        specific_scene_name: str = None,
        factor: int = 2,
        **kwargs,
    ):
        """
        Initialize the dataset attributes.
        Args:
            ROOT: Root directory of the dataset (e.g., /path/to/DyCheck-iphone/).
            dataset_metadata_dir: Path to the dataset metadata directory (optional).
            split: Dataset split ('train', 'val').
            overfit_num_sets: If None, use all sets. Else, truncate to this number of sets.
            sample_specific_scene: Whether to sample a specific scene.
            specific_scene_name: Name of the specific scene to sample.
            factor: Downsampling factor for images (default: 2 for 2x downsampled).
        """
        # Initialize the dataset attributes
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.dataset_metadata_dir = dataset_metadata_dir
        self.split = split
        self.overfit_num_sets = overfit_num_sets
        self.sample_specific_scene = sample_specific_scene
        self.specific_scene_name = specific_scene_name
        self.factor = factor
        self._load_data()

        # Define the dataset type flags
        self.is_metric_scale = True
        self.is_synthetic = False  # Real captured data

    def _load_data(self):
        """Load the dataset scenes and metadata"""
        # If metadata directory is provided and contains scene lists, use that
        if self.dataset_metadata_dir is not None:
            split_metadata_path = os.path.join(
                self.dataset_metadata_dir,
                self.split,
                f"dycheck_scene_list_{self.split}.npy",
            )
            if os.path.exists(split_metadata_path):
                split_scene_list = np.load(split_metadata_path, allow_pickle=True)
                if not self.sample_specific_scene:
                    self.scenes = list(split_scene_list)
                else:
                    self.scenes = [self.specific_scene_name]
                self.num_of_scenes = len(self.scenes)
                return

        # Otherwise, discover scenes from the ROOT directory
        if not self.sample_specific_scene:
            # List all directories in ROOT that contain dataset.json
            all_dirs = [
                d for d in os.listdir(self.ROOT)
                if os.path.isdir(os.path.join(self.ROOT, d))
                and os.path.exists(os.path.join(self.ROOT, d, "dataset.json"))
            ]
            self.scenes = sorted(all_dirs)
        else:
            self.scenes = [self.specific_scene_name]

        # Limit scenes if overfitting
        if self.overfit_num_sets is not None:
            self.scenes = self.scenes[:self.overfit_num_sets]

        self.num_of_scenes = len(self.scenes)

    def _load_scene_info(self, scene_root):
        """Load scene.json containing center, scale, near, far."""
        scene_path = os.path.join(scene_root, "scene.json")
        if os.path.exists(scene_path):
            with open(scene_path, 'r') as f:
                scene_dict = json.load(f)
            center = np.array(scene_dict["center"], dtype=np.float32)
            scale = scene_dict["scale"]
            near = scene_dict["near"]
            far = scene_dict["far"]
            return center, scale, near, far
        return np.zeros(3, dtype=np.float32), 1.0, 0.1, 10.0

    def _load_metadata_info(self, scene_root):
        """Load dataset.json and metadata.json for frame info."""
        dataset_path = os.path.join(scene_root, "dataset.json")
        metadata_path = os.path.join(scene_root, "metadata.json")

        with open(dataset_path, 'r') as f:
            dataset_dict = json.load(f)
        frame_names = np.array(dataset_dict["ids"])

        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)

        time_ids = np.array(
            [metadata_dict[k]["warp_id"] for k in frame_names], dtype=np.uint32
        )
        camera_ids = np.array(
            [metadata_dict[k]["camera_id"] for k in frame_names], dtype=np.uint32
        )

        return frame_names, time_ids, camera_ids

    def _load_split_info(self, scene_root):
        """Load split information from splits directory."""
        splits_dir = os.path.join(scene_root, "splits")
        split_path = os.path.join(splits_dir, f"{self.split}.json")

        print(f"[DEBUG] Looking for split file: {split_path}")
        print(f"[DEBUG] Split file exists: {os.path.exists(split_path)}")

        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                split_dict = json.load(f)
            
            print(f"[DEBUG] Split dict keys: {split_dict.keys()}")
            
            # Check if the expected keys exist
            if "frame_names" in split_dict:
                frame_names = split_dict["frame_names"]
                print(f"[DEBUG] frame_names type: {type(frame_names)}")
                print(f"[DEBUG] First 5 frame_names: {frame_names[:5] if len(frame_names) > 0 else 'empty'}")
                print(f"[DEBUG] Total frame_names: {len(frame_names)}")
                return (
                    np.array(frame_names),
                    np.array(split_dict.get("time_ids", []), dtype=np.uint32),
                    np.array(split_dict.get("camera_ids", []), dtype=np.uint32),
                )
        
        print(f"[DEBUG] Falling back to metadata-based split loading")
        # Fallback: filter frames from dataset.json based on camera_id
        # In DyCheck iPhone dataset: train uses camera_id=0, val uses camera_id!=0
        return self._load_split_from_metadata(scene_root)
    
    def _load_split_from_metadata(self, scene_root):
        """
        Fallback method to create split from metadata when split files don't exist
        or have different structure. Uses camera_id to determine split.
        """
        dataset_path = os.path.join(scene_root, "dataset.json")
        metadata_path = os.path.join(scene_root, "metadata.json")
        
        if not os.path.exists(dataset_path) or not os.path.exists(metadata_path):
            return None, None, None
        
        with open(dataset_path, 'r') as f:
            dataset_dict = json.load(f)
        
        # Check if dataset.json has pre-split ids
        if self.split == "train" and "train_ids" in dataset_dict:
            frame_names = np.array(dataset_dict["train_ids"])
        elif self.split == "val" and "val_ids" in dataset_dict:
            frame_names = np.array(dataset_dict["val_ids"])
        else:
            # Fall back to filtering by camera_id from metadata
            all_frame_names = np.array(dataset_dict["ids"])
            
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            # Filter frames based on camera_id
            # train: camera_id == 0, val: camera_id != 0
            filtered_frames = []
            for frame_name in all_frame_names:
                if frame_name in metadata_dict:
                    camera_id = metadata_dict[frame_name].get("camera_id", 0)
                    if self.split == "train" and camera_id == 0:
                        filtered_frames.append(frame_name)
                    elif self.split == "val" and camera_id != 0:
                        filtered_frames.append(frame_name)
            
            frame_names = np.array(filtered_frames)
        
        if len(frame_names) == 0:
            return None, None, None
        
        # Load time_ids and camera_ids for filtered frames
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        time_ids = np.array(
            [metadata_dict[k]["warp_id"] for k in frame_names if k in metadata_dict],
            dtype=np.uint32
        )
        camera_ids = np.array(
            [metadata_dict[k]["camera_id"] for k in frame_names if k in metadata_dict],
            dtype=np.uint32
        )
        
        return frame_names, time_ids, camera_ids

    def _load_camera(self, scene_root, frame_name, center, scale):
        """
        Load camera parameters from JSON file.

        Args:
            scene_root: Path to scene directory
            frame_name: Name of the frame (e.g., "0_00000")
            center: Scene center for translation
            scale: Scene scale

        Returns:
            intrinsics: 3x3 camera intrinsics matrix
            c2w_pose: 4x4 camera-to-world transformation matrix
        """
        camera_path = os.path.join(scene_root, "camera", f"{frame_name}.json")

        with open(camera_path, 'r') as f:
            camera_dict = json.load(f)

        # Extract camera parameters
        orientation = np.array(camera_dict["orientation"], dtype=np.float32)
        position = np.array(camera_dict["position"], dtype=np.float32)
        focal_length = float(camera_dict["focal_length"])
        principal_point = np.array(camera_dict["principal_point"], dtype=np.float32)
        image_size = np.array(camera_dict["image_size"], dtype=np.uint32)
        pixel_aspect_ratio = float(camera_dict.get("pixel_aspect_ratio", 1.0))
        skew = float(camera_dict.get("skew", 0.0))

        # Apply factor scaling to intrinsics and image size
        scaled_focal_length = focal_length / self.factor
        scaled_principal_point = principal_point / self.factor
        scaled_image_size = image_size // self.factor

        # Apply scene center and scale transformations
        # Translate position relative to center and apply scale
        position = (position - center) * scale

        # Build intrinsics matrix
        fx = scaled_focal_length
        fy = scaled_focal_length * pixel_aspect_ratio
        cx, cy = scaled_principal_point

        intrinsics = np.array([
            [fx, skew, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Build camera-to-world matrix from orientation and position
        # Orientation is w2c rotation (maps world to camera)
        # Position is camera position in world coordinates
        R_w2c = orientation  # (3, 3)
        R_c2w = R_w2c.T  # Transpose to get c2w rotation

        c2w_pose = np.eye(4, dtype=np.float32)
        c2w_pose[:3, :3] = R_c2w
        c2w_pose[:3, 3] = position

        return intrinsics, c2w_pose, scaled_image_size

    def _load_image(self, scene_root, frame_name):
        """Load RGB image for a frame."""
        # Try different possible paths
        rgb_paths = [
            os.path.join(scene_root, "rgb", f"{self.factor}x", f"{frame_name}.png"),
            os.path.join(scene_root, "rgb", f"{self.factor}x", f"{frame_name}.jpg"),
        ]

        for rgb_path in rgb_paths:
            if os.path.exists(rgb_path):
                image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
                if image is not None:
                    # Convert BGR to RGB
                    if len(image.shape) == 3:
                        if image.shape[2] == 4:
                            # BGRA to RGB
                            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                        else:
                            # BGR to RGB
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    return image

        raise FileNotFoundError(f"Image not found for frame {frame_name} in {scene_root}")

    def _load_depth(self, scene_root, frame_name):
        """Load depth map for a frame if available."""
        depth_paths = [
            os.path.join(scene_root, "depth", f"{self.factor}x", f"{frame_name}.npy"),
            os.path.join(scene_root, "depth", "1x", f"{frame_name}.npy"),
        ]

        for depth_path in depth_paths:
            if os.path.exists(depth_path):
                depth = np.load(depth_path).astype(np.float32)
                # Resize if loaded from 1x
                if "1x" in depth_path and self.factor > 1:
                    new_h, new_w = depth.shape[0] // self.factor, depth.shape[1] // self.factor
                    depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                return depth

        return None

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        """
        Get views from the DyCheck dataset.

        Args:
            sampled_idx: Index of the sampled scene
            num_views_to_sample: Number of views to sample
            resolution: Target resolution for the images

        Returns:
            views: List of view dictionaries
        """
        # Get the scene name
        scene_index = sampled_idx
        scene_name = self.scenes[scene_index]

        # Path to the scene directory
        scene_root = os.path.join(self.ROOT, scene_name)

        # Load scene info
        center, scale, near, far = self._load_scene_info(scene_root)

        # Load split-specific frames (handles both split files and metadata-based filtering)
        frame_names, time_ids, camera_ids = self._load_split_info(scene_root)

        if frame_names is None or len(frame_names) == 0:
            raise ValueError(f"No frames found in {scene_name} for split {self.split}")

        # Sample frame indices
        num_frames_in_scene = len(frame_names)

        if num_views_to_sample >= num_frames_in_scene:
            view_indices = list(range(num_frames_in_scene))
        else:
            view_indices = np.random.choice(
                num_frames_in_scene,
                size=num_views_to_sample,
                replace=False
            )

        # Get the views
        views = []
        for view_index in view_indices:
            frame_name = frame_names[view_index]
            time_id = time_ids[view_index] if time_ids is not None else 0
            camera_id = camera_ids[view_index] if camera_ids is not None else 0

            # Load camera parameters
            intrinsics, c2w_pose, image_size = self._load_camera(
                scene_root, frame_name, center, scale
            )

            # Load image
            image = self._load_image(scene_root, frame_name)

            # Load depth if available
            depthmap = self._load_depth(scene_root, frame_name)

            # Ensure depthmap has valid values if it exists
            if depthmap is not None:
                # Apply scale to depth
                depthmap = depthmap * scale
                depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)

            # Resize the data to match the desired resolution
            if depthmap is not None:
                image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    image=image,
                    resolution=resolution,
                    depthmap=depthmap,
                    intrinsics=intrinsics,
                    additional_quantities=None,
                )
            else:
                image, _, intrinsics = self._crop_resize_if_necessary(
                    image=image,
                    resolution=resolution,
                    depthmap=None,
                    intrinsics=intrinsics,
                    additional_quantities=None,
                )

            # Create view dictionary
            view_dict = dict(
                img=image,
                camera_pose=c2w_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset="DyCheck",
                label=scene_name,
                instance=frame_name,
                time_id=int(time_id),
                camera_id=int(camera_id),
            )

            # Add depth if available
            if depthmap is not None:
                view_dict["depthmap"] = depthmap

            views.append(view_dict)

        return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd",
        "--root_dir",
        default="/network/scratch/x/xuolga/Datasets/DyCheck-iphone",
        type=str,
    )
    parser.add_argument(
        "-dmd",
        "--dataset_metadata_dir",
        default=None,
        type=str,
        help="Optional: Path to precomputed dataset metadata directory"
    )
    parser.add_argument(
        "-nv",
        "--num_of_views",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--factor",
        default=2,
        type=int,
        help="Downsampling factor for images (default: 2)"
    )
    parser.add_argument("--viz", action="store_true")

    return parser


if __name__ == "__main__":
    import rerun as rr
    from tqdm import tqdm

    from morphcloud.datasets.base.base_dataset import view_name
    from morphcloud.utils.image import rgb
    from morphcloud.utils.viz import script_add_rerun_args

    parser = get_parser()
    script_add_rerun_args(
        parser
    )  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    dataset = DyCheckWAI(
        num_views=args.num_of_views,
        split="train",
        ROOT=args.root_dir,
        dataset_metadata_dir=args.dataset_metadata_dir,
        resolution=(518, 294),
        transform="imgnorm",
        data_norm_type="dinov2",
        factor=args.factor,
    )
    print(f"Dataset loaded with {len(dataset)} scenes")
    print(dataset.get_stats())

    if args.viz:
        rr.script_setup(args, "DyCheck_Dataloader")
        rr.set_time("stable_time", sequence=0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # Sample a few scenes
    sampled_indices = np.random.choice(
        len(dataset),
        size=min(5, len(dataset)),
        replace=False
    )

    for num, idx in enumerate(tqdm(sampled_indices)):
        try:
            views = dataset[idx]
            assert len(views) == args.num_of_views
            sample_name = f"{idx}"
            for view_idx in range(args.num_of_views):
                sample_name += f" {view_name(views[view_idx])}"
            print(sample_name)

            for view_idx in range(args.num_of_views):
                image = rgb(
                    views[view_idx]["img"],
                    norm_type=views[view_idx].get("data_norm_type", "dinov2")
                )
                pose = views[view_idx]["camera_pose"]
                intrinsics = views[view_idx]["camera_intrinsics"]
                time_id = views[view_idx].get("time_id", 0)
                camera_id = views[view_idx].get("camera_id", 0)

                print(f"  View {view_idx}: time_id={time_id}, camera_id={camera_id}")

                if args.viz:
                    rr.set_time("stable_time", sequence=num)
                    base_name = f"world/view_{view_idx}"

                    height, width = image.shape[0], image.shape[1]
                    rr.log(
                        base_name,
                        rr.Transform3D(
                            translation=pose[:3, 3],
                            mat3x3=pose[:3, :3],
                        ),
                    )
                    rr.log(
                        f"{base_name}/pinhole",
                        rr.Pinhole(
                            image_from_camera=intrinsics,
                            height=height,
                            width=width,
                            camera_xyz=rr.ViewCoordinates.RDF,
                        ),
                    )
                    rr.log(
                        f"{base_name}/pinhole/rgb",
                        rr.Image(image),
                    )

                    if "depthmap" in views[view_idx]:
                        rr.log(
                            f"{base_name}/pinhole/depth",
                            rr.DepthImage(views[view_idx]["depthmap"]),
                        )
        except Exception as e:
            print(f"Error loading scene {idx}: {e}")
            import traceback
            traceback.print_exc()
