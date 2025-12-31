# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
DynaVol Dataset using WAI format data.
Dataset for dynamic neural radiance fields with object motion.
"""

import json
import os

import imageio
import numpy as np

from morphcloud.datasets.base.base_dataset import BaseDataset


class DynaVolWAI(BaseDataset):
    """
    DynaVol dataset containing dynamic scenes with falling/moving objects.
    This dataset uses the NeRF-style transform JSON format with time information.
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
        use_4view_variant: bool = False,
        **kwargs,
    ):
        """
        Initialize the dataset attributes.
        Args:
            ROOT: Root directory of the dataset (e.g., /path/to/DynaVol/).
            dataset_metadata_dir: Path to the dataset metadata directory (optional for DynaVol).
            split: Dataset split ('train', 'val', 'test').
            overfit_num_sets: If None, use all sets. Else, the dataset will be truncated to this number of sets.
            sample_specific_scene: Whether to sample a specific scene from the dataset.
            specific_scene_name: Name of the specific scene to sample.
            use_4view_variant: Whether to use the dynamic_4views variant (default: False).
        """
        # Initialize the dataset attributes
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.dataset_metadata_dir = dataset_metadata_dir
        self.split = split
        self.overfit_num_sets = overfit_num_sets
        self.sample_specific_scene = sample_specific_scene
        self.specific_scene_name = specific_scene_name
        self.use_4view_variant = use_4view_variant
        self._load_data()

        # Define the dataset type flags
        self.is_metric_scale = True
        self.is_synthetic = True

    def _load_data(self):
        """Load the dataset scenes and metadata"""
        # Define scene subdirectory based on variant
        self.scene_subdir = "dynamic_4views" if self.use_4view_variant else "dynamic"
        
        # If metadata directory is provided and contains scene lists, use that
        if self.dataset_metadata_dir is not None:
            split_metadata_path = os.path.join(
                self.dataset_metadata_dir,
                self.split,
                f"dynavol_scene_list_{self.split}.npy",
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
            # List all directories in ROOT that contain the dynamic subdirectory
            all_dirs = [
                d for d in os.listdir(self.ROOT)
                if os.path.isdir(os.path.join(self.ROOT, d))
                and os.path.exists(os.path.join(self.ROOT, d, self.scene_subdir))
            ]
            self.scenes = sorted(all_dirs)
        else:
            self.scenes = [self.specific_scene_name]
        
        # Limit scenes if overfitting
        if self.overfit_num_sets is not None:
            self.scenes = self.scenes[:self.overfit_num_sets]
        
        self.num_of_scenes = len(self.scenes)

    def _load_transforms_and_frames(self, scene_root):
        """
        Load the transform JSON file for the current split and return frames metadata.
        
        Args:
            scene_root: Path to the scene's dynamic directory
            
        Returns:
            meta: Dictionary containing metadata from transforms JSON
            frames: List of frame dictionaries
        """
        transform_file = os.path.join(
            scene_root, f"transforms_{self.split}.json"
        )
        
        if not os.path.exists(transform_file):
            raise FileNotFoundError(
                f"Transform file not found: {transform_file}"
            )
        
        with open(transform_file, 'r') as f:
            meta = json.load(f)
        
        frames = meta.get('frames', [])
        return meta, frames

    def _compute_intrinsics_from_meta(self, meta, width, height):
        """
        Compute camera intrinsics from metadata.
        
        Args:
            meta: Metadata dictionary containing camera parameters
            width: Image width
            height: Image height
            
        Returns:
            intrinsics: 3x3 camera intrinsics matrix
        """
        camera_angle_x = float(meta.get('camera_angle_x', 0.8575))  # default value
        focal = 0.5 * width / np.tan(0.5 * camera_angle_x)
        
        # Create intrinsics matrix
        intrinsics = np.array([
            [focal, 0, width / 2.0],
            [0, focal, height / 2.0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return intrinsics

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        """
        Get views from the DynaVol dataset.
        
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
        
        # Path to the scene's dynamic directory
        scene_root = os.path.join(self.ROOT, scene_name, self.scene_subdir)
        
        # Load transform metadata and frames
        meta, frames = self._load_transforms_and_frames(scene_root)
        
        if len(frames) == 0:
            raise ValueError(f"No frames found in {scene_name}")
        
        # Sample frame indices
        num_frames_in_scene = len(frames)
        
        # For simplicity, we sample frames uniformly or randomly
        # You can modify this to consider covisibility if needed
        if num_views_to_sample >= num_frames_in_scene:
            # If requesting more views than available, use all frames
            view_indices = list(range(num_frames_in_scene))
        else:
            # Randomly sample view indices
            view_indices = np.random.choice(
                num_frames_in_scene, 
                size=num_views_to_sample, 
                replace=False
            )
        
        # Get the views corresponding to the selected view indices
        views = []
        for view_index in view_indices:
            frame = frames[view_index]
            
            # Get the image file path
            file_path = frame['file_path']
            # Handle both relative paths like "train/000" and full paths
            if not file_path.endswith('.png'):
                file_path = file_path + '.png'
            
            # Construct full path to image
            if not file_path.startswith(self.split):
                # If the path doesn't include split dir, add it
                image_path = os.path.join(scene_root, self.split, os.path.basename(file_path))
            else:
                image_path = os.path.join(scene_root, file_path)
            
            if not os.path.exists(image_path):
                # Try alternative path structure
                image_path = os.path.join(scene_root, file_path)
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load the image
            image = imageio.imread(image_path)
            
            # Handle RGBA images - convert to RGB
            if image.shape[-1] == 4:
                # Blend with white background
                alpha = image[..., 3:4] / 255.0
                image = image[..., :3] * alpha + 255 * (1 - alpha)
                image = image.astype(np.uint8)
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Get the camera pose (transform_matrix is c2w)
            transform_matrix = np.array(frame['transform_matrix'], dtype=np.float32)
            
            # Get time information if available
            time = frame.get('time', 0.0)
            
            # Compute camera intrinsics
            intrinsics = self._compute_intrinsics_from_meta(meta, width, height)
            
            # DynaVol doesn't provide depth maps, so we don't include depthmap
            # If you have depth, you can add it here
            
            # Resize the data to match the desired resolution
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
                camera_pose=transform_matrix,  # cam2world
                camera_intrinsics=intrinsics,
                dataset="DynaVol",
                label=scene_name,
                instance=os.path.join(self.scene_subdir, self.split, os.path.basename(file_path)),
                time=time,  # Include time information for dynamic scenes
            )
            
            views.append(view_dict)
        
        return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd", 
        "--root_dir", 
        default="/network/scratch/x/xuolga/Datasets/DynaVol", 
        type=str
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
        "--use_4view_variant",
        action="store_true",
        help="Use dynamic_4views variant instead of dynamic"
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

    dataset = DynaVolWAI(
        num_views=args.num_of_views,
        split="train",
        ROOT=args.root_dir,
        dataset_metadata_dir=args.dataset_metadata_dir,
        resolution=(518, 294),
        transform="imgnorm",
        data_norm_type="dinov2",
        use_4view_variant=args.use_4view_variant,
    )
    print(f"Dataset loaded with {len(dataset)} scenes")
    print(dataset.get_stats())

    if args.viz:
        rr.script_setup(args, "DynaVol_Dataloader")
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
                time = views[view_idx].get("time", 0.0)
                
                print(f"  View {view_idx}: time={time:.3f}, pose shape={pose.shape}")
                
                if args.viz:
                    rr.set_time("stable_time", sequence=num)
                    base_name = f"world/view_{view_idx}"
                    
                    # Log camera info and loaded data
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
        except Exception as e:
            print(f"Error loading scene {idx}: {e}")
            import traceback
            traceback.print_exc()
