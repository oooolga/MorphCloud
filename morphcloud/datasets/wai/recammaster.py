# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
ReCamMaster Dataset using WAI format data.
"""

import json
import os

import cv2
import imageio
import numpy as np
import pandas as pd

from morphcloud.datasets.base.base_dataset import BaseDataset


class ReCamMasterWAI(BaseDataset):
    """
    ReCamMaster dataset containing video sequences with camera trajectories for video generation.
    """

    def __init__(
        self,
        *args,
        ROOT,
        dataset_metadata_dir=None,
        split="test",
        overfit_num_sets=None,
        sample_specific_scene: bool = False,
        specific_scene_name: str = None,
        **kwargs,
    ):
        """
        Initialize the dataset attributes.
        Args:
            ROOT: Root directory of the dataset (e.g., ./example_test_data).
            dataset_metadata_dir: Path to the dataset metadata directory (optional for ReCamMaster).
            split: Dataset split (train, val, test).
            overfit_num_sets: If None, use all sets. Else, the dataset will be truncated to this number of sets.
            sample_specific_scene: Whether to sample a specific scene from the dataset.
            specific_scene_name: Name of the specific scene to sample.
        """
        # Initialize the dataset attributes
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.dataset_metadata_dir = dataset_metadata_dir
        self.split = split
        self.overfit_num_sets = overfit_num_sets
        self.sample_specific_scene = sample_specific_scene
        self.specific_scene_name = specific_scene_name
        self._load_data()

        # Define the dataset type flags
        self.is_metric_scale = False  # Camera trajectories are normalized
        self.is_synthetic = False  # Real video data

    def _load_data(self):
        """Load the dataset metadata from metadata.csv"""
        metadata_path = os.path.join(self.ROOT, "metadata.csv")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
        # Load metadata CSV
        metadata = pd.read_csv(metadata_path)
        
        # Get video file names (scenes)
        if not self.sample_specific_scene:
            self.scenes = metadata["file_name"].tolist()
        else:
            self.scenes = [self.specific_scene_name]
        
        # Load text descriptions
        self.scene_texts = dict(zip(metadata["file_name"], metadata["text"]))
        
        # Limit scenes if overfitting
        if self.overfit_num_sets is not None:
            self.scenes = self.scenes[: self.overfit_num_sets]
        
        self.num_of_scenes = len(self.scenes)
        
        # Load camera extrinsics
        camera_path = os.path.join(self.ROOT, "cameras", "camera_extrinsics.json")
        if os.path.exists(camera_path):
            with open(camera_path, 'r') as f:
                self.camera_extrinsics = json.load(f)
        else:
            print(f"Warning: Camera extrinsics not found at {camera_path}")
            self.camera_extrinsics = None

    def _parse_camera_matrix(self, matrix_str):
        """Parse camera matrix from string format to numpy array"""
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    def _load_video_frames(self, video_path, num_frames_to_sample):
        """Load frames from video file"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        reader = imageio.get_reader(video_path)
        total_frames = reader.count_frames()
        
        # Sample frame indices uniformly across the video
        if total_frames >= num_frames_to_sample:
            frame_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
        else:
            # If video has fewer frames than requested, repeat frames
            frame_indices = np.arange(total_frames)
            frame_indices = np.resize(frame_indices, num_frames_to_sample)
        
        frames = []
        for idx in frame_indices:
            frame = reader.get_data(idx)
            frames.append(frame)
        
        reader.close()
        return frames

    def _get_camera_pose(self, frame_idx, cam_id=1):
        """Get camera pose for a specific frame and camera"""
        if self.camera_extrinsics is None:
            # Return identity pose if no camera data available
            return np.eye(4, dtype=np.float32)
        
        frame_key = f"frame{frame_idx}"
        cam_key = f"cam{int(cam_id):02d}"
        
        if frame_key not in self.camera_extrinsics:
            return np.eye(4, dtype=np.float32)
        
        if cam_key not in self.camera_extrinsics[frame_key]:
            return np.eye(4, dtype=np.float32)
        
        # Parse the camera matrix from string
        cam_matrix_str = self.camera_extrinsics[frame_key][cam_key]
        c2w = self._parse_camera_matrix(cam_matrix_str)
        c2w = c2w.transpose()
        
        # Apply coordinate system transformations (as in ReCamMaster)
        c2w = c2w[:, [1, 2, 0, 3]]
        c2w[:3, 1] *= -1.0
        c2w[:3, 3] /= 100.0
        
        return c2w.astype(np.float32)

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        """Get views for the sampled scene"""
        # Get the scene name
        scene_index = sampled_idx
        scene_name = self.scenes[scene_index]
        
        # Get video path
        video_path = os.path.join(self.ROOT, "videos", scene_name)
        
        # Load video frames
        frames = self._load_video_frames(video_path, num_views_to_sample)
        
        # Get text description
        text_description = self.scene_texts.get(scene_name, "")
        
        # Sample frame indices for camera poses
        total_frames = len(frames)
        frame_indices = np.linspace(0, total_frames - 1, num_views_to_sample, dtype=int)
        
        # Build views
        views = []
        for view_idx, (frame, frame_num) in enumerate(zip(frames, frame_indices)):
            # Convert frame to RGB if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
            image = frame.astype(np.uint8)
            
            # Get camera pose
            c2w_pose = self._get_camera_pose(frame_num)
            
            # Create basic intrinsics (assuming standard camera parameters)
            # You may need to adjust these based on your specific camera setup
            height, width = image.shape[:2]
            focal_length = max(height, width)  # Rough estimate
            cx, cy = width / 2.0, height / 2.0
            
            intrinsics = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Resize if necessary
            image, _, intrinsics = self._crop_resize_if_necessary(
                image=image,
                resolution=resolution,
                depthmap=None,
                intrinsics=intrinsics,
                additional_quantities=None,
            )
            
            # Append view
            views.append(
                dict(
                    img=image,
                    depthmap=None,  # ReCamMaster doesn't provide depth
                    camera_pose=c2w_pose,
                    camera_intrinsics=intrinsics,
                    dataset="ReCamMaster",
                    label=scene_name,
                    instance=f"{scene_name}_frame_{view_idx:04d}",
                    text_description=text_description,
                )
            )
        
        return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd", 
        "--root_dir", 
        default="/Users/joelmoniz/code/ReCamMaster/example_test_data", 
        type=str
    )
    parser.add_argument(
        "-nv",
        "--num_of_views",
        default=8,
        type=int,
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

    dataset = ReCamMasterWAI(
        num_views=args.num_of_views,
        covisibility_thres=None,  # Not used for ReCamMaster
        ROOT=args.root_dir,
        dataset_metadata_dir=None,
        resolution=(480, 832),
        seed=777,
        transform="imgnorm",
        data_norm_type="dinov2",
    )
    print(dataset.get_stats())

    if args.viz:
        rr.script_setup(args, "ReCamMaster_Dataloader")
        rr.set_time("stable_time", sequence=0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

        for sample_idx in tqdm(range(len(dataset))):
            sample = dataset[sample_idx]

            for view_idx, view in enumerate(sample["views"]):
                view_str = view_name(view_idx)
                rr.set_time("stable_time", sequence=view_idx)

                # Log image
                rr.log(
                    f"world/camera_{view_str}/rgb",
                    rr.Image(rgb(view["img"])),
                )

                # Log camera
                rr.log(
                    f"world/camera_{view_str}",
                    rr.Transform3D(
                        translation=view["camera_pose"][:3, 3],
                        mat3x3=view["camera_pose"][:3, :3],
                        from_parent=False,
                    ),
                )
                rr.log(
                    f"world/camera_{view_str}",
                    rr.Pinhole(
                        image_from_camera=view["camera_intrinsics"],
                        width=view["img"].shape[1],
                        height=view["img"].shape[0],
                    ),
                )

            print(f"Logged sample {sample_idx}: {sample['label']}")
