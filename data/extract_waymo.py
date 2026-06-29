'''This code is adapted from DriveStudio (https://github.com/ziyc/drivestudio/blob/main/datasets/waymo/)'''
# Acknowledgement:
#   1. https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/waymo_converter.py
#   2. https://github.com/leolyj/DCA-SRSFE/blob/main/data_preprocess/Waymo/generate_flow.py
try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-6-0" '
        ">1.4.5 to install the official devkit first."
    )

import argparse
from glob import glob
import json
import os
import shutil

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset.utils.frame_utils import (
    convert_range_image_to_point_cloud,
    parse_range_image_and_camera_projection,
)
from os.path import join

try:
    from analysis.datasets.sparse_depth import SPARSE_DEPTH_FORMAT_VERSION, save_sparse_depth
except ImportError:
    from sparse_depth import SPARSE_DEPTH_FORMAT_VERSION, save_sparse_depth

OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
)

WAYMO_CAMERA_NAMES = [
    "FRONT",
    "FRONT_LEFT",
    "FRONT_RIGHT",
    "SIDE_LEFT",
    "SIDE_RIGHT",
]

from waymo_open_dataset.utils import range_image_utils, transform_utils


def compute_range_image_cartesian(
    range_image_polar,
    extrinsic,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
):
    """Computes range image cartesian coordinates from polar ones.

    Args:
      range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
        coordinate in sensor frame.
      extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
      pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
        range image pixel.
      frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
        It decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_cartesian: [B, H, W, 3] cartesian coordinates.
    """
    range_image_polar_dtype = range_image_polar.dtype
    range_image_polar = tf.cast(range_image_polar, dtype=dtype)
    extrinsic = tf.cast(extrinsic, dtype=dtype)
    if pixel_pose is not None:
        pixel_pose = tf.cast(pixel_pose, dtype=dtype)
    if frame_pose is not None:
        frame_pose = tf.cast(frame_pose, dtype=dtype)

    with tf.compat.v1.name_scope(
        scope,
        "ComputeRangeImageCartesian",
        [range_image_polar, extrinsic, pixel_pose, frame_pose],
    ):
        azimuth, inclination, range_image_range = tf.unstack(range_image_polar, axis=-1)

        cos_azimuth = tf.cos(azimuth)
        sin_azimuth = tf.sin(azimuth)
        cos_incl = tf.cos(inclination)
        sin_incl = tf.sin(inclination)

        # [B, H, W].
        x = cos_azimuth * cos_incl * range_image_range
        y = sin_azimuth * cos_incl * range_image_range
        z = sin_incl * range_image_range

        # [B, H, W, 3]
        range_image_points = tf.stack([x, y, z], -1)
        range_image_origins = tf.zeros_like(range_image_points)
        # [B, 3, 3]
        rotation = extrinsic[..., 0:3, 0:3]
        # translation [B, 1, 3]
        translation = tf.expand_dims(tf.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

        # To vehicle frame.
        # [B, H, W, 3]
        range_image_points = (
            tf.einsum("bkr,bijr->bijk", rotation, range_image_points) + translation
        )
        range_image_origins = (
            tf.einsum("bkr,bijr->bijk", rotation, range_image_origins) + translation
        )
        if pixel_pose is not None:
            # To global frame.
            # [B, H, W, 3, 3]
            pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
            # [B, H, W, 3]
            pixel_pose_translation = pixel_pose[..., 0:3, 3]
            # [B, H, W, 3]
            range_image_points = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_points)
                + pixel_pose_translation
            )
            range_image_origins = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_origins)
                + pixel_pose_translation
            )

            if frame_pose is None:
                raise ValueError("frame_pose must be set when pixel_pose is set.")
            # To vehicle frame corresponding to the given frame_pose
            # [B, 4, 4]
            world_to_vehicle = tf.linalg.inv(frame_pose)
            world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
            world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
            # [B, H, W, 3]
            range_image_points = (
                tf.einsum(
                    "bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_points
                )
                + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
            )
            range_image_origins = (
                tf.einsum(
                    "bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_origins
                )
                + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
            )

        range_image_points = tf.cast(range_image_points, dtype=range_image_polar_dtype)
        range_image_origins = tf.cast(
            range_image_origins, dtype=range_image_polar_dtype
        )
        return range_image_points, range_image_origins


def extract_point_cloud_from_range_image(
    range_image,
    extrinsic,
    inclination,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
):
    """Extracts point cloud from range image.

    Args:
      range_image: [B, H, W] tensor. Lidar range images.
      extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
      inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
      pixel_pose: [B, H, W, 4, 4] tensor. If not None, it sets pose for each range
        image pixel.
      frame_pose: [B, 4, 4] tensor. This must be set when pixel_pose is set. It
        decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_points: [B, H, W, 3] with {x, y, z} as inner dims in vehicle frame.
      range_image_origins: [B, H, W, 3] with {x, y, z}, the origin of the range image
    """
    with tf.compat.v1.name_scope(
        scope,
        "ExtractPointCloudFromRangeImage",
        [range_image, extrinsic, inclination, pixel_pose, frame_pose],
    ):
        range_image_polar = range_image_utils.compute_range_image_polar(
            range_image, extrinsic, inclination, dtype=dtype
        )
        (
            range_image_points_cartesian,
            range_image_origins_cartesian,
        ) = compute_range_image_cartesian(
            range_image_polar,
            extrinsic,
            pixel_pose=pixel_pose,
            frame_pose=frame_pose,
            dtype=dtype,
        )
        return range_image_origins_cartesian, range_image_points_cartesian



class WaymoProcessor(object):
    """Process Waymo dataset.

    Args:
        load_dir (str): Directory to load waymo raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename.
        workers (int, optional): Number of workers for the parallel process.
            Defaults to 64.
            Defaults to False.
        save_cam_sync_labels (bool, optional): Whether to save cam sync labels.
            Defaults to True.
    """

    def __init__(
        self,
        load_dir,
        save_dir,
        process_keys=[
            "images",
            "lidar",
            "calib",
            "depth",
        ],
        workers=64,
        min_depth=0.1,
        max_depth=80.0,
        min_depth_samples=100,
        depth_only=False,
    ):
        self.filter_no_label_zone_points = True

        # Only data collected in specific locations will be converted
        # If set None, this filter is disabled
        # Available options: location_sf (main dataset)
        self.selected_waymo_locations = None
        self.save_track_id = False
        self.process_keys = process_keys
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.min_depth_samples = int(min_depth_samples)
        self.depth_only = bool(depth_only)
        print("will process keys: ", self.process_keys)

        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split(".")[0]) < 2:
            tf.enable_eager_execution()

        # Export only the front camera. ``WAYMO_CAMERA_NAMES`` keeps the
        # official proto id -> name mapping for filtering.
        self.cam_list = ["FRONT"]

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.workers = int(workers)
        # a list of tfrecord pathnames
        self.tfrecord_pathnames = sorted(glob(join(self.load_dir, "*.tfrecord")))
        self.scene_name = [f"{name.split('-')[1].split('_')[0]}" for name in sorted(os.listdir(self.load_dir)) if name.endswith('.tfrecord')]
        

    def convert_one(self, file_idx):
        """Convert action for single file.

        Args:
            file_idx (int): Index of the file to be converted.
        """
        self.create_folder(file_idx)
        pathname = self.tfrecord_pathnames[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type="")
        num_frames = sum(1 for _ in dataset)
        frame_idx = 0
        for i, data in enumerate(
            tqdm(dataset, desc=f"File {self.scene_name[file_idx]}", total=num_frames, dynamic_ncols=True)
        ):
            if i % 5 != 0:
                continue
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if (
                self.selected_waymo_locations is not None
                and frame.context.stats.location not in self.selected_waymo_locations
            ):
                continue
            if "images" in self.process_keys:
                self.save_image(frame, self.scene_name[file_idx], frame_idx)
            if "calib" in self.process_keys:
                self.save_calib(frame, self.scene_name[file_idx], frame_idx)
            parsed_range_data = None
            if "lidar" in self.process_keys or "depth" in self.process_keys:
                parsed_range_data = parse_range_image_and_camera_projection(frame)
            if "lidar" in self.process_keys:
                self.save_lidar(frame, self.scene_name[file_idx], frame_idx, parsed_range_data)
            if "depth" in self.process_keys:
                self.save_sparse_camera_depth(frame, self.scene_name[file_idx], frame_idx, parsed_range_data)
            frame_idx += 1

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)


    def save_image(self, frame, file_name, frame_idx):
        """Parse and save the images in jpg format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        for img in frame.images:
            cam_name = WAYMO_CAMERA_NAMES[img.name - 1]
            if cam_name not in self.cam_list:
                continue
            img_path = (
                join(self.save_dir, f"{file_name}/image/{cam_name}/{str(frame_idx).zfill(3)}.jpg")
            )
            with open(img_path, "wb") as fp:
                fp.write(img.image)

    def save_calib(self, frame, file_name, frame_idx):
        """Parse and save the calibration data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_name (str): Current file name.
            frame_idx (int): Current frame index.
        """
        # waymo front camera to kitti reference camera
        
        ego2world = np.array(frame.pose.transform).reshape(4, 4)

        for camera in frame.context.camera_calibrations:
            cam_name = WAYMO_CAMERA_NAMES[camera.name - 1]
            if cam_name not in self.cam_list:
                continue
            # extrinsic parameters
            cam2ego = np.array(camera.extrinsic.transform).reshape(4, 4) @ OPENCV2DATASET
            cam2world = ego2world @ cam2ego
            intrinsic = list(camera.intrinsic)

            def _K_from_intrinsic_list(intrinsic_list) -> np.ndarray:
                fx, fy, cx, cy = intrinsic_list[:4]
                K = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]], dtype=np.float64)
                return K
            
            intrinsic = _K_from_intrinsic_list(intrinsic)
            np.savetxt(
                join(f"{self.save_dir}", f"{file_name}/cam2world/{cam_name}/{str(frame_idx).zfill(3)}.txt"),
                cam2world,
            )
            np.savetxt(
                join(f"{self.save_dir}", f"{file_name}/intrinsic/{cam_name}.txt"),
                intrinsic,
            )
        # all camera ids are saved as id-1 in the result because
        # camera 0 is unknown in the proto


    def save_lidar(self, frame, file_name, frame_idx, parsed_range_data=None):
        """Save lidar point cloud.
        Args:
            color (bool): Whether to colorize lidar points using camera projection.
                        If False, only XYZ (float32) will be saved.
        """
        if parsed_range_data is None:
            parsed_range_data = parse_range_image_and_camera_projection(frame)
        (range_images, camera_projections, seg_labels, range_image_top_pose) = parsed_range_data
        if range_image_top_pose is None:
            return

        frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(range_image_top_pose.data),
            range_image_top_pose.shape.dims
        )
        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0],
            range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2],
        )
        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation
        )

        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        all_pts, all_rgb = [], []

        for c in calibrations:
            ri_index = 0
            if c.name not in range_images:
                continue
            range_image = range_images[c.name][ri_index]

            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0],
                )
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)
            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])

            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims
            )
            range_image_mask = range_image_tensor[..., 0] > 0
            mask_index = tf.where(range_image_mask)

            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = tf.expand_dims(range_image_top_pose_tensor, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)

            _, points_cart = extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local,
            )
            points_cart = tf.squeeze(points_cart, axis=0)
            points_tensor = tf.gather_nd(points_cart, mask_index)
            pts = points_tensor.numpy()
            all_pts.append(pts)

        pts = np.concatenate(all_pts, axis=0)

        ego2world = np.array(frame.pose.transform).reshape(4, 4)
        pts = (ego2world[:3, :3] @ pts.T + ego2world[:3, 3:4]).T
        pts.astype(np.float32).tofile(
            f"{self.save_dir}/{file_name}/lidar/{str(frame_idx).zfill(3)}.bin"
        )

    def save_sparse_camera_depth(self, frame, file_name, frame_idx, parsed_range_data=None):
        """Save official range-image camera projections with OpenCV camera Z depth."""
        if parsed_range_data is None:
            parsed_range_data = parse_range_image_and_camera_projection(frame)
        range_images, camera_projections, _seg_labels, range_image_top_pose = parsed_range_data
        if range_image_top_pose is None:
            raise RuntimeError(f"{file_name} frame {frame_idx}: missing TOP range-image pose")

        # This official helper preserves the correspondence between motion-
        # compensated vehicle-frame points and the camera-projection records.
        points_by_laser, cp_by_laser = convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
        )
        camera = next(
            (c for c in frame.context.camera_calibrations if c.name == dataset_pb2.CameraName.FRONT),
            None,
        )
        if camera is None:
            raise RuntimeError(f"{file_name} frame {frame_idx}: FRONT calibration missing")

        vehicle_from_camera_cv = np.asarray(camera.extrinsic.transform, dtype=np.float64).reshape(4, 4) @ OPENCV2DATASET
        camera_cv_from_vehicle = np.linalg.inv(vehicle_from_camera_cv)
        all_u, all_v, all_z = [], [], []
        for points, cp_points in zip(points_by_laser, cp_by_laser):
            points = points.numpy() if hasattr(points, "numpy") else np.asarray(points)
            cp_points = cp_points.numpy() if hasattr(cp_points, "numpy") else np.asarray(cp_points)
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            cp_points = np.asarray(cp_points).reshape(-1, 6)
            if points.shape[0] != cp_points.shape[0]:
                raise RuntimeError(
                    f"{file_name} frame {frame_idx}: official point/projection count mismatch "
                    f"{points.shape[0]} vs {cp_points.shape[0]}"
                )
            points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
            camera_points = (camera_cv_from_vehicle @ points_h.T).T
            camera_z = camera_points[:, 2]
            # Each Waymo point can carry a primary and a secondary camera projection.
            for offset in (0, 3):
                projection = cp_points[:, offset : offset + 3]
                keep = projection[:, 0].astype(np.int64) == int(dataset_pb2.CameraName.FRONT)
                if np.any(keep):
                    all_u.append(projection[keep, 1].astype(np.float32, copy=False))
                    all_v.append(projection[keep, 2].astype(np.float32, copy=False))
                    all_z.append(camera_z[keep].astype(np.float32, copy=False))

        if not all_z:
            raise RuntimeError(f"{file_name} frame {frame_idx}: no official FRONT camera projections")
        sparse = save_sparse_depth(
            join(self.save_dir, file_name, "depth_points", "FRONT", f"{str(frame_idx).zfill(3)}.npz"),
            u=np.concatenate(all_u),
            v=np.concatenate(all_v),
            depth=np.concatenate(all_z),
            image_width=int(camera.width),
            image_height=int(camera.height),
            source="waymo_devkit_convert_range_image_to_point_cloud_camera_projection",
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )
        if sparse.count < self.min_depth_samples:
            raise RuntimeError(
                f"{file_name} frame {frame_idx}: official projection returned only "
                f"{sparse.count} valid FRONT samples"
            )

    def create_folder(self, i):
        """Create folder for data preprocessing."""
        scene_dir = join(self.save_dir, f"{self.scene_name[i]}")
        if self.depth_only:
            if not os.path.isdir(scene_dir):
                raise FileNotFoundError(
                    f"Existing extracted Waymo scene not found for depth-only backfill: {scene_dir}"
                )
        else:
            shutil.rmtree(scene_dir, ignore_errors=True)
        if "images" in self.process_keys:
            for cam in self.cam_list:
                os.makedirs(join(self.save_dir, f"{self.scene_name[i]}/image/{cam}"), exist_ok=True)
        if "calib" in self.process_keys:
            for cam in self.cam_list:
                os.makedirs(join(self.save_dir, f"{self.scene_name[i]}/cam2world/{cam}"), exist_ok=True)
            os.makedirs(join(self.save_dir, f"{self.scene_name[i]}/intrinsic"), exist_ok=True)
        if "lidar" in self.process_keys:
            os.makedirs(f"{self.save_dir}/{self.scene_name[i]}/lidar", exist_ok=True)
        if "depth" in self.process_keys:
            os.makedirs(join(self.save_dir, f"{self.scene_name[i]}/depth_points/FRONT"), exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser("Waymo to custom format (per-scene folders; world-coord cam2world & lidar)")
    ap.add_argument("--data_root", default="/media/fengyi/bb/waymo_raw", help="Waymo data root containing .tfrecord files")
    ap.add_argument("--target_dir", default="data/waymo", help="Output root")
    ap.add_argument("--min_depth", type=float, default=0.1, help="Minimum valid OpenCV camera Z in metres")
    ap.add_argument("--max_depth", type=float, default=80.0, help="Maximum valid OpenCV camera Z in metres")
    ap.add_argument(
        "--min_depth_samples",
        type=int,
        default=100,
        help="Fail extraction when a frame has fewer official projected FRONT samples",
    )
    ap.add_argument(
        "--depth_only",
        action="store_true",
        help="Only backfill depth_points into existing scenes; never rewrite RGB/LiDAR/calibration",
    )


    return ap.parse_args()


def main():
    args = parse_args()
    process_keys = ["depth"] if args.depth_only else ["images", "lidar", "calib", "depth"]
    dataset_processor = WaymoProcessor(
        load_dir=args.data_root,
        save_dir=args.target_dir,
        process_keys=process_keys,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        min_depth_samples=args.min_depth_samples,
        depth_only=args.depth_only,
    )
    for file_idx in range(len(dataset_processor)):
        dataset_processor.convert_one(file_idx)
    os.makedirs(args.target_dir, exist_ok=True)
    with open(join(args.target_dir, "sparse_depth_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "format_version": SPARSE_DEPTH_FORMAT_VERSION,
                "depth_type": "camera_z",
                "source": "waymo_devkit_convert_range_image_to_point_cloud_camera_projection",
                "min_depth": float(args.min_depth),
                "max_depth": float(args.max_depth),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
