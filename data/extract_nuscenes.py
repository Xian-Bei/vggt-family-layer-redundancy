import argparse
import json
import random

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import shutil
import os
import numpy as np
from nuscenes.nuscenes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

try:
    from analysis.datasets.sparse_depth import SPARSE_DEPTH_FORMAT_VERSION, save_sparse_depth
except ImportError:
    from sparse_depth import SPARSE_DEPTH_FORMAT_VERSION, save_sparse_depth

w = 1.73
l = 4.08
h = 1.84
eps = 0.1

def extract(
    scene,
    data_root,
    nusc,
    save_root,
    *,
    min_depth=0.1,
    max_depth=80.0,
    min_depth_samples=100,
):
    # print('Processing ', scene['name'])
    cam_list = ['CAM_FRONT']
    scene_dir = os.path.join(save_root, scene['name'])
    shutil.rmtree(scene_dir, ignore_errors=True)
    ######################################################################
    for dir_name in ['lidar', 'ego', 'intrinsic']:
        dir_path = os.path.join(scene_dir, dir_name)
        shutil.rmtree(dir_path, ignore_errors=True)
        os.makedirs(dir_path, exist_ok=True)
    for cam_name in cam_list:
        cam_dir = os.path.join(scene_dir, 'image', cam_name)
        shutil.rmtree(cam_dir, ignore_errors=True)
        os.makedirs(cam_dir, exist_ok=True)
        cam2world_dir = os.path.join(scene_dir, 'cam2world', cam_name)
        shutil.rmtree(cam2world_dir, ignore_errors=True)
        os.makedirs(cam2world_dir, exist_ok=True)
        depth_dir = os.path.join(scene_dir, 'depth_points', cam_name)
        shutil.rmtree(depth_dir, ignore_errors=True)
        os.makedirs(depth_dir, exist_ok=True)
    ######################################################################
    first_sample_token = scene['first_sample_token']
    key_frame_id = 0
    sample_token = first_sample_token
    while sample_token != '':
        sample = nusc.get('sample', sample_token)
        # print('Processing ', key_frame_id)

        sensor_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', sensor_token)
        lidar2ego = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar2ego = transform_matrix(lidar2ego['translation'], Quaternion(lidar2ego['rotation']), inverse=False)
        ego2world = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego2world = transform_matrix(ego2world['translation'], Quaternion(ego2world['rotation']), inverse=False)
        lidar2world = ego2world @ lidar2ego
        # ######################################################################
        lidar_path = os.path.join(data_root, lidar_data['filename'])
        lidar_points = LidarPointCloud.from_file(lidar_path)
        lidar_points.transform(lidar2world)
        lidar_save_path = os.path.join(scene_dir, 'lidar', f'{key_frame_id:0>3d}.bin')
        xyz_lidar = lidar_points.points.T.astype(np.float32).reshape(-1, 4)[:, :3]
        inside_ego_range = np.array([
            [-w/2-eps, -l*0.4-eps, -h-eps],
            [w/2+eps, l*0.6+eps, eps]
        ])
        inside_mask = (xyz_lidar >= inside_ego_range[0:1]).all(1) & (xyz_lidar <= inside_ego_range[1:]).all(1) 
        xyz_lidar = xyz_lidar[~inside_mask]
        xyz_lidar.tofile(lidar_save_path)
        
        for sensor_name in cam_list:
            sensor_token = sample['data'][sensor_name]
            cam_data = nusc.get('sample_data', sensor_token)
            ######################################################################
            key_path = nusc.get_sample_data_path(cam_data['token'])
            key_save_path = os.path.join(scene_dir, 'image', sensor_name, f'{key_frame_id:0>3d}.jpg')
            shutil.copy(key_path, key_save_path)
            ######################################################################
            camera_pose = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            cam2ego = transform_matrix(camera_pose['translation'], Quaternion(camera_pose['rotation']), inverse=False)
            ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
            ego2world = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
            cam2world = ego2world @ cam2ego
            cam2world_save_path = os.path.join(scene_dir, 'cam2world', sensor_name, f'{key_frame_id:0>3d}.txt')
            np.savetxt(cam2world_save_path, cam2world)
            ######################################################################
            intrinsics = np.array(camera_pose['camera_intrinsic'])
            np.savetxt(os.path.join(scene_dir, 'intrinsic', f'{sensor_name}.txt'), intrinsics)
            ######################################################################
            # Official nuScenes projection handles the LiDAR/camera timestamps by
            # traversing calibrated-sensor and ego-pose records for both sensors.
            projected, camera_z, image = nusc.explorer.map_pointcloud_to_image(
                pointsensor_token=sample['data']['LIDAR_TOP'],
                camera_token=sensor_token,
                min_dist=float(min_depth),
                render_intensity=False,
            )
            sparse = save_sparse_depth(
                os.path.join(scene_dir, 'depth_points', sensor_name, f'{key_frame_id:0>3d}.npz'),
                u=projected[0],
                v=projected[1],
                depth=camera_z,
                image_width=int(image.size[0]),
                image_height=int(image.size[1]),
                source='nuscenes_devkit_map_pointcloud_to_image',
                min_depth=float(min_depth),
                max_depth=float(max_depth),
            )
            if sparse.count < int(min_depth_samples):
                raise RuntimeError(
                    f"{scene['name']} frame {key_frame_id}: official projection returned "
                    f"only {sparse.count} valid CAM_FRONT samples"
                )
        key_frame_id += 1
                
        sample_token = sample['next']
    print('Processing done ', scene['name'])


def extract_sparse_depth_only(
    scene,
    nusc,
    save_root,
    *,
    min_depth=0.1,
    max_depth=80.0,
    min_depth_samples=100,
):
    """Backfill official sparse depth without modifying existing extracted data."""
    sensor_name = 'CAM_FRONT'
    scene_dir = os.path.join(save_root, scene['name'])
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"Existing extracted scene not found: {scene_dir}")
    depth_dir = os.path.join(scene_dir, 'depth_points', sensor_name)
    os.makedirs(depth_dir, exist_ok=True)

    key_frame_id = 0
    sample_token = scene['first_sample_token']
    while sample_token != '':
        sample = nusc.get('sample', sample_token)
        sensor_token = sample['data'][sensor_name]
        image_path = os.path.join(scene_dir, 'image', sensor_name, f'{key_frame_id:0>3d}.jpg')
        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"Existing RGB frame missing; refusing depth-only backfill: {image_path}"
            )
        projected, camera_z, image = nusc.explorer.map_pointcloud_to_image(
            pointsensor_token=sample['data']['LIDAR_TOP'],
            camera_token=sensor_token,
            min_dist=float(min_depth),
            render_intensity=False,
        )
        sparse = save_sparse_depth(
            os.path.join(depth_dir, f'{key_frame_id:0>3d}.npz'),
            u=projected[0],
            v=projected[1],
            depth=camera_z,
            image_width=int(image.size[0]),
            image_height=int(image.size[1]),
            source='nuscenes_devkit_map_pointcloud_to_image',
            min_depth=float(min_depth),
            max_depth=float(max_depth),
        )
        if sparse.count < int(min_depth_samples):
            raise RuntimeError(
                f"{scene['name']} frame {key_frame_id}: official projection returned "
                f"only {sparse.count} valid CAM_FRONT samples"
            )
        key_frame_id += 1
        sample_token = sample['next']
    print('Sparse depth backfill done ', scene['name'])

def parse_args():
    ap = argparse.ArgumentParser("Extract front-camera nuScenes scenes to custom format.")
    ap.add_argument("--data_root", default="/media/fengyi/bb/nuscenes", help="nuScenes data root")
    ap.add_argument("--target_dir", default="data/nuscenes", help="Output root")
    ap.add_argument("--version", default="v1.0-trainval", help="nuScenes version")
    ap.add_argument("--seed", type=int, default=0, help="Fixed seed for train/val scene sampling")
    ap.add_argument(
        "--num_scenes_per_split",
        type=int,
        default=50,
        help="Number of scenes to sample from each of train and val",
    )
    ap.add_argument("--min_depth", type=float, default=0.1, help="Minimum valid camera Z depth in metres")
    ap.add_argument("--max_depth", type=float, default=80.0, help="Maximum valid camera Z depth in metres")
    ap.add_argument(
        "--min_depth_samples",
        type=int,
        default=100,
        help="Fail extraction when a frame has fewer official projected samples",
    )
    ap.add_argument(
        "--depth_only",
        action="store_true",
        help="Only backfill depth_points into existing scenes; never rewrite RGB/LiDAR/poses",
    )
    return ap.parse_args()


def select_train_val_scenes(nusc, num_scenes_per_split: int, seed: int):
    splits = create_splits_scenes()
    scene_by_name = {scene["name"]: scene for scene in nusc.scene}
    selected = {}
    for split in ("train", "val"):
        names = sorted(name for name in splits[split] if name in scene_by_name)
        n = min(int(num_scenes_per_split), len(names))
        rng = random.Random(f"{int(seed)}:{split}")
        selected[split] = sorted(rng.sample(names, n))
    return selected


if __name__ == "__main__":
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)
    os.makedirs(args.target_dir, exist_ok=True)
    manifest_path = os.path.join(args.target_dir, "selected_scenes.json")
    if args.depth_only:
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"Depth-only mode requires the existing selection manifest: {manifest_path}"
            )
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        selected = manifest.get("splits", {})
        for split in ("train", "val"):
            if not isinstance(selected.get(split), list):
                raise ValueError(f"Invalid selected_scenes.json split: {split}")
    else:
        selected = select_train_val_scenes(
            nusc,
            num_scenes_per_split=args.num_scenes_per_split,
            seed=args.seed,
        )
    if not args.depth_only:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": args.version,
                    "seed": int(args.seed),
                    "num_scenes_per_split": int(args.num_scenes_per_split),
                    "sparse_depth": {
                        "format_version": SPARSE_DEPTH_FORMAT_VERSION,
                        "depth_type": "camera_z",
                        "source": "nuscenes_devkit_map_pointcloud_to_image",
                        "min_depth": float(args.min_depth),
                        "max_depth": float(args.max_depth),
                    },
                    "splits": selected,
                },
                f,
                indent=2,
            )

    scene_by_name = {scene["name"]: scene for scene in nusc.scene}
    for split in ("train", "val"):
        for scene_name in selected[split]:
            if args.depth_only:
                extract_sparse_depth_only(
                    scene_by_name[scene_name],
                    nusc=nusc,
                    save_root=args.target_dir,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    min_depth_samples=args.min_depth_samples,
                )
            else:
                extract(
                    scene_by_name[scene_name],
                    data_root=args.data_root,
                    nusc=nusc,
                    save_root=args.target_dir,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    min_depth_samples=args.min_depth_samples,
                )
