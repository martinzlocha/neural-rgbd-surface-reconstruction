import os
import imageio
import json
import cv2
from tqdm import tqdm
from pyquaternion import Quaternion
from dataloader_util import *


def load_record3d_data(basedir, trainskip, downsample_factor=1, translation=0.0, sc_factor=1., crop=0):
    # Get image filenames, poses and intrinsics
    img_files = [f for f in sorted(os.listdir(os.path.join(basedir, 'rgb')), key=alphanum_key) if f.endswith('jpg')]
    depth_files = [f for f in sorted(os.listdir(os.path.join(basedir, 'depth')), key=alphanum_key) if f.endswith('exr')]

    with open(os.path.join(basedir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    # Train, val and test split
    num_frames = len(img_files)
    train_frame_ids = list(range(0, num_frames, trainskip))

    # Lists for the data to load into
    images = []
    depth_maps = []
    poses = []
    frame_indices = []

    # Read images and depth maps for which valid poses exist
    for i in tqdm(train_frame_ids):
        depth = cv2.imread(os.path.join(basedir, 'depth', depth_files[i]), -1)
        depth = resize_images(np.array([depth]), depth.shape[0] * 4, depth.shape[1] * 4)[0]
        depth = depth[:, :, 2]

        img = imageio.imread(os.path.join(basedir, 'rgb', img_files[i]))
        img = resize_images(np.array([img]), depth.shape[0], depth.shape[1])[0]

        pose_arr = metadata['poses'][i]
        quat = Quaternion(w=pose_arr[3], x=pose_arr[0], y=pose_arr[1], z=pose_arr[2])
        pose_matrix = quat.transformation_matrix
        pose_matrix[:3, 3] = pose_arr[4:]

        images.append(img)
        depth_maps.append(depth)
        poses.append(pose_matrix)
        frame_indices.append(i)

    # Map images to [0, 1] range
    images = (np.array(images) / 255.).astype(np.float32)

    depth_maps = np.array(depth_maps).astype(np.float32)
    depth_maps *= sc_factor
    depth_maps = depth_maps[..., np.newaxis]

    poses = np.array(poses).astype(np.float32)
    poses[:, :3, 3] += translation
    poses[:, :3, 3] *= sc_factor

    # Intrinsics
    image_H = metadata['h']
    H, W = depth_maps[0].shape[:2]
    focal = metadata['K'][0]
    focal *= (H / image_H)

    # Crop the undistortion artifacts
    if crop > 0:
        images = images[:, crop:-crop, crop:-crop, :]
        depth_maps = depth_maps[:, crop:-crop, crop:-crop, :]
        H, W = depth_maps[0].shape[:2]

    if downsample_factor > 1:
        H = H//downsample_factor
        W = W//downsample_factor
        focal = focal/downsample_factor
        images = resize_images(images, H, W)
        depth_maps = resize_images(depth_maps, H, W, interpolation=cv2.INTER_NEAREST)

    print(f'Image shape: {images.shape}')
    print(f'Depth map shape: {depth_maps.shape}')
    print(f'Depth min: {np.min(depth_maps)}')
    print(f'Depth max: {np.max(depth_maps)}')
    print(f'Poses: {poses.shape}')
    print(f'Focal: {focal}')
    print(f'X: {np.min(poses[:, 0, 3])} - {np.mean(poses[:, 0, 3])} - {np.max(poses[:, 0, 3])}')
    print(f'X: {np.min(poses[:, 1, 3])} - {np.mean(poses[:, 1, 3])} - {np.max(poses[:, 1, 3])}')
    print(f'X: {np.min(poses[:, 2, 3])} - {np.mean(poses[:, 2, 3])} - {np.max(poses[:, 2, 3])}')

    return images, depth_maps, poses, [H, W, focal], frame_indices
