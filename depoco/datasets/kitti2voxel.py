
import numpy as np
import argparse
from numpy.linalg import inv
import time
import os
from ruamel import yaml
from matplotlib import pyplot as plt
import open3d as o3d
import depoco.utils.point_cloud_utils as pcu
from pathlib import Path
import octree_handler


def open_label(filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
        raise TypeError("Filename should be string type, "
                        "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    # if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
    #     raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.int32)
    label = label.reshape((-1))
    label = label & 0xFFFF

    # set it
    return label


def parse_calibration(filename):
    """ read calibration file with given filename
        Returns
        -------
        dict
            Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        # print(key,values)
        if len(values) == 12:
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

    calib_file.close()
    print('calibration', calib)
    return calib


def parse_poses(filename, calibration):
    """ read poses file with per-scan poses from given filename
        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
    """
    file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = inv(Tr)

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses


def distanceMatrix(x, y):
    """ x[nxd], y[mxd],  d = dimensionality (eg. 3)
     distance between each point of x to each point of y
    Return
    -----
    distance matrix [nxm]

    """
    dims = x.shape[1]
    dist = np.zeros((x.shape[0], y.shape[0]))
    for i in range(dims):
        dist += (x[:, i][..., np.newaxis] - y[:, i][np.newaxis, ...])**2
    return dist**0.5


def getKeyPoses(pose_list, delta=50):
    '''
    creates a key pose for every pose which is delta apart (horizontal distance)
    returns
    -------
      idx, poses, distance_matrix [nxn]
    '''
    poses = np.asarray(pose_list)
    xy = poses[:, 0:2, -1]
    dist = distanceMatrix(xy, xy)

    key_pose_idx = []
    indices = np.arange(poses.shape[0])

    dist_it = dist.copy()
    while (dist_it.shape[0] > 0):
        key_pose_idx.append(indices[0])
        valid_idx = dist_it[0, :] > delta
        dist_it = dist_it[valid_idx, :]
        dist_it = dist_it[:, valid_idx]
        indices = indices[valid_idx]
    return key_pose_idx, poses[key_pose_idx], dist


class Kitti2voxelConverter():
    def __init__(self, config):
        self.config = config
        self.train_folders = []
        self.valid_folders = []
        self.test_folders = []
        if type(config["dataset"]["data_folders"]["train"]) is list:
            self.train_folders = [pcu.path(config["dataset"]["data_folders"]["prefix"])+pcu.path(
                fldid) for fldid in config["dataset"]["data_folders"]["train"]]
        if type(config["dataset"]["data_folders"]["valid"]) is list:
            self.valid_folders = [pcu.path(config["dataset"]["data_folders"]["prefix"])+pcu.path(
                fldid) for fldid in config["dataset"]["data_folders"]["valid"]]
        if type(config["dataset"]["data_folders"]["test"]) is list:
            self.test_folders = [pcu.path(config["dataset"]["data_folders"]["prefix"])+pcu.path(
                fldid) for fldid in config["dataset"]["data_folders"]["test"]]

    def getMaxMinHeight(self):
        folders = self.train_folders + self.valid_folders + self.test_folders
        n_bins = 1000
        hist = np.zeros((n_bins))
        time_start = time.time()

        for p in folders:
            # calibration = parse_calibration(p+ "calib.txt")
            # poses = parse_poses( p+"poses.txt", calibration)
            scan_files = [
                f for f in sorted(os.listdir(os.path.join(p, "velodyne")))
                if f.endswith(".bin")]
            for i, f in enumerate(scan_files):
                scan = np.fromfile(p+"velodyne/"+f, dtype=np.float32)
                scan = scan.reshape((-1, 4))
                hist_i, temp_range = np.histogram(
                    scan[:, 2], bins=n_bins, range=(-30, 30))
                hist += hist_i

            print('min max time', time.time()-time_start)
            print('hist_size', hist.shape)

        plt.figure()
        plt.plot(temp_range[1:], hist)
        np.savetxt('hist', hist)
        plt.show()

    def sparsifieO3d(self, poses, key_pose_idx, seq_path, distance_matrix):
        grid_size = np.array((self.config['grid']['size']))
        center = poses[key_pose_idx][0:3, -1] + \
            np.array((0, 0, self.config['grid']['dz']))
        upper_bound = center + grid_size/2
        lower_bound = center - grid_size/2
        valid_scans = np.argwhere(
            distance_matrix[key_pose_idx, :] < grid_size[0] + self.config['grid']['max_range']).squeeze()
        # print('valid scans', valid_scans,'shape',valid_scans.shape)
        point_cld = ()
        features = ()
        for i in valid_scans:
            sfile = seq_path + "velodyne/" + str(i).zfill(6)+'.bin'
            scan = np.fromfile(sfile, dtype = np.float32) if os.path.isfile(sfile) else np.zeros((0, 4))
            scan=scan.reshape((-1, 4))
            dists=np.linalg.norm(scan[:, 0:3], axis=1)
            valid_p=(dists > self.config['grid']['min_range']) & (
                dists < self.config['grid']['max_range'])
            scan_hom=np.ones_like(scan)
            scan_hom[:, 0:3]=scan[:, 0:3]
            points=np.matmul(poses[i], scan_hom[valid_p, :].T).T
            #### intensity and label #######
            intensity=scan[valid_p, 3:4]
            label=np.full((points.shape[0],), 2)
            if os.path.isfile(seq_path + "labels/" +
                               str(i).zfill(6)+'.label'):
                label=open_label(filename=seq_path + "labels/" +
                               str(i).zfill(6)+'.label')[valid_p]
            # print('max min label',np.max(label),np.min(label))
            feature=np.hstack((intensity, np.expand_dims(
                label.astype('float'), axis=1), np.zeros_like(intensity)))  # concat intensity,label and a 0 (0 for havin 3dims)

            points=points[:, 0:3]
            # print('features', feature.shape, 'points', points.shape)
            valids=np.all(points > lower_bound, axis=1) & np.all(
                points < upper_bound, axis=1).reshape(-1) & (label < 200) & (label > 1)  # remove moving objects and outlier
            point_cld += (points[valids, :],)
            features += (feature[valids],)

        # o3d
        pcd=o3d.geometry.PointCloud()
        cloud=np.concatenate(point_cld)
        cloud_clr=np.concatenate(features)
        pcd.points=o3d.utility.Vector3dVector(cloud)
        pcd.colors=o3d.utility.Vector3dVector(cloud_clr)
        downpcd=pcd.voxel_down_sample(
            voxel_size=self.config['grid']['voxel_size'])
        sparse_points=np.asarray(downpcd.points)
        sparse_features=np.asarray(downpcd.colors)
        sparse_features=sparse_features[:, :2]
        sparse_features[:, 1]=np.around(sparse_features[:, 1])
        sparse_points=np.hstack((sparse_points, sparse_features))
        print('#points', cloud.shape, 'to', sparse_points.shape,
              'features', sparse_features.shape)

        # print('grid time', time.time() - time_start)
        return sparse_points

    def convert(self):
        time_very_start=time.time()
        folders=self.train_folders + self.valid_folders + self.test_folders
        for j, p in enumerate(folders):
            # print('pcu.path',p.split('/')[-2])
            out_dir=pcu.path(
                self.config['dataset']['data_folders']['grid_output'])+pcu.path(p.split('/')[-2])
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            calibration=parse_calibration(p + "calib.txt")
            poses=parse_poses(p+"poses.txt", calibration)
            scan_files=[
                f for f in sorted(os.listdir(os.path.join(p, "velodyne")))
                if f.endswith(".bin")]
            key_poses_idx, key_poses, distance_matrix=getKeyPoses(
                poses, self.config['grid']['pose_distance'])
            print('kp shape', key_poses.shape)
            np.savetxt(out_dir+'key_poses.txt',
                       np.reshape(key_poses, (key_poses.shape[0], 16)))
            for i, idx in enumerate(key_poses_idx):
                bla=1
                time_start=time.time()
                sparse_points_features=self.sparsifieO3d(
                    poses, idx, p, distance_matrix).astype('float32')
                print('seq', j, 'from', len(folders),
                      'keypose', i, 'from', len(key_poses_idx))
                print('sparsifie time', time.time() - time_start)
                time_start=time.time()
                octree=octree_handler.Octree()
                points=sparse_points_features[:, :3]
                octree.setInput(points)
                eig_normals=octree.computeEigenvaluesNormal(
                    self.config['grid']['normal_eigenvalue_radius'])
                sparse_points_features=np.hstack(
                    (sparse_points_features, eig_normals))
                print('normal and eigenvalues estimation time',
                      time.time() - time_start)
                # print('sparse_points',sparse_points.shape)
                pcu.saveCloud2Binary(sparse_points_features, str(
                    i).zfill(6)+'.bin', out_dir)

                # time_start = time.time()
                # sparse_points = self.sparsifieVoxelGrid(poses, idx, p, distance_matrix)
                # sparse_points = self.sparsifieO3d(poses, idx, p, distance_matrix)
                # print('sparsifie2 time', time.time() - time_start)
                # print('sparse_points2',sparse_points.shape)
                # pcu.visPointCloud(sparse_points)
                # np.savetxt('testgrid.xyz',sparse_points)
                # return None
        print('convert time', time.time() - time_very_start)


if __name__ == "__main__":
    start_time=time.time()

    parser=argparse.ArgumentParser("./kitti2voxel.py")
    parser.add_argument(
        '--dataset',
        '-d',
        type=str,
        required=False,
        default="/mnt/91d100fa-d283-4eeb-b68c-e2b4b199d2de/wiesmann/data/data_kitti/dataset",
        help='dataset folder containing all sequences in a folder called "sequences".',
    )
    parser.add_argument(
        '--arch_cfg', '-cfg',
        type=str,
        required=False,
        default='config/arch/sample_net.yaml',
        help='Architecture yaml cfg file. See /config/arch for sample. No default!',
    )

    FLAGS, unparsed=parser.parse_known_args()
    ARCH=yaml.safe_load(open(FLAGS.arch_cfg, 'r'))

    input_folder=FLAGS.dataset + '/sequences/00/'
    calibration=parse_calibration(os.path.join(input_folder, "calib.txt"))
    poses=parse_poses(os.path.join(input_folder, "poses.txt"), calibration)
    idx, keypose, d=getKeyPoses(poses, delta=ARCH["grid"]["pose_distance"])

    xy=np.asarray(poses)
    xy=xy[:, 0:2, -1]
    x=xy[:, 0]
    y=xy[:, 1]
    print(xy.shape)
    plt.figure
    plt.plot(xy[:, 0], xy[:, 1])
    plt.plot(xy[idx, 0], xy[idx, 1], 'xr')
    plt.axis('equal')

    plt.show()

    converter=Kitti2voxelConverter(ARCH)
    # converter.getMaxMinHeight() # -9 to 4
    converter.convert()
