
import numpy as np
import depoco.utils.point_cloud_utils as pcu
# import torch
import ruamel.yaml as yaml
import argparse
import glob
import time
from typing import Tuple, Union
from torch.utils.data import Dataset, Sampler
import torch
import os
########################################
# Torch Data loader
########################################


class SubMapParser():
    def __init__(self, config):
        self.config = config
        nr_submaps = config['train']['nr_submaps']
        self.grid_size = np.reshape(np.asarray(config['grid']['size']), (1, 3))
        # collect directories of the datasets
        out_path = os.environ.get(
            'DATA_SUBMAPS', config["dataset"]["data_folders"]["grid_output"])

        self.train_folders = [pcu.path(out_path)+pcu.path(
            fldid) for fldid in config["dataset"]["data_folders"]["train"]] if config["dataset"]["data_folders"]["train"] else []
        self.valid_folders = [pcu.path(out_path)+pcu.path(
            fldid) for fldid in config["dataset"]["data_folders"]["valid"]] if config["dataset"]["data_folders"]["valid"] else []
        # print('valid folders',self.valid_folders)
        self.test_folders = [pcu.path(out_path)+pcu.path(
            fldid) for fldid in config["dataset"]["data_folders"]["test"]] if config["dataset"]["data_folders"]["test"] else []
        cols = 3+sum(config['grid']['feature_dim'])
        # Trainingset
        self.train_dataset = SubMapDataSet(data_dirs=self.train_folders,
                                           nr_submaps=nr_submaps,
                                           nr_points=config['train']['max_nr_pts'], cols=cols, on_the_fly=True,
                                           grid_size=np.max(self.grid_size))
        self.train_sampler = SubMapSampler(nr_submaps=len(self.train_dataset),
                                           sampling_method=config['train']['sampling_method'])
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        sampler=self.train_sampler,
                                                        batch_size=None,
                                                        num_workers=0,
                                                        )
        self.train_iter = iter(self.train_loader)

        # Ordered Trainingset
        self.train_loader_ordered = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                                batch_size=None,
                                                                shuffle=False,
                                                                num_workers=config['train']['workers'])
        self.train_iter_ordered = iter(self.train_loader_ordered)

        # Validationset
        self.valid_dataset = SubMapDataSet(data_dirs=self.valid_folders,
                                           nr_submaps=0,
                                           nr_points=config['train']['max_nr_pts'],
                                           cols=cols, on_the_fly=True,
                                           grid_size=np.max(self.grid_size))
        self.valid_sampler = SubMapSampler(nr_submaps=len(self.valid_dataset),
                                           sampling_method='ordered')
        self.valid_loader = torch.utils.data.DataLoader(dataset=self.valid_dataset,
                                                        batch_size=None,
                                                        sampler=self.valid_sampler,
                                                        num_workers=config['train']['workers'],
                                                        )
        self.valid_iter = iter(self.valid_loader)

        # Testset
        self.test_dataset = SubMapDataSet(data_dirs=self.test_folders,
                                          nr_submaps=0,
                                          nr_points=config['train']['max_nr_pts'],
                                          cols=cols, on_the_fly=True,
                                          grid_size=np.max(self.grid_size))
        self.test_sampler = SubMapSampler(nr_submaps=len(self.test_dataset),
                                          sampling_method='ordered')
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=None,
                                                       sampler=self.test_sampler,
                                                       num_workers=config['train']['workers'],
                                                       )
        self.test_iter = iter(self.test_loader)
        # print('test folder:',self.test_folders)

    def getOrderedTrainSet(self):
        return self.train_loader_ordered

    def setTrainProbabilities(self, probs):
        self.train_loader.sampler.setSampleProbs(probs)

    def getTrainBatch(self):
        scans = self.train_iter.next()
        return scans

    def getTrainSet(self):
        return self.train_loader

    def getValidBatch(self):
        scans = self.valid_iter.next()
        return scans

    def getValidSet(self):
        return self.valid_loader

    def getTestBatch(self):
        scans = self.test_iter.next()
        return scans

    def getTestSet(self):
        return self.test_loader

    def getTrainSize(self):
        return len(self.train_loader)

    def getValidSize(self):
        return len(self.valid_loader)

    def getTestSize(self):
        return len(self.test_loader)


class SubMapSampler(Sampler):
    def __init__(self, nr_submaps, sampling_method='random', nr_samples=-1):
        """[summary]

        Arguments:
            nr_submaps {int} -- [description]
            sampling_method {string} -- in [random, ordered], random: sample from specified distribution (init with uniform)

        Keyword Arguments:
            nr_samples {int} -- [description] (default: {-1}) -1: all submaps
        """
        self.probs = None
        self.nr_submaps = nr_submaps
        if nr_samples < 0:
            self.nr_samples = nr_submaps
        else:
            self.nr_samples = nr_samples

        self.sample_fkt = getattr(self, sampling_method)
        self.p_func = torch.ones(self.nr_submaps, dtype=torch.float)
        self.dist = torch.distributions.Categorical(self.p_func)

    def setSampleProbs(self, probs):
        self.p_func = probs
        self.dist = torch.distributions.Categorical(self.p_func)

    def random(self):
        return (self.dist.sample() for _ in range(self.nr_samples))

    def ordered(self):
        return (i for i in torch.arange(self.nr_samples))

    def __iter__(self):
        return self.sample_fkt()

    def __len__(self):
        return self.nr_samples


class SubMapDataSet(Dataset):
    def __init__(self, data_dirs,
                 nr_submaps=0,
                 nr_points=10000,
                 cols=3,
                 on_the_fly=True,
                 init_ones=True,
                 feature_cols=[],
                 grid_size = 40):
        self.data_dirs = data_dirs
        self.nr_submaps = nr_submaps
        self.nr_points = nr_points
        self.cols = cols
        self.init_ones = init_ones
        self.fc = feature_cols
        self.submaps = createSubmaps(
            data_dirs, nr_submaps=self.nr_submaps, cols=cols, on_the_fly=on_the_fly,grid_size=grid_size)  # list of submaps

    def __getitem__(self, index):
        out_dict = {'idx': index}
        self.submaps[index].initialize()
        if self.cols <= 3:
            out_dict['points'] = self.submaps[index].getRandPoints(
                self.nr_points, seed=index)
            out_dict['map'] = self.submaps[index].getPoints()
            if self.init_ones:
                out_dict['features'] = np.ones(
                    (out_dict['points'].shape[0], 1), dtype='float32')
        else:
            points = self.submaps[index].getRandPoints(
                self.nr_points, seed=index)
            out_dict['points'] = points[:, :3]
            out_dict['points_attributes'] = points[:, 3:]
            map_ = self.submaps[index].getPoints()
            out_dict['map'] = map_[:, :3]
            out_dict['map_attributes'] = map_[:, 3:]
            if self.init_ones:
                out_dict['features'] = np.hstack(
                    (np.ones((points.shape[0], 1), dtype='float32'), out_dict['points_attributes'][:, self.fc]))
            else:
                out_dict['features'] = out_dict['points_attributes'][:, self.fc]
        out_dict['features_original'] = out_dict['features']
        out_dict['scale'] = self.submaps[index].getScale()
        return out_dict

    def __len__(self):
        return len(self.submaps)


class SubMap():
    def __init__(self, file, grid_size=None, on_the_fly=False, file_cols=3):
        self.file = file
        # self.embedding = torch.zeros((embedding_dim))
        self.seq = file.split('/')[-2]
        self.id = file.split('/')[-1]
        self.cols = file_cols
        self.points = pcu.loadCloudFromBinary(
            self.file, cols=file_cols) if not on_the_fly else None
        self.normalizer = None
        self.grid_size = grid_size

        self.initialized = False
        if not on_the_fly:
            self.initialize()
            self.points = np.hstack((self.normalizer.normalize(
                self.points[:, :3]), self.points[:, 3:]))

    def initialize(self):
        if not self.initialized:
            self.normalizer = Normalizer(
                data=self.getPoints(normalize=False)[:, :3], dif=self.grid_size)
            self.initialized = True

    def normRange(self):
        return self.normalizer.normRange()

    def getScale(self):
        return self.normalizer.getScale()

    def __len__(self):
        # points = pcu.loadCloudFromBinary(self.file)
        return self.getPoints().shape[0]

    def getSample(self, idx):
        # points = pcu.loadCloudFromBinary(self.file)
        return self.getPoints()[idx, :]

    def getPoints(self, normalize=True):
        # points = pcu.loadCloudFromBinary(self.file)
        if self.points is None:
            points = pcu.loadCloudFromBinary(self.file, cols=self.cols)
            if normalize:
                points = np.hstack(
                    (self.normalizer.normalize(points[:, :3]), points[:, 3:]))
            return points
        else:
            return self.points

    def getRandPoints(self, nr_points, seed=0):
        points = self.getPoints()
        act_nr_pts = points.shape[0]
        subm_idx = np.arange(act_nr_pts)
        np.random.seed(seed)
        np.random.shuffle(subm_idx)
        # print('shuffled idx',subm_idx)
        subm_idx = subm_idx[0:min(act_nr_pts, nr_points)]
        return points[subm_idx, :]


def createSubmaps(folders, nr_submaps=0, cols=3, on_the_fly=False,grid_size=40):
    submap_files = []
    for folder in sorted(folders):
        submap_files += sorted(glob.glob(folder+'*bin'))
    if int(nr_submaps) != 0:
        # print('taking n maps:',nr_submaps)
        n = min((len(submap_files), nr_submaps))
        submap_files = submap_files[:n]
    submaps = [SubMap(f, file_cols=cols,
                      on_the_fly=on_the_fly,grid_size=grid_size) for f in submap_files]
    # print(submaps[0].seq, submaps[0].id,len(submaps[0]))
    return submaps


class Normalizer():
    def __init__(self, data, dif=None):
        self.min = np.amin(data, axis=0, keepdims=True)
        self.max = np.amax(data, axis=0, keepdims=True)
        if dif is None:
            self.dif = self.max-self.min
        else:
            self.dif = dif

    def getScale(self):
        return self.dif

    def normalize(self, points):
        return (points - self.min)/self.dif

    def recover(self, norm_points):
        return (norm_points * self.dif)+self.min


if __name__ == "__main__":
    parser = argparse.ArgumentParser("./submap_handler.py")
    parser.add_argument(
        '--cfg', '-c',
        type=str,
        required=False,
        default='config/arch/sample_net.yaml',
        help='Architecture yaml cfg file. See /config/arch for sample. No default!',
    )

    FLAGS, unparsed = parser.parse_known_args()
    config = yaml.safe_load(open(FLAGS.cfg, 'r'))
    s = time.time()

    print(20*'#', 'Torch data loader', 20*'#')
    s = time.time()
    submap_parser = SubMapParser(config)
    print('submap init time', time.time()-s)
    for epoch in range(2):
        for it, out_dict in enumerate(submap_parser.getTrainSet()):
            print(it, 'sm:', out_dict)
            print('points shape', out_dict['points'].shape)
            print('map shape', out_dict['map'].shape)
            submap_parser.setTrainProbabilities(
                torch.tensor([1.0, 0, 0, 0, 0]))
