import torch
import chamfer3D.dist_chamfer_3D

import depoco.evaluation.occupancy_grid as occupancy_grid
import numpy as np
from collections import defaultdict
import torch.nn as nn


class Evaluator():
    def __init__(self, config):
        self.config = config
        self.cham_loss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
        self.running_loss = 0.0
        self.n = 0
        self.eval_results = defaultdict(list)
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def chamferDist(self, gt_points: torch.tensor, source_points: torch.tensor):
        """computes the chamfer distance between 2 point clouds

        Arguments:
            gt_points {torch.tensor} -- [description]
            source_points {torch.tensor} -- [description]

        Returns:
            [type] -- [description]
        """
        gt_points = gt_points.cuda().detach()
        source_points = source_points.cuda().detach()
        d_gt2source, d_source2gt, idx3, idx4 = self.cham_loss(
            gt_points.unsqueeze(0), source_points.unsqueeze(0))
        # mean(squared_d(gt->source)) + mean(squared_d(source->gt))
        loss = (d_gt2source.mean() + d_source2gt.mean())  # /2 FIXME:
        self.running_loss += loss.cpu().item()
        self.n += 1
        return loss

    def evaluate(self, gt_points: torch.tensor, source_points: torch.tensor, gt_normals=None):
        """computes the chamfer distance between 2 point clouds

        Arguments:
            gt_points {torch.tensor} -- [description]
            source_points {torch.tensor} -- [description]

        Returns:
            [dict] -- [description]
        """
        ##### Computing Chamfer Distances ######
        gt_points = gt_points.cuda().detach()
        source_points = source_points.cuda().detach()
        d_gt2source, d_source2gt, idx3, idx4 = self.cham_loss(
            gt_points.unsqueeze(0), source_points.unsqueeze(0))
        idx3 = idx3.long().squeeze()
        idx4 = idx4.long().squeeze()
        # mean(squared_d(gt->source)) + mean(squared_d(source->gt))
        chamfer_dist = (d_gt2source.mean() + d_source2gt.mean())/2
        chamfer_dist_abs = (d_gt2source.sqrt().mean() +
                            d_source2gt.sqrt().mean())/2
        out_dict = {}
        out_dict['chamfer_dist'] = chamfer_dist.cpu().item()
        self.eval_results['chamfer_dist'].append(out_dict['chamfer_dist'])
        out_dict['chamfer_dist_abs'] = chamfer_dist_abs.cpu().item()
        self.eval_results['chamfer_dist_abs'].append(
            out_dict['chamfer_dist_abs'])


        ############ PSNR ##############
        if gt_normals is not None:  # Computing PSNR if we have normals
            gt_normals = gt_normals.cuda().detach()
            d_plane_gt2source = torch.sum(
                (gt_points - source_points[idx3, :])*gt_normals, dim=1)
            d_plane_source2gt = torch.sum(
                (source_points - gt_points[idx4, :])*gt_normals[idx4, :], dim=1)
            chamfer_plane = (d_plane_gt2source.abs().mean() +
                             d_plane_source2gt.abs().mean())/2
            out_dict['chamfer_dist_plane'] = chamfer_plane.cpu().item()
            self.eval_results['chamfer_dist_plane'].append(
                out_dict['chamfer_dist_plane'])

        ###### IOU #######
        gt_points_np = gt_points.cpu().numpy()
        source_points_np = source_points.cpu().numpy()

        # print('gt_points shape',g)
        center = (np.max(gt_points_np, axis=0, keepdims=True) +
                  np.min(gt_points_np, axis=0, keepdims=True))/2
        resolution = np.array(
            [self.config['evaluation']['iou_grid']['resolution']])
        size_meter = np.array([self.config['grid']['size']])
        gt_grid = occupancy_grid.OccupancyGrid(
            center=center, resolution=resolution, size_meter=size_meter)
        gt_grid.addPoints(gt_points_np)
        source_grid = occupancy_grid.OccupancyGrid(
            center=center, resolution=resolution, size_meter=size_meter)
        source_grid.addPoints(source_points_np)

        out_dict['iou'] = occupancy_grid.gridIOU(
            gt_grid.grid, source_grid.grid)
        self.eval_results['iou'].append(out_dict['iou'])

        return out_dict

    def getRunningLoss(self):
        """returns the running loss:  loss/n
            sets the loss back to 0

        Returns:
            [int] -- average chamfer distance
        """
        if self.n == 0:
            return None
        loss = self.running_loss / self.n
        self.running_loss = 0.0
        self.n = 0
        return loss
