import numpy as np


class VoxelGrid():
    def __init__(self, VOXEL, center, grid_size, voxel_size,point_dim=3):
        self.grid_size = grid_size
        self.center = center
        self.voxel_size = voxel_size

        self.grid_dim = np.ceil(grid_size / voxel_size).astype('int')
        self.num_voxel = int(np.prod(self.grid_dim))
        self.origin_offset = center - \
            np.ceil(grid_size / voxel_size) / 2 * self.voxel_size
        # print(self.num_voxel)
        self.grid = [VOXEL(point_dim) for i in range(self.num_voxel)]
        self.used_voxel = []

    def addPoint(self, point):
        idx = self.xyz2index(point)
        if(idx is not None):
            if (self.grid[idx].isEmpty()):
                self.used_voxel.append(idx)
            self.grid[idx].addPoint(point)

    def xyz2index(self, point):
        rcl = ((point - self.origin_offset)/self.voxel_size).astype("int")
        if np.any(rcl < 0) or np.any(rcl >= (self.grid_dim)):
            return None
        return rcl[0] + rcl[1] * self.grid_dim[0] + rcl[2] * self.grid_dim[0] * self.grid_dim[1]

    def cloud2indices(self, point_cld):
        '''
            computes the grid index of each point and a bool if they are inside the grid
                point_cld [nx3]
            return
            ------
                grid_idx [m,], m times 1d indices of the grid
                cloud_idx [m,] the indices of the points which are inside the grid
                | m: number valid points
        '''
        rcl = ((point_cld - self.origin_offset)/self.voxel_size).astype("int")
        # print('rcl',rcl)
        valid= np.argwhere( np.all(rcl >= 0, axis=1) & np.all(rcl < (self.grid_dim),axis=1)).reshape(-1)
        # print('valid',valid)
        idx = rcl[valid,0] + rcl[valid,1] * self.grid_dim[0] + rcl[valid,2] * self.grid_dim[0] * self.grid_dim[1]
        # print('idx',idx)

        return idx, valid

    def addPointCloud(self, point_cld):
        '''
            add all points [d dimensional] to the grid
                point_cld [nxd]  | first 3 cols need to be xyz
        '''
        grid_idx, cloud_idx = self.cloud2indices(point_cld[:, 0:3])
        for g_idx, c_idx in zip(grid_idx, cloud_idx):
            if (self.grid[g_idx].isEmpty()):
                self.used_voxel.append(g_idx)
            self.grid[g_idx].addPoint(point_cld[c_idx, :])

    def getPointCloud(self):
        '''
            returns the voxels which are not empty
        '''
        return np.asarray([self.grid[i].getValue() for i in self.used_voxel])

class AverageVoxel():
    def __init__(self,point_dim):
        self.point = np.zeros((point_dim))
        self.weight = 0
        # print('hi')

    def addPoint(self, point):
        self.point += point
        self.weight += 1

    def getValue(self):
        dim = self.point.shape[0]
        val = np.ones( (dim+1),dtype= np.float32 ) *self.weight
        val[0:dim]= self.point/self.weight
        return val

    def isEmpty(self):
        return self.weight == 0


class AverageGrid(VoxelGrid):
    def __init__(self, center, grid_size, voxel_size, point_dim=3):
        super().__init__(AverageVoxel, center, grid_size, voxel_size, point_dim=point_dim)
        # print(self.grid_dim)




if __name__ == "__main__":
    center = np.array([0.0, 0.0, 0.0])
    grid_size = np.array([10.0, 10.0, 10.0])
    voxel_grid = AverageGrid(center, grid_size, 5)

    p1 = np.array([2.0, -1.0, -1.0])
    p2 = np.array([2.0, 1.0, 1.0])
    p3 = np.array([6.0, 1.2, 1.0])
    voxel_grid.addPoint(p1)
    voxel_grid.addPoint(p2)
    voxel_grid.addPoint(p3)
    voxel_grid.addPoint(p3)
    print('v1 used voxel',voxel_grid.used_voxel)
    p = voxel_grid.getPointCloud()
    print('v1 points',p.shape,p)

    voxel_grid2 = AverageGrid(center, grid_size, 5)
    pcs = (p1[np.newaxis,:],p2[np.newaxis,:],p3[np.newaxis,:],p3[np.newaxis,:])
    cld = np.concatenate(pcs)
    print('cld',cld.shape)
    voxel_grid2.addPointCloud(cld)
    p2 = voxel_grid2.getPointCloud()
    print('v2 points',p2.shape,p2)
    print('diff',p-p2)

