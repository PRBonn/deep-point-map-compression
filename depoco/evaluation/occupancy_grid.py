import numpy as np


class OccupancyGrid():
    def __init__(self, center: np.array, resolution: np.array, size_meter: np.array):
        self.center = center  # 1x3
        self.resolution = resolution
        self.size_meter = size_meter
        self.size = np.ceil(size_meter / resolution)  # rows x cols x layer
        self.min_corner = self.center - self.size*self.resolution/2

        self.grid = np.zeros(np.squeeze(self.size.astype('int')), dtype='bool')

    def addPoints(self, points: np.array):
        points_l = np.floor((points - self.min_corner)/self.resolution).astype('int')
        valids = np.all((points_l >= 0) & (points_l < self.size),axis=1)
        points_l = points_l[valids,:]
        self.grid[points_l[:,0],points_l[:,1],points_l[:,2]]= True

def gridIOU(gt_grid:np.array, source_grid:np.array):
    return np.sum((gt_grid & source_grid))/np.sum((gt_grid | source_grid))

if __name__ == "__main__":
    occ_grid = OccupancyGrid(center=np.zeros((1,3)),resolution=np.full((1,3),2),size_meter=np.full((1,3),6))
    print('occ grid \n',occ_grid.grid)
    points = np.array([
                    # [0,0,0],
                    [0.5,0.5,0.5],
                    [0.7,0.7,0.7],
                    [0.7,0.7,0.7],
                    [2.5,2.5,2.5],
                    [-2.5,2.5,2.5],
                    [100,100,100]
                    ])
    occ_grid.addPoints(points)
    print('occ grid \n',occ_grid.grid)

    occ_grid2 = OccupancyGrid(center=np.zeros((1,3)),resolution=np.full((1,3),2),size_meter=np.full((1,3),6))
    points = np.array([
                    # [0,0,0],
                    [0.5,0.5,0.5],
                    [0.7,0.7,0.7],
                    [0.7,0.7,0.7],
                    [2.5,-2.5,2.5],
                    [-2.5,2.5,2.5],
                    [100,100,100]
                    ])
    occ_grid2.addPoints(points)
    iou = gridIOU(occ_grid.grid,occ_grid2.grid)
    print('occ grid2 \n',occ_grid2.grid)
    print('iou',iou)
    ### Big grid:
    big_grid= OccupancyGrid(center=np.zeros((1,3)),resolution=np.array((0.2,0.2,0.1)),size_meter=np.array((40,40,15.0)))
    print(big_grid.grid.shape)