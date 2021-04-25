import numpy as np
import open3d as o3d
import torch
import os
import time
import octree_handler
import random
import pickle


def path(text):
    '''
        adds "/" to the end if needed
    '''
    if text is "":
        return ""
    if(text.endswith('/')):
        return text
    else:
        return text+"/"


def isEveryNPercent(current_it: int, max_it: int, percent: float = 10):
    curr_percent = current_it/max_it*100
    n = int(curr_percent/percent)

    prev_percent = (current_it-1)/max_it*100
    return ((curr_percent >= n * percent) & (prev_percent < n * percent)) or (curr_percent >= 100)


def visPointCloud(pcd, colors=None, normals=None, downsample=None, show_normals=False):
    pcd_o3 = o3d.geometry.PointCloud()
    pcd_o3.points = o3d.utility.Vector3dVector(pcd[:, 0:3])
    if colors is not None:
        pcd_o3.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd_o3.normals = o3d.utility.Vector3dVector(normals)
    if downsample is not None:
        pcd_o3 = pcd_o3.voxel_down_sample(downsample)
    o3d.visualization.draw_geometries([pcd_o3], point_show_normal=show_normals)


def visPointClouds(pcd_list, colors_list=None):
    pcd_o3_list = []
    for i, pcd in enumerate(pcd_list):
        pcd_o3 = o3d.geometry.PointCloud()
        pcd_o3.points = o3d.utility.Vector3dVector(pcd[:, 0:3])
        if colors_list is not None:
            if len(colors_list) > i:
                colors = colors_list[i]
                if colors is not None:
                    pcd_o3.colors = o3d.utility.Vector3dVector(colors)
        else:
            print('have colors')

        pcd_o3_list += [pcd_o3]
    print(f'pcd_o3_list len= {len(pcd_o3_list)}')
    o3d.visualization.draw_geometries(pcd_o3_list)

# def visAll(input,)

def randomSample(nr_samples, nr_points, seed =0):
    """Samples nr_samples indices. All valures in range of nr_points, no duplication

    Args:
        nr_samples ([type]): [description]
        nr_points ([type]): [description]
        seed (int, optional): [description]. Defaults to 0.
    """
    subm_idx = np.arange(nr_points)
    np.random.seed(seed)
    np.random.shuffle(subm_idx)
    # print('shuffled idx',subm_idx)
    return subm_idx[0:min(nr_points, nr_samples)]

def visVectorField(start, end, ref=None, colors=None):
    nr_p = start.shape[0]
    pcd_o3 = o3d.geometry.PointCloud()
    pcd_o3.points = o3d.utility.Vector3dVector(end[:, 0:3])
    # o3d.visualization.draw_geometries([pcd_o3])
    lines = np.concatenate((np.reshape(np.arange(nr_p), (-1, 1)),
                            np.reshape(np.arange(nr_p)+nr_p, (-1, 1))), axis=1)
    points = np.concatenate((start, end), axis=0)
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                    lines=o3d.utility.Vector2iVector(lines))
    if ref is None:
        o3d.visualization.draw_geometries([line_set, pcd_o3])
    else:
        ref_o3 = o3d.geometry.PointCloud()
        ref_o3.points = o3d.utility.Vector3dVector(ref[:, 0:3])
        if colors is not None:
            ref_o3.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([line_set, pcd_o3, ref_o3])


def renderVectorField(start, end, ref=None, colors=None, file_path='test.png'):
    nr_p = start.shape[0]
    pcd_o3 = o3d.geometry.PointCloud()
    pcd_o3.points = o3d.utility.Vector3dVector(end[:, 0:3])
    # o3d.visualization.draw_geometries([pcd_o3])
    lines = np.concatenate((np.reshape(np.arange(nr_p), (-1, 1)),
                            np.reshape(np.arange(nr_p)+nr_p, (-1, 1))), axis=1)
    points = np.concatenate((start, end), axis=0)
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                    lines=o3d.utility.Vector2iVector(lines))
    if ref is None:
        renderO3d([line_set, pcd_o3], file_path=file_path)
    else:
        ref_o3 = o3d.geometry.PointCloud()
        ref_o3.points = o3d.utility.Vector3dVector(ref[:, 0:3])
        if colors is not None:
            ref_o3.colors = o3d.utility.Vector3dVector(colors)
        renderO3d([line_set, pcd_o3, ref_o3], file_path=file_path)


def renderO3d(o3d_list, file_path='test.png'):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for o3d_g in o3d_list:
        vis.add_geometry(o3d_g)
    # vis.add_geometry(o3d_list)
    # vis.update_geometry()

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(file_path)
    # vis.run()
    vis.destroy_window()


def renderCloud(pcd, colors, file_path='test.png'):
    pcd_o3 = o3d.geometry.PointCloud()
    pcd_o3.points = o3d.utility.Vector3dVector(pcd[:, 0:3])
    if colors is not None:
        pcd_o3.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_o3)
    # vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(file_path)
    # vis.run()
    vis.destroy_window()


def saveCloud2Binary(cld, file, out_path=None):
    if out_path is None:
        out_path = ''
    else:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    f = open(out_path+file, "wb")
    f.write(cld.astype('float32').T.tobytes())
    f.close()


def loadCloudFromBinary(file, cols=3):
    f = open(file, "rb")
    binary_data = f.read()
    f.close()
    temp = np.frombuffer(
        binary_data, dtype='float32', count=-1)
    data = np.reshape(temp, (cols, int(temp.size/cols)))
    return data.T


def colorizeConv(in_pcl:np.ndarray, out_pcl:np.ndarray, kernel_radius, max_nr_neighbors,kernel_pos=None, kernel_points=None):
    octree = octree_handler.Octree()
    octree.setInput(in_pcl)
    out_index = random.randrange(out_pcl.shape[0])
    in_index = octree.radiusSearchPoints(
        out_pcl[out_index:out_index+1, :], max_nr_neighbors, kernel_radius)
    in_index = in_index[in_index < in_pcl.shape[0]]
    in_clr = np.zeros_like(in_pcl)
    in_clr[in_index, :] = np.ones_like(
        in_pcl[in_index, :])*np.array([1, 0, 0])

    out_clr = np.zeros_like(out_pcl)
    out_clr[out_index, :] = np.array([1, 0, 0])
    
    if kernel_pos is not None:
        out_pt = out_pcl[out_index:out_index+1,:]
        k_in = kernel_pos[out_index,:,:]+out_pt
        k_clr = np.ones_like(k_in)*np.array([0, 1, 0])
        print('kernel_i coords ',kernel_pos[out_index,:,:])
        print('kernel_i coords ',kernel_points)
        print(f'in {in_pcl.shape},out {out_pcl.shape},kernel_def {kernel_pos.shape},kernel_def_i {k_in.shape},kernel {kernel_points.shape}',)
        in_pcl = np.vstack((in_pcl,k_in))
        in_clr = np.vstack((in_clr,k_clr))

    if kernel_points is not None:
        out_pt = out_pcl[out_index:out_index+1,:]
        k_in = kernel_points+out_pt
        k_clr = np.ones_like(k_in)*np.array([0, 0, 1])
        in_pcl = np.vstack((in_pcl,k_in))
        in_clr = np.vstack((in_clr,k_clr))
    # visPointClouds([in_pcl,out_pcl+1],[in_clr,out_clr])
    # if kernel_pos is not None:
    

    return (in_pcl, out_pcl), (in_clr, out_clr)

# def colorizeDeformedKP(in_pcl, out_pcl, kernel_radius, max_nr_neighbors,kern_pcl=None):
#     octree = octree_handler.Octree()
#     octree.setInput(in_pcl)

def visualizeConv(in_out_pts, in_out_clr):
    """[summary]

    Arguments:
        in_out_pts [list] -- [(in_pcl1,out_pcl1),(in_pcl2,out_pcl2),...] pointclouds 
        in_out_clr [list] -- [(in_clr1,out_clr1),(in_clr2,out_clr2),...] colors
    """
    extend = 1.2
    pts = []
    clrs = []
    n = len(in_out_pts)
    for i in range(n):
        in_pcl, out_pcl = in_out_pts[i]
        row = np.array([1, 0, 0])*extend * i
        col = np.array([0, -1, 0])*extend
        pts.append(in_pcl+row)
        pts.append(out_pcl+row+col)
        clrs.append(in_out_clr[i][0])
        clrs.append(in_out_clr[i][1])
    visPointClouds(pts, clrs)

# https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def findList(inp_list, value):
    for i, item in enumerate(inp_list):
        # print( item,value)
        if item == value:
            return i


if __name__ == "__main__":
    a = np.random.rand(12, 3)
    # saveCloud2Binary(a,'test.bin')
    # b = loadCloudFromBinary('test.bin')
    # print(a-b)
    # cld = loadCloudFromBinary('/media/lwiesmann/WiesmannIPB/data/data_kitti/dataset/submaps/04/000004.bin')
    # visPointCloud(cld)
    # start = np.array([[0, 0, 0],
    #                   [0, 1, 0],
    #                   [0, 0, 1]])

    # end = np.array([[0, 0, 0],
    #                 [1, 0, 0],
    #                 [0, 0, 1]])+2
    # visVectorField(start, end)
    # renderCloud(a)
