
import numpy as np
import octree_handler
import time

if __name__ == "__main__":
    octree = octree_handler.Octree()
    points = np.array([[0, -5, 0],  # 0
                       [0, -4, 0],  # 1
                       [0, -3, 0],  # 2
                       [0, -2, 0],  # 3
                       [0, -1, 0],  # 4
                       [0, 0, 0],  # 5
                       [1, 0, 0],  # 6
                       [2, 0, 0],  # 7
                       [3, 0, 0],  # 8
                       [4, 0, 0],  # 9
                       [5, 0, 0],  # 10
                       #    [100, 100, 100],  # 11: not valid point
                       ], dtype='float32')
    print(points)
    octree.setInput(points)
    b = octree.radiusSearch(5, 3.1)
    print(b)
    print(20*"-", "search all", 20*"-")
    ind = octree.radiusSearchAll(20, 2.1)
    print(ind)
    print(ind.shape, ind.dtype)

    ##############################
    ###### lot of points #########
    ##############################
    nr_p = int(1e4)
    print(20*"-", f"#points: {nr_p}", 20*"-")
    points = np.random.normal(scale=10.0, size=(nr_p, 3))
    t_init = time.time()
    octree.setInput(points)
    t_init = time.time()-t_init
    t_search = time.time()
    ind = octree.radiusSearchAll(10, 0.1)
    t_search = time.time() - t_search
    t_overall = t_init+t_search
    print(f'time: init {t_init}s, search {t_search}s, overall {t_overall}s')
    print(ind.shape)
