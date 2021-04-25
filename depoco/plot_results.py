import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
from ruamel import yaml
import depoco.utils.point_cloud_utils as pcu

def plotResults(files, x_key, y_key, ax, draw_line=False, label=None, set_lim=True):
    x = []
    y = []
    for f in files:
        eval_dict = pcu.load_obj(f)
        if((x_key in eval_dict.keys()) & (y_key in eval_dict.keys())):
            for v in eval_dict.values():
                v = np.array(v)
            if not draw_line:
                ax.plot(np.mean(eval_dict[x_key]),
                        np.mean(eval_dict[y_key]), '.')
                ax.text(np.mean(eval_dict[x_key]), np.mean(
                    eval_dict[y_key]), f.split('/')[-1][:-4])

            x.append(np.mean(eval_dict[x_key]))
            y.append(np.mean(eval_dict[y_key]))

    if draw_line:
        line, = ax.plot(x, y, '-x', label=label)
        # line.set_label(label)

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)

    if set_lim:
        ax.set_xlim(0,None)
        ax.set_ylim(0,None)
    # ax.grid()


def genPlots(files, f, ax, draw_line=False, label=None, x_key='memory'):
    # print('shape',ax[0,0])
    plotResults(files, x_key=x_key, y_key='chamfer_dist_abs',
                ax=ax[0], draw_line=draw_line, label=label)
    plotResults(files, x_key=x_key, y_key='chamfer_dist_plane',
                ax=ax[1], draw_line=draw_line, label=label)
    plotResults(files, x_key=x_key, y_key='iou',
                ax=ax[2], draw_line=draw_line, label=label)


if __name__ == "__main__":
    ####### radius fct ##############
    path = 'experiments/results/kitti/'
    files = sorted(glob.glob(path+'*.pkl'))

    f, ax = plt.subplots(1, 3)
    f.suptitle('Radius FPS')
    genPlots(files, f, ax, draw_line=True, label='our', x_key='bpp')
    for a in ax:
        a.grid()
        # a.set_ylim([0,None])
        a.legend()
    plt.show()
