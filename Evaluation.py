import Mesh
import Projection as pj
import procrustes as p
import numpy as np
from copy import deepcopy
import cv2
from chumpy.utils import row, col
import chumpy as ch
import vtk
from opendr.everything import *
from opendr.renderer import BoundaryRenderer
from opendr.renderer import ColoredRenderer
from opendr.renderer import DepthRenderer
from opendr.camera import ProjectPoints
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.misc
import scipy.spatial
import scipy.optimize as op
import lmfit
from lmfit import Parameters
import time
import random
from scipy.spatial.transform import Rotation as R
import cv2
from config import cfg
import os
import argparse


#visualization for checking
def draw_pixel(contour, points1, points2 = None):
    img = deepcopy(contour)
    for p in points1:
        center = (p[1], p[0])
        color = (255, 0, 0)
        cv2.circle(img, center, 1, color, thickness=-1)
    if points2 is not None:
        for p in points2:
            center = (p[1], p[0])
            color = (0, 255, 0)
            cv2.circle(img, center, 1, color, thickness=-1)
    #cv2.imshow('pair points', img)
    plt.imshow(img)
    return 0


def randome_deviation(rseed, rd_range, td_range):
    random.seed(rseed)
    rx = random.random()
    ry = random.uniform(0, 1-rx)
    rz = random.uniform(0, 1-rx-ry)
    randr = np.array([rx, ry, rz]) * np.pi/rd_range
    tx = random.random()
    ty = random.uniform(0, 1-tx)
    tz = random.uniform(0, 1-tx-ty)
    randt = np.array([tx, ty, tz]) * td_range
    return randr, randt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--t_model", help="tooth model file folder")
    args = parser.parse_args()

    ### default value ####
    teeth_file_folder = '/home/jiaming/MultiviewFitting/data/upper_segmented/HBF_12681/before'
    # teeth_file_folder = 'data/from GuMin/seg/model_0'
    # moved_mesh_folder = '/home/jiaming/MultiviewFitting/data/observation/12681/movedRow_real1.obj'


    #parsing the input arguments
    if args.t_model is not None:
        teeth_file_folder = args.t_model

    print(teeth_file_folder)

    teeth_gt_mesh = Mesh.TeethRowMesh(teeth_file_folder, False)
    teeth_row_mesh = Mesh.TeethRowMesh(teeth_file_folder, False)
    row_mesh = teeth_row_mesh.row_mesh

    # moved_mesh = Mesh.TeethRowMesh(moved_mesh_folder, True)
    # t0 = ch.asarray(teeth_row_mesh.positions_in_row)

    numTooth = len(teeth_row_mesh.mesh_list)
    Ri_list = [ch.zeros(3) for i in range(numTooth)]
    ti_list = [ch.zeros(3) for i in range(numTooth)]

    R_row = ch.array([0, 0, 0.06])
    t_row = ch.array([0.09, 0, 0])
    teeth_row_mesh.rotate(R_row)
    teeth_row_mesh.translate(t_row)

    # #random deviation
    # for i in range(numTooth):
    #     tmprd, tmptd = randome_deviation(i*(time.time()), 18, 0.04)
    #     teeth_row_mesh.rotate(tmprd, i)
    #     teeth_row_mesh.translate(tmptd, i)

    Vi_list = [ch.array(teeth_row_mesh.mesh_list[i].v) for i in range(numTooth)]
    Vi_offset = [ch.mean(Vi_list[i], axis=0) for i in range(numTooth)]
    # print(Vi_offset)

    Vi_center = [(Vi_list[i] - Vi_offset[i]) for i in range(numTooth)]

    # V_row = t_row + ch.vstack([ti_list[i] + Vi_offset[i] + Vi_center[i].dot(Rodrigues(Ri_list[i])) for i in range(numTooth)]).dot(Rodrigues(R_row))
    V_row = ch.vstack([Vi_list[i] for i in range(numTooth)])

    # Mesh.save_to_obj('result/V_row_gt.obj', V_row.r, row_mesh.f)

    t_c = len(teeth_row_mesh.mesh_list[0].v)
    vert_t = [[] for i in range(numTooth)]
    mvert_t = [[] for i in range(numTooth)]
    idx = 0
    for i in range(V_row.shape[0]):
        if i < t_c:
            vert_t[idx].append(V_row[i].r)
        else:
            idx += 1
            vert_t[idx].append(V_row[i].r)
            t_c += len(teeth_row_mesh.mesh_list[idx].v)
        mvert_t[idx].append(teeth_gt_mesh.row_mesh.v[i])

    neighbor = [[] for i in range(numTooth)]
    for i in range(numTooth):
        if i == 0:
            neighbor[i].append(numTooth/2)
            neighbor[i].append(i+1)
            continue
        if i == (numTooth/2-1):
            neighbor[i].append(i-1)
            continue
        if i == numTooth/2:
            neighbor[i].append(0)
            neighbor[i].append(i+1)
            continue
        if i == numTooth-1:
            neighbor[i].append(i-1)
            continue
        neighbor[i].append(i+1)
        neighbor[i].append(i-1)


    for i in range(numTooth):
        d, Z, tform = p.procrustes(np.array(vert_t[i]), np.array(mvert_t[i]), scaling=False)
        print ("--tooth_{}".format(i), 'translation error:', tform['translation'], 'rotation error:', p.rotationMatrixToEulerAngles(tform['rotation']))
        tr = ch.array(p.rotationMatrixToEulerAngles(tform['rotation']))
        tt = ch.array(tform['translation'])
        for k in neighbor[i]:
            # mean = np.mean(mvert_t[k], axis=0, keepdims=True)
            V_n = tt + ch.vstack(mvert_t[k]).dot(Rodrigues(tr))
            d, Z, newtform = p.procrustes(np.array(vert_t[k]), np.array(V_n), scaling=False)
            print ("--tooth_{} relative to tooth {}".format(k, i), 'translation error:', newtform['translation'], 'rotation error:',
                    p.rotationMatrixToEulerAngles(newtform['rotation']))

        # new_mvert_t = mvert_t / 2.0
        # new_mvert_t *= moved_mesh.max_v
        # new_vert_t = vert_t / 2.0
        # new_vert_t *= teeth_row_mesh.max_v
        # d1, Z1, tform1 = p.procrustes(np.array(new_mvert_t[i]), np.array(new_vert_t[i]), scaling=False)
        # print ("--tooth_{}'s errors are: ".format(i), d, tform['translation'], p.rotationMatrixToEulerAngles(tform['rotation']), 'scale_reverse:', tform1['translation'], p.rotationMatrixToEulerAngles(tform1['rotation']))
