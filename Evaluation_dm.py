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
import icp

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
    parser.add_argument("--model_u", help="upper tooth model file folder")
    parser.add_argument("--model_l", help="lower tooth model file folder")
    parser.add_argument("--gt_model_u", help="gt upper tooth model file folder")
    parser.add_argument("--gt_model_l", help="gt lower tooth model file folder")
    args = parser.parse_args()

    ### default value ####
    teeth_file_folder = r'/home/sky7hate/Project/MultiviewFitting/data/seg/model1_u'
    teeth_file_folder_low = r'/home/sky7hate/Project/MultiviewFitting/data/seg/model1_l'
    gt_teeth_file_folder = r'/home/sky7hate/Project/MultiviewFitting/data/seg/model0_u'
    gt_teeth_file_folder_low = r'/home/sky7hate/Project/MultiviewFitting/data/seg/model0_l'


    #parsing the input arguments
    if args.model_u is not None:
        teeth_file_folder = args.model_u
    if args.model_l is not None:
        teeth_file_folder_low = args.model_l
    if args.gt_model_u is not None:
        gt_teeth_file_folder = args.gt_model_u
    if args.gt_model_l is not None:
        gt_teeth_file_folder_low = args.gt_model_l

    print(teeth_file_folder)

    teeth_fit_mesh_u = Mesh.TeethRowMesh(teeth_file_folder, False)
    teeth_gt_mesh_u = Mesh.TeethRowMesh(gt_teeth_file_folder, False)

    teeth_fit_mesh_l = Mesh.TeethRowMesh(teeth_file_folder_low, False)
    teeth_gt_mesh_l = Mesh.TeethRowMesh(gt_teeth_file_folder_low, False)

    # moved_mesh = Mesh.TeethRowMesh(moved_mesh_folder, True)
    # t0 = ch.asarray(teeth_row_mesh.positions_in_row)

    numTooth_u = len(teeth_fit_mesh_u.mesh_list)
    numTooth_l = len(teeth_fit_mesh_l.mesh_list)

    #random deviation
    # for i in range(numTooth):
    #     tmprd, tmptd = randome_deviation(i*(time.time()), 36, 0.009)
    #     print tmptd, tmprd
    #     teeth_row_mesh.rotate(tmprd, i)
    #     teeth_row_mesh.translate(tmptd, i)

    Vi_list_u = [ch.array(teeth_fit_mesh_u.mesh_list[i].v) for i in range(numTooth_u)]
    Vi_offset_u = [ch.mean(Vi_list_u[i], axis=0) for i in range(numTooth_u)]
    gt_Vi_list_u = [ch.array(teeth_gt_mesh_u.mesh_list[i].v) for i in range(numTooth_u)]
    # print(Vi_offset)

    Vi_list_l = [ch.array(teeth_fit_mesh_l.mesh_list[i].v) for i in range(numTooth_l)]
    Vi_offset_l = [ch.mean(Vi_list_l[i], axis=0) for i in range(numTooth_l)]
    gt_Vi_list_l = [ch.array(teeth_gt_mesh_l.mesh_list[i].v) for i in range(numTooth_l)]

    neighbor_u = [[] for i in range(numTooth_u)]
    for i in range(numTooth_u):
        if i == 0:
            neighbor_u[i].append(numTooth_u/2)
            neighbor_u[i].append(i+1)
            continue
        if i == (numTooth_u/2-1):
            neighbor_u[i].append(i-1)
            continue
        if i == numTooth_u/2:
            neighbor_u[i].append(0)
            neighbor_u[i].append(i+1)
            continue
        if i == numTooth_u-1:
            neighbor_u[i].append(i-1)
            continue
        neighbor_u[i].append(i+1)
        neighbor_u[i].append(i-1)

    print ("upper individual tooth:")
    for i in range(numTooth_u):
        T, dis, inum = icp.icp(np.array(Vi_list_u[i]), np.array(gt_Vi_list_u[i]))
        print ("--tooth_{}".format(i), 'translation error:', T[0:3,3].T, 'rotation error:', Rodrigues(T[0:3,0:3]).T)
        # print dis
        for k in neighbor_u[i]:
            V_n = T[0:3,3].T + ch.vstack(Vi_list_u[k]).dot(T[0:3,0:3])
            T_n, disn, inumn = icp.icp(np.array(V_n), np.array(gt_Vi_list_u[k]))
            print ("--tooth_{} relative to tooth {}".format(k, i), 'translation error:', T_n[0:3,3].T,
                   'rotation error:', Rodrigues(T_n[0:3,0:3]).T)

    neighbor_l = [[] for i in range(numTooth_l)]
    for i in range(numTooth_l):
        if i == 0:
            neighbor_l[i].append(numTooth_l / 2)
            neighbor_l[i].append(i + 1)
            continue
        if i == (numTooth_l / 2 - 1):
            neighbor_l[i].append(i - 1)
            continue
        if i == numTooth_l / 2:
            neighbor_l[i].append(0)
            neighbor_l[i].append(i + 1)
            continue
        if i == numTooth_l - 1:
            neighbor_l[i].append(i - 1)
            continue
        neighbor_l[i].append(i + 1)
        neighbor_l[i].append(i - 1)

    print ("lower individual tooth:")
    for i in range(numTooth_l):
        T, dis, inum = icp.icp(np.array(Vi_list_l[i]), np.array(gt_Vi_list_l[i]))
        print ("--tooth_{}".format(i), 'translation error:', T[0:3, 3].T, 'rotation error:', Rodrigues(T[0:3, 0:3]).T)
        # print dis
        for k in neighbor_l[i]:
            V_n = T[0:3, 3].T + ch.vstack(Vi_list_l[k]).dot(T[0:3, 0:3])
            T_n, disn, inumn = icp.icp(np.array(V_n), np.array(gt_Vi_list_l[k]))
            print ("--tooth_{} relative to tooth {}".format(k, i), 'translation error:', T_n[0:3, 3].T,
                   'rotation error:', Rodrigues(T_n[0:3, 0:3]).T)