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

def get_sample_pts(contour, colormap = None, t_id = None):
    #decrease the size
    # contour1 = scipy.misc.imresize(contour, 0.5)
    #Sample points
    t_sample_pts = []
    f_sample_pts = []
    # print contour1.size

    f_sample_pts = np.argwhere(contour[:, :, 0] > 0)
    # index = t_sample_pts.shape[0]
    # for i in range(index):
    #     if contour[t_sample_pts[i][0]*2, t_sample_pts[i][1]*2, 0] > 0:
    #         f_sample_pts.append([t_sample_pts[i][0]*2, t_sample_pts[i][1]*2])
    #     elif contour[t_sample_pts[i][0]*2, t_sample_pts[i][1]*2+1, 0] > 0:
    #         f_sample_pts.append([t_sample_pts[i][0]*2, t_sample_pts[i][1]*2+1])
    #     elif contour[t_sample_pts[i][0]*2+1, t_sample_pts[i][1]*2, 0] > 0:
    #         f_sample_pts.append([t_sample_pts[i][0]*2+1, t_sample_pts[i][1]*2])
    #     elif contour[t_sample_pts[i][0]*2+1, t_sample_pts[i][1]*2+1, 0] > 0:
    #         f_sample_pts.append([t_sample_pts[i][0]*2+1, t_sample_pts[i][1]*2+1])
    f_sample_pts = np.vstack(f_sample_pts)
    # print index, cc
    sp_pts = []
    if t_id is not None:
        # print colormap[f_sample_pts[:]]
        # sp_pts = f_sample_pts[np.rint(colormap[f_sample_pts[:]][0]*20)== t_id]
        for i in range(f_sample_pts.shape[0]):
            # if np.rint(colormap[f_sample_pts[i][0], f_sample_pts[i][1]][0]*20) == t_id:
            if check_around(colormap, t_id, f_sample_pts[i]):
                sp_pts.append(f_sample_pts[i])
        print len(sp_pts)
        return sp_pts
    return f_sample_pts

def check_around(colormap, t_id, checkpixel):
    for i in range(-1, 1, 1):
        for j in range(-1, 1, 1):
            if np.rint(colormap[checkpixel[0]+i, checkpixel[1]+j][0]*20) == t_id and colormap[checkpixel[0]+i, checkpixel[1]+j][1]>0:
                return True
    return False

def get_pair_pts(gt_contour, sp_pts, ins_pts, pair_id=None):
    #sample ground truth contour points
    gt_pts = []
    index = 0
    gt_pts = np.argwhere(gt_contour[:, :, 0] > 0.5)
    # draw_pixel(np.array(gt_contour), gt_pts)

    #build KDTree for gt points
    gtpts_tree = scipy.spatial.KDTree(gt_pts)
    #get sample points which finds back projection 3D verts
    s_sp_pts = []
    if pair_id is not None:
        for i in range(pair_id.shape[0]):
            s_sp_pts.append(sp_pts[pair_id[i]])
    else:
        s_sp_pts = sp_pts

    #pairing
    pair_pts = []
    new_ins_pts = []
    pair_res = gtpts_tree.query(s_sp_pts)

    #trimming: set a threshhold to filter outliers
    mean_dis = np.mean(pair_res, axis=1)[0]
    # print mean_dis
    threshhold = 2 * mean_dis

    for i in range(len(pair_res[1])):
        if pair_res[0][i] <= threshhold:
            pair_pts.append(gt_pts[pair_res[1][i]])
            new_ins_pts.append(ins_pts[i])
    print len(pair_pts), 'out of', len(pair_res[1])

    # draw_pixel(np.array(gt_contour), pair_pts, s_sp_pts)
    return pair_pts, new_ins_pts

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

#residual for only translation
def residual_t(pars, verts, Crt, Ct, pair_pts):
    tp = np.array([pars['tx'], pars['ty'], pars['tz']])
    t_verts = verts + tp
    tmpr = R.from_rotvec(Crt)
    r_mat = tmpr.as_dcm()
    t_vec = Ct.T
    cor_mtx = np.zeros((4, 4), dtype='float32')
    cor_mtx[0:3, 0:3] = r_mat
    cor_mtx[0:3, 3] = t_vec
    cor_mtx[3, 3] = 1

    residuals = []
    for i in range(t_verts.size / 3):
        tmp_v = cor_mtx.dot(np.array([t_verts[i][0], t_verts[i][1], t_verts[i][2], 1]).T)
        pix_u = 320*(tmp_v[0]/tmp_v[2]) + 320
        pix_v = 320*(tmp_v[1]/tmp_v[2]) + 240
        residuals.append(pix_v-pair_pts[i][0])
        residuals.append(pix_u-pair_pts[i][1])
    residuals = np.vstack(residuals)

    return residuals

#residual for translation and rotation for one view
def residual_rtt(pars, offset, verts, Crt, Ct, pair_pts):
    tp = np.array([pars['tx'], pars['ty'], pars['tz']])
    rtp = np.array([pars['rtx'], pars['rty'], pars['rtz']])
    t_verts = (verts-offset).dot(Rodrigues(rtp)) + offset + tp
    Crt = np.array(Crt.r)
    Ct = np.array(Ct.r)
    tmpr = R.from_rotvec(Crt)
    r_mat = tmpr.as_dcm()
    t_vec = Ct.T
    cor_mtx = np.zeros((4, 4), dtype='float32')
    cor_mtx[0:3, 0:3] = r_mat
    cor_mtx[0:3, 3] = t_vec
    cor_mtx[3, 3] = 1

    residuals = []
    for i in range(t_verts.size / 3):
        tmp_v = cor_mtx.dot(np.array([t_verts[i][0], t_verts[i][1], t_verts[i][2], 1]).T)
        pix_u = 320*(tmp_v[0]/tmp_v[2]) + 320
        pix_v = 320*(tmp_v[1]/tmp_v[2]) + 240
        residuals.append(pix_v-pair_pts[i][0])
        residuals.append(pix_u-pair_pts[i][1])
    residuals = np.vstack(residuals)

    return residuals

#residual for translation and rotation for all the views (individual tooth)
def residual_rtt_allview(pars, offset, verts1, verts2, verts3, verts4, Crt1, Ct1, Crt2, Ct2, Crt3, Ct3, Crt4, Ct4, pair_pts1, pair_pts2, pair_pts3, pair_pts4):
    residuals = []
    tp = np.array([pars['tx'], pars['ty'], pars['tz']])
    rtp = np.array([pars['rtx'], pars['rty'], pars['rtz']])

    if len(pair_pts1) > 0:
        t_verts = (verts1-offset).dot(Rodrigues(rtp)) + offset + tp
        Crt = np.array(Crt1.r)
        Ct = np.array(Ct1.r)
        tmpr = R.from_rotvec(Crt)
        r_mat = tmpr.as_dcm()
        t_vec = Ct.T
        cor_mtx = np.zeros((4, 4), dtype='float32')
        cor_mtx[0:3, 0:3] = r_mat
        cor_mtx[0:3, 3] = t_vec
        cor_mtx[3, 3] = 1
        for i in range(t_verts.size / 3):
            tmp_v = cor_mtx.dot(np.array([t_verts[i][0], t_verts[i][1], t_verts[i][2], 1]).T)
            pix_u = 320 * (tmp_v[0] / tmp_v[2]) + 320
            pix_v = 320 * (tmp_v[1] / tmp_v[2]) + 240
            residuals.append(pix_v - pair_pts1[i][0])
            residuals.append(pix_u - pair_pts1[i][1])

    if len(pair_pts2) > 0:
        t_verts = (verts2-offset).dot(Rodrigues(rtp)) + offset + tp
        Crt = np.array(Crt2.r)
        Ct = np.array(Ct2.r)
        tmpr = R.from_rotvec(Crt)
        r_mat = tmpr.as_dcm()
        t_vec = Ct.T
        cor_mtx = np.zeros((4, 4), dtype='float32')
        cor_mtx[0:3, 0:3] = r_mat
        cor_mtx[0:3, 3] = t_vec
        cor_mtx[3, 3] = 1
        for i in range(t_verts.size / 3):
            tmp_v = cor_mtx.dot(np.array([t_verts[i][0], t_verts[i][1], t_verts[i][2], 1]).T)
            pix_u = 320 * (tmp_v[0] / tmp_v[2]) + 320
            pix_v = 320 * (tmp_v[1] / tmp_v[2]) + 240
            residuals.append(pix_v - pair_pts2[i][0])
            residuals.append(pix_u - pair_pts2[i][1])

    if len(pair_pts3) > 0:
        t_verts = (verts3-offset).dot(Rodrigues(rtp)) + offset + tp
        Crt = np.array(Crt3.r)
        Ct = np.array(Ct3.r)
        tmpr = R.from_rotvec(Crt)
        r_mat = tmpr.as_dcm()
        t_vec = Ct.T
        cor_mtx = np.zeros((4, 4), dtype='float32')
        cor_mtx[0:3, 0:3] = r_mat
        cor_mtx[0:3, 3] = t_vec
        cor_mtx[3, 3] = 1
        for i in range(t_verts.size / 3):
            tmp_v = cor_mtx.dot(np.array([t_verts[i][0], t_verts[i][1], t_verts[i][2], 1]).T)
            pix_u = 320 * (tmp_v[0] / tmp_v[2]) + 320
            pix_v = 320 * (tmp_v[1] / tmp_v[2]) + 240
            residuals.append(pix_v - pair_pts3[i][0])
            residuals.append(pix_u - pair_pts3[i][1])

    if len(pair_pts4) > 0:
        t_verts = (verts4-offset).dot(Rodrigues(rtp)) + offset + tp
        Crt = np.array(Crt4.r)
        Ct = np.array(Ct4.r)
        tmpr = R.from_rotvec(Crt)
        r_mat = tmpr.as_dcm()
        t_vec = Ct.T
        cor_mtx = np.zeros((4, 4), dtype='float32')
        cor_mtx[0:3, 0:3] = r_mat
        cor_mtx[0:3, 3] = t_vec
        cor_mtx[3, 3] = 1
        for i in range(t_verts.size / 3):
            tmp_v = cor_mtx.dot(np.array([t_verts[i][0], t_verts[i][1], t_verts[i][2], 1]).T)
            pix_u = 320 * (tmp_v[0] / tmp_v[2]) + 320
            pix_v = 320 * (tmp_v[1] / tmp_v[2]) + 240
            residuals.append(pix_v - pair_pts4[i][0])
            residuals.append(pix_u - pair_pts4[i][1])

    residuals = np.vstack(residuals)

    return residuals

def residual_allpars_allview(pars, offset, verts1, verts2, verts3, verts4, Crt1, Ct1, Crt2, Ct2, Crt3, Ct3, Crt4, Ct4, pair_pts1, pair_pts2, pair_pts3, pair_pts4):
    residuals = []
    tp = []
    rtp = []
    tp.append(np.array([pars['tx1'], pars['ty1'], pars['tz1']]))
    rtp.append(np.array([pars['rtx1'], pars['rty1'], pars['rtz1']]))
    tp.append(np.array([pars['tx2'], pars['ty2'], pars['tz2']]))
    rtp.append(np.array([pars['rtx2'], pars['rty2'], pars['rtz2']]))
    tp.append(np.array([pars['tx3'], pars['ty3'], pars['tz3']]))
    rtp.append(np.array([pars['rtx3'], pars['rty3'], pars['rtz3']]))
    tp.append(np.array([pars['tx4'], pars['ty4'], pars['tz4']]))
    rtp.append(np.array([pars['rtx4'], pars['rty4'], pars['rtz4']]))
    tp.append(np.array([pars['tx5'], pars['ty5'], pars['tz5']]))
    rtp.append(np.array([pars['rtx5'], pars['rty5'], pars['rtz5']]))
    tp.append(np.array([pars['tx6'], pars['ty6'], pars['tz6']]))
    rtp.append(np.array([pars['rtx6'], pars['rty6'], pars['rtz6']]))
    tp.append(np.array([pars['tx7'], pars['ty7'], pars['tz7']]))
    rtp.append(np.array([pars['rtx7'], pars['rty7'], pars['rtz7']]))
    tp.append(np.array([pars['tx8'], pars['ty8'], pars['tz8']]))
    rtp.append(np.array([pars['rtx8'], pars['rty8'], pars['rtz8']]))
    tp.append(np.array([pars['tx9'], pars['ty9'], pars['tz9']]))
    rtp.append(np.array([pars['rtx9'], pars['rty9'], pars['rtz9']]))
    tp.append(np.array([pars['tx10'], pars['ty10'], pars['tz10']]))
    rtp.append(np.array([pars['rtx10'], pars['rty10'], pars['rtz10']]))
    tp.append(np.array([pars['tx11'], pars['ty11'], pars['tz11']]))
    rtp.append(np.array([pars['rtx11'], pars['rty11'], pars['rtz11']]))
    tp.append(np.array([pars['tx12'], pars['ty12'], pars['tz12']]))
    rtp.append(np.array([pars['rtx12'], pars['rty12'], pars['rtz12']]))
    crtp1 = np.array([pars['crtx1'], pars['crty1'], pars['crtz1']])
    ctp1 = np.array([pars['ctx1'], pars['cty1'], pars['ctz1']])
    crtp2 = np.array([pars['crtx2'], pars['crty2'], pars['crtz2']])
    ctp2 = np.array([pars['ctx2'], pars['cty2'], pars['ctz2']])
    crtp3 = np.array([pars['crtx3'], pars['crty3'], pars['crtz3']])
    ctp3 = np.array([pars['ctx3'], pars['cty3'], pars['ctz3']])
    crtp4 = np.array([pars['crtx4'], pars['crty4'], pars['crtz4']])
    ctp4 = np.array([pars['ctx4'], pars['cty4'], pars['ctz4']])

    for k in range(12):
        if len(pair_pts1[k]) > 0:
            t_verts = (verts1[k]-offset[k]).dot(Rodrigues(rtp[k])) + offset[k] + tp[k] #individual tooth movement

            #initial camera pose matrix
            Crt = np.array(Crt1.r)
            Ct = np.array(Ct1.r)
            tmpr = R.from_rotvec(Crt)
            r_mat = tmpr.as_dcm()
            t_vec = Ct.T
            cor_mtx = np.zeros((4, 4), dtype='float32')
            cor_mtx[0:3, 0:3] = r_mat
            cor_mtx[0:3, 3] = t_vec
            cor_mtx[3, 3] = 1

            #camera pose optimization pars matrix
            tmpcr = R.from_rotvec(crtp1)
            pr_mat = tmpcr.as_dcm()
            cp_mtx = np.zeros((4, 4), dtype='float32')
            cp_mtx[0:3, 0:3] = pr_mat
            cp_mtx[0:3, 3] = ctp1.T
            cp_mtx[3, 3] = 1

            new_cp = cor_mtx.dot(cp_mtx)

            for i in range(t_verts.size / 3):
                tmp_v = new_cp.dot(np.array([t_verts[i][0], t_verts[i][1], t_verts[i][2], 1]).T)
                pix_u = 320 * (tmp_v[0] / tmp_v[2]) + 320
                pix_v = 320 * (tmp_v[1] / tmp_v[2]) + 240
                residuals.append(pix_v - pair_pts1[k][i][0])
                residuals.append(pix_u - pair_pts1[k][i][1])

        if len(pair_pts2[k]) > 0:
            t_verts = (verts2[k]-offset[k]).dot(Rodrigues(rtp[k])) + offset[k] + tp[k]

            Crt = np.array(Crt2.r)
            Ct = np.array(Ct2.r)
            tmpr = R.from_rotvec(Crt)
            r_mat = tmpr.as_dcm()
            t_vec = Ct.T
            cor_mtx = np.zeros((4, 4), dtype='float32')
            cor_mtx[0:3, 0:3] = r_mat
            cor_mtx[0:3, 3] = t_vec
            cor_mtx[3, 3] = 1

            tmpcr = R.from_rotvec(crtp2)
            pr_mat = tmpcr.as_dcm()
            cp_mtx = np.zeros((4, 4), dtype='float32')
            cp_mtx[0:3, 0:3] = pr_mat
            cp_mtx[0:3, 3] = ctp2.T
            cp_mtx[3, 3] = 1

            new_cp = cor_mtx.dot(cp_mtx)

            for i in range(t_verts.size / 3):
                tmp_v = new_cp.dot(np.array([t_verts[i][0], t_verts[i][1], t_verts[i][2], 1]).T)
                pix_u = 320 * (tmp_v[0] / tmp_v[2]) + 320
                pix_v = 320 * (tmp_v[1] / tmp_v[2]) + 240
                residuals.append(pix_v - pair_pts2[k][i][0])
                residuals.append(pix_u - pair_pts2[k][i][1])

        if len(pair_pts3[k]) > 0:
            t_verts = (verts3[k]-offset[k]).dot(Rodrigues(rtp[k])) + offset[k] + tp[k]

            Crt = np.array(Crt3.r)
            Ct = np.array(Ct3.r)
            tmpr = R.from_rotvec(Crt)
            r_mat = tmpr.as_dcm()
            t_vec = Ct.T
            cor_mtx = np.zeros((4, 4), dtype='float32')
            cor_mtx[0:3, 0:3] = r_mat
            cor_mtx[0:3, 3] = t_vec
            cor_mtx[3, 3] = 1

            tmpcr = R.from_rotvec(crtp3)
            pr_mat = tmpcr.as_dcm()
            cp_mtx = np.zeros((4, 4), dtype='float32')
            cp_mtx[0:3, 0:3] = pr_mat
            cp_mtx[0:3, 3] = ctp3.T
            cp_mtx[3, 3] = 1

            new_cp = cor_mtx.dot(cp_mtx)

            for i in range(t_verts.size / 3):
                tmp_v = new_cp.dot(np.array([t_verts[i][0], t_verts[i][1], t_verts[i][2], 1]).T)
                pix_u = 320 * (tmp_v[0] / tmp_v[2]) + 320
                pix_v = 320 * (tmp_v[1] / tmp_v[2]) + 240
                residuals.append(pix_v - pair_pts3[k][i][0])
                residuals.append(pix_u - pair_pts3[k][i][1])

        if len(pair_pts4[k]) > 0:
            t_verts = (verts4[k]-offset[k]).dot(Rodrigues(rtp[k])) + offset[k] + tp[k]

            Crt = np.array(Crt4.r)
            Ct = np.array(Ct4.r)
            tmpr = R.from_rotvec(Crt)
            r_mat = tmpr.as_dcm()
            t_vec = Ct.T
            cor_mtx = np.zeros((4, 4), dtype='float32')
            cor_mtx[0:3, 0:3] = r_mat
            cor_mtx[0:3, 3] = t_vec
            cor_mtx[3, 3] = 1

            tmpcr = R.from_rotvec(crtp4)
            pr_mat = tmpcr.as_dcm()
            cp_mtx = np.zeros((4, 4), dtype='float32')
            cp_mtx[0:3, 0:3] = pr_mat
            cp_mtx[0:3, 3] = ctp4.T
            cp_mtx[3, 3] = 1

            new_cp = cor_mtx.dot(cp_mtx)

            for i in range(t_verts.size / 3):
                tmp_v = new_cp.dot(np.array([t_verts[i][0], t_verts[i][1], t_verts[i][2], 1]).T)
                pix_u = 320 * (tmp_v[0] / tmp_v[2]) + 320
                pix_v = 320 * (tmp_v[1] / tmp_v[2]) + 240
                residuals.append(pix_v - pair_pts4[k][i][0])
                residuals.append(pix_u - pair_pts4[k][i][1])

    residuals = np.vstack(residuals)

    return residuals

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

def reverse_camerapose(c_rt, c_t, op_rt, op_t, mean):
    rtn = np.array(c_rt)
    tmprt = R.from_rotvec(rtn)
    rt_mat = tmprt.as_dcm()
    # inv_rt = np.linalg.inv(rt_mat)
    t = np.array(c_t)
    cor_mtx = np.zeros((4, 4), dtype='float32')
    cor_mtx[0:3, 0:3] = rt_mat
    cor_mtx[0:3, 3] = t.T
    cor_mtx[3, 3] = 1
    # print cor_mtx
    # inv_cormtx = np.linalg.inv(cor_mtx)

    # Crt = np.array(op_rt)
    # Ct = np.array(t_row.r)
    tmpr = R.from_rotvec(op_rt)
    r_mat = tmpr.as_dcm()
    # inv_r = np.linalg.inv(r_mat)
    o_mtx = np.zeros((4, 4), dtype='float32')
    o_mtx[0:3, 0:3] = np.linalg.inv(r_mat)
    o_mtx[0:3, 3] = (mean - r_mat.dot(mean) + op_t).T
    o_mtx[3, 3] = 1

    new_RT = cor_mtx.dot(o_mtx)
    # print new_RT

    rc_mat = new_RT[0:3, 0:3]
    tmprt = R.from_dcm(rc_mat)
    new_crt = tmprt.as_rotvec()
    new_ct = new_RT[0:3, 3].T
    return new_crt, new_ct

def get_new_camerapose(c_rt, c_t, op_rt, op_t):
    rtn = np.array(c_rt)
    tmprt = R.from_rotvec(rtn)
    rt_mat = tmprt.as_dcm()
    # inv_rt = np.linalg.inv(rt_mat)
    t = np.array(c_t)
    cor_mtx = np.zeros((4, 4), dtype='float32')
    cor_mtx[0:3, 0:3] = rt_mat
    cor_mtx[0:3, 3] = t.T
    cor_mtx[3, 3] = 1
    # print cor_mtx
    # inv_cormtx = np.linalg.inv(cor_mtx)

    # Crt = np.array(op_rt)
    # Ct = np.array(t_row.r)
    tmpr = R.from_rotvec(op_rt)
    r_mat = tmpr.as_dcm()
    # inv_r = np.linalg.inv(r_mat)
    o_mtx = np.zeros((4, 4), dtype='float32')
    o_mtx[0:3, 0:3] = np.linalg.inv(r_mat)
    o_mtx[0:3, 3] = op_t.T
    o_mtx[3, 3] = 1

    new_RT = cor_mtx.dot(o_mtx)
    # print new_RT

    rc_mat = new_RT[0:3, 0:3]
    tmprt = R.from_dcm(rc_mat)
    new_crt = ch.array(tmprt.as_rotvec())
    new_ct = ch.array(new_RT[0:3, 3].T)
    return new_crt, new_ct

def parsing_camera_pose(file_camera):
    file = open(file_camera, 'r')
    lines = file.readlines()
    cc = 0
    cpose = []
    for line in lines:
        if cc == 0:
            pf = float(line) / 52 * 640
            cc +=1
            continue
        tmp = line.split(', ')
        cpose.append([float(tmp[0]), float(tmp[1]), float(tmp[2])])
    prt1 = ch.array(cpose[0])
    pt1 = ch.array(cpose[1])
    prt2 = ch.array(cpose[2])
    pt2 = ch.array(cpose[3])
    prt3 = ch.array(cpose[4])
    pt3 = ch.array(cpose[5])
    prt4 = ch.array(cpose[6])
    pt4 = ch.array(cpose[7])
    return prt1, pt1, prt2, pt2, prt3, pt3, prt4, pt4, pf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--t_model", help="tooth model file folder")
    parser.add_argument("--view1", help="view1 ground truth contour file")
    parser.add_argument("--view2", help="view2 ground truth contour file")
    parser.add_argument("--view3", help="view3 ground truth contour file")
    parser.add_argument("--view4", help="view4 ground truth contour file")
    parser.add_argument("--camera_pose", help="camera pose parameters file")
    parser.add_argument("--rt1", nargs=3, type=float, help="view1 camera pose rotation pars")
    parser.add_argument("--t1", nargs=3, type=float, help="view1 camera pose translation pars")
    parser.add_argument("--rt2", nargs=3, type=float, help="view2 camera pose rotation pars")
    parser.add_argument("--t2", nargs=3, type=float, help="view2 camera pose translation pars")
    parser.add_argument("--rt3", nargs=3, type=float, help="view3 camera pose rotation pars")
    parser.add_argument("--t3", nargs=3, type=float, help="view3 camera pose translation pars")
    parser.add_argument("--rt4", nargs=3, type=float, help="view4 camera pose rotation pars")
    parser.add_argument("--t4", nargs=3, type=float, help="view4 camera pose translation pars")
    parser.add_argument("--f", type=float, help="focal length")
    args = parser.parse_args()


    ### default value ####
    teeth_file_folder = '/home/jiaming/MultiviewFitting/data/upper_segmented/HBF_12681/before'
    # teeth_file_folder = 'data/from GuMin/seg/model_0'
    # moved_mesh_folder = '/home/jiaming/MultiviewFitting/data/observation/12681/movedRow_real1.obj'
    img1_file_path = '/home/jiaming/MultiviewFitting/data/observation/12681/real_rc1.jpg'
    img2_file_path = '/home/jiaming/MultiviewFitting/data/observation/12681/real_rc2.jpg'
    img3_file_path = '/home/jiaming/MultiviewFitting/data/observation/12681/real_rc3.jpg'
    img4_file_path = '/home/jiaming/MultiviewFitting/data/observation/12681/real_rc4.jpg'
    rt1 = ch.array([0, -0.3, 0]) * np.pi / 2
    t1 = ch.array([1.2, 0.2, 0])
    rt2 = ch.array([0.08, 0, 0]) * np.pi / 2
    t2 = ch.array([-0.05, 0.2, -0.25])
    rt3 = ch.array([-0.9, 0, 0]) * np.pi / 3
    t3 = ch.array([0, -1.5, 0.2])
    rt4 = ch.array([0.1, 0.4, 0]) * np.pi / 2
    t4 = ch.array([-1.4, 0.3, 0.2])
    f = 320

    #parsing the input arguments
    if args.t_model is not None:
        teeth_file_folder = args.t_model
    if args.view1 is not None:
        img1_file_path = args.view1
    if args.view2 is not None:
        img2_file_path = args.view2
    if args.view3 is not None:
        img3_file_path = args.view3
    if args.view4 is not None:
        img4_file_path = args.view4
    if args.camera_pose is not None:
        rt1, t1, rt2, t2, rt3, t3, rt4, t4, f = parsing_camera_pose(args.camera_pose)
    else:
        if args.rt1 is not None:
            rt1 = ch.array(args.rt1)
        if args.t1 is not None:
            t1 = ch.array(args.t1)
        if args.rt2 is not None:
            rt2 = ch.array(args.rt2)
        if args.t2 is not None:
            t2 = ch.array(args.t2)
        if args.rt3 is not None:
            rt3 = ch.array(args.rt3)
        if args.t3 is not None:
            t3 = ch.array(args.t3)
        if args.rt4 is not None:
            rt4 = ch.array(args.rt4)
        if args.t4 is not None:
            t4 = ch.array(args.t4)


    print(teeth_file_folder)

    teeth_row_mesh = Mesh.TeethRowMesh(teeth_file_folder, False)
    row_mesh = teeth_row_mesh.row_mesh

    # moved_mesh = Mesh.TeethRowMesh(moved_mesh_folder, True)
    # t0 = ch.asarray(teeth_row_mesh.positions_in_row)

    numTooth = len(teeth_row_mesh.mesh_list)
    Ri_list = [ch.zeros(3) for i in range(numTooth)]
    ti_list = [ch.zeros(3) for i in range(numTooth)]
    # R_row = ch.zeros(3)
    # t_row = ch.zeros(3)
    # R_row = ch.array([0, 0, 0.06])
    # t_row = ch.array([0, 0.06, 0])

    #random deviation
    for i in range(numTooth):
        tmprd, tmptd = randome_deviation(i*(time.time()), 18, 0.04)
        teeth_row_mesh.rotate(tmprd, i)
        teeth_row_mesh.translate(tmptd, i)


    # teeth_row_mesh.rotate(R_row)
    # teeth_row_mesh.translate(t_row)

    Vi_list = [ch.array(teeth_row_mesh.mesh_list[i].v) for i in range(numTooth)]
    Vi_offset = [ch.mean(Vi_list[i], axis=0) for i in range(numTooth)]
    # print(Vi_offset)

    Vi_center = [(Vi_list[i] - Vi_offset[i]) for i in range(numTooth)]

    # V_row = t_row + ch.vstack([ti_list[i] + Vi_offset[i] + Vi_center[i].dot(Rodrigues(Ri_list[i])) for i in range(numTooth)]).dot(Rodrigues(R_row))
    V_row = ch.vstack([Vi_list[i] for i in range(numTooth)])
    # V_comb = ch.vstack([ti_list[i] + Vi_list[i].mean(axis=0) + (Vi_list[i] - Vi_list[i].mean(axis=0)).dot(Rodrigues(Ri_list[i])) for i in range(numTooth)])
    # V_row = t_row + V_comb.mean(axis=0) + (V_comb - V_comb.mean(axis=0)).dot(Rodrigues(R_row))

    # Mesh.save_to_obj('result/V_row_gt.obj', V_row.r, row_mesh.f)

    w, h = (640, 480)

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # vw = cv2.VideoWriter('result/optimRecord.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (2*w, 2*h))

    # rn = _create_colored_renderer()
    # imtmp = simple_renderer(rn, m.v, m.f)

    rn = BoundaryRenderer()
    drn = DepthRenderer()
    crn = ColoredRenderer()
    # rn.camera = ProjectPoints(v=V_row, rt=ch.zeros(3), t=ch.array([0, 0, 0]), f=ch.array([w, w]) / 2.,
    #                           c=ch.array([w, h]) / 2.,
    #                           k=ch.zeros(5))
    # rt = ch.array([0, 1, 0]) * np.pi / 4
    # rn.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([-1.5, 0, 0.5]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    #12681

    rn.camera = ProjectPoints(v=V_row, rt=rt1, t=t1, f=ch.array([f, f]),
                                   c=ch.array([w, h]) / 2.,
                                   k=ch.zeros(5))
    drn.camera = ProjectPoints(v=V_row, rt=rt1, t=t1, f=ch.array([f, f]),
                              c=ch.array([w, h]) / 2.,
                              k=ch.zeros(5))
    crn.camera = ProjectPoints(v=V_row, rt=rt1, t=t1, f=ch.array([f, f]),
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))

    #13282
    # rt = ch.array([0, -0.25, 0.1]) * np.pi/2
    # rn.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([0.7, 0.5, 0]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    #13157
    # rt = ch.array([0, -0.4, 0]) * np.pi/2
    # rn.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([1.6, 0.2, 0.3]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    #from GuMin_1
    # rn.camera = ProjectPoints(v=V_row, rt=ch.array([0,0,0]), t=ch.array([0, 0, 0]), f=ch.array([w, w]) / 2.,
    #                               c=ch.array([w, h]) / 2.,
    #                               k=ch.zeros(5))
    rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    drn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    drn.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    crn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    crn.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc, bgcolor=ch.zeros(3), num_channels=3)

    rn2 = BoundaryRenderer()
    drn2 = DepthRenderer()
    crn2 = ColoredRenderer()
    # rt = ch.array([0, 1, 0]) * np.pi / 2
    # rn2.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([-2, 0, 2]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    # rt = ch.array([-1, 0, 0]) * np.pi / 4
    # rn2.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([0, -1.5, 0.4]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    #12681

    rn2.camera = ProjectPoints(v=V_row, rt=rt2, t=t2, f=ch.array([f, f]),
                                   c=ch.array([w, h]) / 2.,
                                   k=ch.zeros(5))
    drn2.camera = ProjectPoints(v=V_row, rt=rt2, t=t2, f=ch.array([f, f]),
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))
    crn2.camera = ProjectPoints(v=V_row, rt=rt2, t=t2, f=ch.array([f, f]),
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))
    #13282
    # rt = ch.array([0.2, 0, 0.05]) * np.pi/2
    # rn2.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([-0.1, 0.6, 0.2]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    #13157
    # rt = ch.array([0, 0, 0]) * np.pi / 2
    # rn2.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([0, 0.1, 0]), f=ch.array([w, w]) / 2.,
    #                           c=ch.array([w, h]) / 2.,
    #                           k=ch.zeros(5))
    #from GuMin_1
    # rt = ch.array([0, -1, 0]) * np.pi/4
    # rn2.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([1.2, 0, 0.5]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    rn2.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    rn2.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    drn2.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    drn2.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    crn2.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    crn2.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc, bgcolor=ch.zeros(3), num_channels=3)

    rn3 = BoundaryRenderer()
    drn3 = DepthRenderer()
    crn3 = ColoredRenderer()
    # rt = ch.array([0, -1, 0]) * np.pi/2
    # rn3.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([2, 0, 2]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    # rt = ch.array([-0.3, -1, 0]) * np.pi / 4
    # rn3.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([1.5, -0.5, 0.8]), f=ch.array([w, w]) / 2.,
    #                           c=ch.array([w, h]) / 2.,
    #                           k=ch.zeros(5))
    #12681

    rn3.camera = ProjectPoints(v=V_row, rt=rt3, t=t3, f=ch.array([f, f]),
                              c=ch.array([w, h]) / 2.,
                              k=ch.zeros(5))
    drn3.camera = ProjectPoints(v=V_row, rt=rt3, t=t3, f=ch.array([f, f]),
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))
    crn3.camera = ProjectPoints(v=V_row, rt=rt3, t=t3, f=ch.array([f, f]),
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))
    #13282
    # rt = ch.array([-0.7, 0, 0]) * np.pi / 3
    # rn3.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([-0.1, -1.1, 0]), f=ch.array([w, w]) / 2.,
    #                           c=ch.array([w, h]) / 2.,
    #                           k=ch.zeros(5))
    #13157
    # rt = ch.array([-0.85, 0, 0]) * np.pi/3
    # rn3.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([0, -1.3, 0.6]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    #from GuMin_1
    # rt = ch.array([0, 1.4, 0]) * np.pi / 4
    # rn3.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([-1.5, 0, 0.8]), f=ch.array([w, w]) / 2.,
    #                           c=ch.array([w, h]) / 2.,
    #                           k=ch.zeros(5))
    rn3.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    rn3.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    drn3.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    drn3.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    crn3.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    crn3.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc, bgcolor=ch.zeros(3), num_channels=3)


    rn4 = BoundaryRenderer()
    drn4 = DepthRenderer()
    crn4 = ColoredRenderer()
    # rt = ch.array([0, -1, 0]) * np.pi/4
    # rn4.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([1.5, 0, 0.5]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    # rt = ch.array([0, -1, 0]) * np.pi / 6
    # rn4.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([0.8, 0, 0.6]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    #12681

    rn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                                   c=ch.array([w, h]) / 2.,
                                   k=ch.zeros(5))
    drn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))
    crn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))
    #13282
    # rt = ch.array([0.1, 0.4, 0]) * np.pi/2
    # rn4.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([-1.3, 0.5, 0.2]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    #13157
    # rt = ch.array([0, 0.4, 0]) * np.pi/2
    # rn4.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([-1.4, 0.2, 0.2]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    #from GuMin_1
    # rt = ch.array([-0.8, 0, 0]) * np.pi/2
    # rn4.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([0, -1.5, 1]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    rn4.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    rn4.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    drn4.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    drn4.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    crn4.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    crn4.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc, bgcolor=ch.zeros(3), num_channels=3)

    # rn5 = BoundaryRenderer()
    # # rt = ch.array([0, 1, 0]) * np.pi/4
    # # rn5.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([-1.5, 0, 0.5]), f=ch.array([w, w]) / 2.,
    # #                                c=ch.array([w, h]) / 2.,
    # #                                k=ch.zeros(5))
    # rt = ch.array([0, -1, 0]) * np.pi / 4
    # rn5.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([1.5, 0, 0.5]), f=ch.array([w, w]) / 2.,
    #                            c=ch.array([w, h]) / 2.,
    #                            k=ch.zeros(5))
    # rn5.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    # rn5.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)

    observed1 = load_image(img1_file_path)
    observed2 = load_image(img2_file_path)
    observed3 = load_image(img3_file_path)
    observed4 = load_image(img4_file_path)

    # draw preparation
    plt.ion()
    fig, axarr = None, None
    ob1_dc = deepcopy(observed1)
    ob2_dc = deepcopy(observed2)
    ob3_dc = deepcopy(observed3)
    ob4_dc = deepcopy(observed4)
    # ob5_dc = deepcopy(observed5)
    ob1_dc[ob1_dc[:, :, 0] > 0] *= [0, 1, 0]
    ob2_dc[ob2_dc[:, :, 0] > 0] *= [0, 1, 0]
    ob3_dc[ob3_dc[:, :, 0] > 0] *= [0, 1, 0]
    ob4_dc[ob4_dc[:, :, 0] > 0] *= [0, 1, 0]

    start_time = time.time()
    total_time = 0

    #package the info
    obs = []
    obs.append(observed1)
    obs.append(observed2)
    obs.append(observed3)
    obs.append(observed4)
    cb_err = [100000, 100000, 100000, 100000]
    for iter in range(1):
        #optimize all the camera poses and individual tooth poses together
        err = 100000
        err_dif = 100
        iter = 0
        while not (err < 10 or err_dif < 1 or iter > 20):
            mean = []
            ins_pts1 = []
            ins_pts2 = []
            ins_pts3 = []
            ins_pts4 = []
            pairing_pts1 = []
            pairing_pts2 = []
            pairing_pts3 = []
            pairing_pts4 = []
            # print V_row.shape
            # cur_tooth = V_row[teeth_row_mesh.start_idx_list[i]:teeth_row_mesh.start_idx_list[i+1], 0:3]
            # print cur_tooth.shape

            #prepare data
            for i in range(numTooth):
                mean.append(np.mean(Vi_list[i].r, axis=0))
                # print mean

                sample_pts1 = get_sample_pts(rn.r, crn.r, i) #first time using intial rn
                if len(sample_pts1) < 2:
                    pair_pts1 = []
                    trim_ins_pts1 = []
                else:
                    # get back projection 3D verts and id of 2D points which can find corresponding verts
                    sp_ins_pts1 = pj.back_projection_depth(sample_pts1, rt1, t1, drn.r)
                    pair_pts1, trim_ins_pts1 = get_pair_pts(observed1, sample_pts1, sp_ins_pts1)  # get pairing points
                ins_pts1.append(trim_ins_pts1)
                pairing_pts1.append(pair_pts1)


                sample_pts2 = get_sample_pts(rn2.r, crn2.r, i)
                if len(sample_pts2) < 2:
                    pair_pts2 = []
                    trim_ins_pts2 = []
                else:
                    sp_ins_pts2 = pj.back_projection_depth(sample_pts2, rt2, t2, drn2.r)
                    pair_pts2, trim_ins_pts2 = get_pair_pts(observed2, sample_pts2, sp_ins_pts2)
                ins_pts2.append(trim_ins_pts2)
                pairing_pts2.append(pair_pts2)


                sample_pts3 = get_sample_pts(rn3.r, crn3.r, i)
                if len(sample_pts3) < 2:
                    pair_pts3 = []
                    trim_ins_pts3 = []
                else:
                    sp_ins_pts3 = pj.back_projection_depth(sample_pts3, rt3, t3, drn3.r)
                    pair_pts3, trim_ins_pts3 = get_pair_pts(observed3, sample_pts3, sp_ins_pts3)
                ins_pts3.append(trim_ins_pts3)
                pairing_pts3.append(pair_pts3)


                sample_pts4 = get_sample_pts(rn4.r, crn4.r, i)
                if len(sample_pts4) < 2:
                    pair_pts4 = []
                    trim_ins_pts4 = []
                else:
                    sp_ins_pts4 = pj.back_projection_depth(sample_pts4, rt4, t4, drn4.r)
                    pair_pts4, trim_ins_pts4 = get_pair_pts(observed4, sample_pts4, sp_ins_pts4)
                ins_pts4.append(trim_ins_pts4)
                pairing_pts4.append(pair_pts4)

            #optimization
            cur_time = time.time()
            # pars = np.array([0, 0, 0, 0, 0, 0], dtype='float32')
            pars = Parameters()
            pars.add('rtx1', value=0)
            pars.add('rty1', value=0)
            pars.add('rtz1', value=0)
            pars.add('tx1', value=0)
            pars.add('ty1', value=0)
            pars.add('tz1', value=0)
            pars.add('rtx2', value=0)
            pars.add('rty2', value=0)
            pars.add('rtz2', value=0)
            pars.add('tx2', value=0)
            pars.add('ty2', value=0)
            pars.add('tz2', value=0)
            pars.add('rtx3', value=0)
            pars.add('rty3', value=0)
            pars.add('rtz3', value=0)
            pars.add('tx3', value=0)
            pars.add('ty3', value=0)
            pars.add('tz3', value=0)
            pars.add('rtx4', value=0)
            pars.add('rty4', value=0)
            pars.add('rtz4', value=0)
            pars.add('tx4', value=0)
            pars.add('ty4', value=0)
            pars.add('tz4', value=0)
            pars.add('rtx5', value=0)
            pars.add('rty5', value=0)
            pars.add('rtz5', value=0)
            pars.add('tx5', value=0)
            pars.add('ty5', value=0)
            pars.add('tz5', value=0)
            pars.add('rtx6', value=0)
            pars.add('rty6', value=0)
            pars.add('rtz6', value=0)
            pars.add('tx6', value=0)
            pars.add('ty6', value=0)
            pars.add('tz6', value=0)
            pars.add('rtx7', value=0)
            pars.add('rty7', value=0)
            pars.add('rtz7', value=0)
            pars.add('tx7', value=0)
            pars.add('ty7', value=0)
            pars.add('tz7', value=0)
            pars.add('rtx8', value=0)
            pars.add('rty8', value=0)
            pars.add('rtz8', value=0)
            pars.add('tx8', value=0)
            pars.add('ty8', value=0)
            pars.add('tz8', value=0)
            pars.add('rtx9', value=0)
            pars.add('rty9', value=0)
            pars.add('rtz9', value=0)
            pars.add('tx9', value=0)
            pars.add('ty9', value=0)
            pars.add('tz9', value=0)
            pars.add('rtx10', value=0)
            pars.add('rty10', value=0)
            pars.add('rtz10', value=0)
            pars.add('tx10', value=0)
            pars.add('ty10', value=0)
            pars.add('tz10', value=0)
            pars.add('rtx11', value=0)
            pars.add('rty11', value=0)
            pars.add('rtz11', value=0)
            pars.add('tx11', value=0)
            pars.add('ty11', value=0)
            pars.add('tz11', value=0)
            pars.add('rtx12', value=0)
            pars.add('rty12', value=0)
            pars.add('rtz12', value=0)
            pars.add('tx12', value=0)
            pars.add('ty12', value=0)
            pars.add('tz12', value=0)
            pars.add('crtx1', value=0)
            pars.add('crty1', value=0)
            pars.add('crtz1', value=0)
            pars.add('ctx1', value=0)
            pars.add('cty1', value=0)
            pars.add('ctz1', value=0)
            pars.add('crtx2', value=0)
            pars.add('crty2', value=0)
            pars.add('crtz2', value=0)
            pars.add('ctx2', value=0)
            pars.add('cty2', value=0)
            pars.add('ctz2', value=0)
            pars.add('crtx3', value=0)
            pars.add('crty3', value=0)
            pars.add('crtz3', value=0)
            pars.add('ctx3', value=0)
            pars.add('cty3', value=0)
            pars.add('ctz3', value=0)
            pars.add('crtx4', value=0)
            pars.add('crty4', value=0)
            pars.add('crtz4', value=0)
            pars.add('ctx4', value=0)
            pars.add('cty4', value=0)
            pars.add('ctz4', value=0)

            out = lmfit.minimize(residual_allpars_allview, pars,
                                 args=(
                                 mean, ins_pts1, ins_pts2, ins_pts3, ins_pts4, rt1, t1, rt2, t2,
                                 rt3, t3, rt4, t4, pairing_pts1, pairing_pts2, pairing_pts3, pairing_pts4),
                                 method='leastsq')
            print('optimization time: %s s' % (time.time() - cur_time))

            err_dif = err - out.chisqr
            if (err_dif > 0):
                err = out.chisqr
                # out.params.pretty_print()
                tmprt = []
                tmpt = []
                tmprt.append(np.array([out.params['rtx1'], out.params['rty1'], out.params['rtz1']]))
                tmpt.append(np.array([out.params['tx1'], out.params['ty1'], out.params['tz1']]))
                tmprt.append(np.array([out.params['rtx2'], out.params['rty2'], out.params['rtz2']]))
                tmpt.append(np.array([out.params['tx2'], out.params['ty2'], out.params['tz2']]))
                tmprt.append(np.array([out.params['rtx3'], out.params['rty3'], out.params['rtz3']]))
                tmpt.append(np.array([out.params['tx3'], out.params['ty3'], out.params['tz3']]))
                tmprt.append(np.array([out.params['rtx4'], out.params['rty4'], out.params['rtz4']]))
                tmpt.append(np.array([out.params['tx4'], out.params['ty4'], out.params['tz4']]))
                tmprt.append(np.array([out.params['rtx5'], out.params['rty5'], out.params['rtz5']]))
                tmpt.append(np.array([out.params['tx5'], out.params['ty5'], out.params['tz5']]))
                tmprt.append(np.array([out.params['rtx6'], out.params['rty6'], out.params['rtz6']]))
                tmpt.append(np.array([out.params['tx6'], out.params['ty6'], out.params['tz6']]))
                tmprt.append(np.array([out.params['rtx7'], out.params['rty7'], out.params['rtz7']]))
                tmpt.append(np.array([out.params['tx7'], out.params['ty7'], out.params['tz7']]))
                tmprt.append(np.array([out.params['rtx8'], out.params['rty8'], out.params['rtz8']]))
                tmpt.append(np.array([out.params['tx8'], out.params['ty8'], out.params['tz8']]))
                tmprt.append(np.array([out.params['rtx9'], out.params['rty9'], out.params['rtz9']]))
                tmpt.append(np.array([out.params['tx9'], out.params['ty9'], out.params['tz9']]))
                tmprt.append(np.array([out.params['rtx10'], out.params['rty10'], out.params['rtz10']]))
                tmpt.append(np.array([out.params['tx10'], out.params['ty10'], out.params['tz10']]))
                tmprt.append(np.array([out.params['rtx11'], out.params['rty11'], out.params['rtz11']]))
                tmpt.append(np.array([out.params['tx11'], out.params['ty11'], out.params['tz11']]))
                tmprt.append(np.array([out.params['rtx12'], out.params['rty12'], out.params['rtz12']]))
                tmpt.append(np.array([out.params['tx12'], out.params['ty12'], out.params['tz12']]))
                print tmprt, tmpt

                for i in range(12):
                    Vi_list[i] = (Vi_list[i] - mean[i]).dot(Rodrigues(tmprt[i])) + mean[i] + tmpt[i]

                V_row = ch.vstack([Vi_list[k] for k in range(numTooth)])
                # V_row = (V_row-mean).dot(Rodrigues(tmprt)) + mean + tmpt

                tcrt1 = np.array([out.params['crtx1'], out.params['crty1'], out.params['crtz1']])
                tct1 = np.array([out.params['ctx1'], out.params['cty1'], out.params['ctz1']])
                tcrt2 = np.array([out.params['crtx2'], out.params['crty2'], out.params['crtz2']])
                tct2 = np.array([out.params['ctx2'], out.params['cty2'], out.params['ctz2']])
                tcrt3 = np.array([out.params['crtx3'], out.params['crty3'], out.params['crtz3']])
                tct3 = np.array([out.params['ctx3'], out.params['cty3'], out.params['ctz3']])
                tcrt4 = np.array([out.params['crtx4'], out.params['crty4'], out.params['crtz4']])
                tct4 = np.array([out.params['ctx4'], out.params['cty4'], out.params['ctz4']])
                print tcrt1, tct1, tcrt2, tct2, tcrt3, tct3, tcrt4, tct4

                rt1, t1 = get_new_camerapose(rt1, t1, tcrt1, tct1)
                rt2, t2 = get_new_camerapose(rt2, t2, tcrt2, tct2)
                rt3, t3 = get_new_camerapose(rt3, t3, tcrt3, tct3)
                rt4, t4 = get_new_camerapose(rt4, t4, tcrt4, tct4)

            print out.message, out.chisqr

            # reproject 2D contour
            rn.camera = ProjectPoints(v=V_row, rt=rt1, t=t1, f=ch.array([f, f]),
                                      c=ch.array([w, h]) / 2.,
                                      k=ch.zeros(5))
            rn2.camera = ProjectPoints(v=V_row, rt=rt2, t=t2, f=ch.array([f, f]),
                                       c=ch.array([w, h]) / 2.,
                                       k=ch.zeros(5))
            rn3.camera = ProjectPoints(v=V_row, rt=rt3, t=t3, f=ch.array([f, f]),
                                       c=ch.array([w, h]) / 2.,
                                       k=ch.zeros(5))
            rn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                                       c=ch.array([w, h]) / 2.,
                                       k=ch.zeros(5))
            drn.camera = ProjectPoints(v=V_row, rt=rt1, t=t1, f=ch.array([f, f]),
                                       c=ch.array([w, h]) / 2.,
                                       k=ch.zeros(5))
            drn2.camera = ProjectPoints(v=V_row, rt=rt2, t=t2, f=ch.array([f, f]),
                                        c=ch.array([w, h]) / 2.,
                                        k=ch.zeros(5))
            drn3.camera = ProjectPoints(v=V_row, rt=rt3, t=t3, f=ch.array([f, f]),
                                        c=ch.array([w, h]) / 2.,
                                        k=ch.zeros(5))
            drn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                                        c=ch.array([w, h]) / 2.,
                                        k=ch.zeros(5))
            crn.camera = ProjectPoints(v=V_row, rt=rt1, t=t1, f=ch.array([f, f]),
                                       c=ch.array([w, h]) / 2.,
                                       k=ch.zeros(5))
            crn2.camera = ProjectPoints(v=V_row, rt=rt2, t=t2, f=ch.array([f, f]),
                                        c=ch.array([w, h]) / 2.,
                                        k=ch.zeros(5))
            crn3.camera = ProjectPoints(v=V_row, rt=rt3, t=t3, f=ch.array([f, f]),
                                        c=ch.array([w, h]) / 2.,
                                        k=ch.zeros(5))
            crn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                                        c=ch.array([w, h]) / 2.,
                                        k=ch.zeros(5))

            # Mesh.save_to_obj('result/V_row_op.obj', V_row, row_mesh.f)
            # rn_c2 = deepcopy(rn.r)
            # rn_c2[rn_c2[:, :, 0] > 0] *= [0, 1, 0]
            #
            # plt.imshow(rn_c2 + rn_c)
            # plt.show()
            # break
            iter += 1

            # draw contours of different views
            if axarr is None:
                fig, axarr = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'wspace': 0, 'hspace': 0})
                # fig.subplots_adjust(hspace=0, wspace=0)
            fig.patch.set_facecolor('grey')

            rn1_dc = deepcopy(rn.r)
            rn2_dc = deepcopy(rn2.r)
            rn3_dc = deepcopy(rn3.r)
            rn4_dc = deepcopy(rn4.r)

            rn1_dc[rn1_dc[:, :, 0] > 0] *= [1, 0, 0]
            rn2_dc[rn2_dc[:, :, 0] > 0] *= [1, 0, 0]
            rn3_dc[rn3_dc[:, :, 0] > 0] *= [1, 0, 0]
            rn4_dc[rn4_dc[:, :, 0] > 0] *= [1, 0, 0]

            axarr[0, 0].imshow(rn1_dc + ob1_dc)
            axarr[0, 1].imshow(rn2_dc + ob2_dc)
            axarr[1, 0].imshow(rn3_dc + ob3_dc)
            axarr[1, 1].imshow(rn4_dc + ob4_dc)

            scipy.misc.imsave('result/log_allpars/fittingresult1_iter{}.jpg'.format(i), rn1_dc + ob1_dc)
            scipy.misc.imsave('result/log_allpars/fittingresult1_iter{}a.jpg'.format(i), rn1_dc)
            scipy.misc.imsave('result/log_allpars/fittingresult1_iter{}b.jpg'.format(i), ob1_dc)
            scipy.misc.imsave('result/log_allpars/fittingresult2_iter{}.jpg'.format(i), rn2_dc + ob2_dc)
            scipy.misc.imsave('result/log_allpars/fittingresult2_iter{}a.jpg'.format(i), rn2_dc)
            scipy.misc.imsave('result/log_allpars/fittingresult2_iter{}b.jpg'.format(i), ob2_dc)
            scipy.misc.imsave('result/log_allpars/fittingresult3_iter{}.jpg'.format(i), rn3_dc + ob3_dc)
            scipy.misc.imsave('result/log_allpars/fittingresult3_iter{}a.jpg'.format(i), rn3_dc)
            scipy.misc.imsave('result/log_allpars/fittingresult3_iter{}b.jpg'.format(i), ob3_dc)
            scipy.misc.imsave('result/log_allpars/fittingresult4_iter{}.jpg'.format(i), rn4_dc + ob4_dc)
            scipy.misc.imsave('result/log_allpars/fittingresult4_iter{}a.jpg'.format(i), rn4_dc)
            scipy.misc.imsave('result/log_allpars/fittingresult4_iter{}b.jpg'.format(i), ob4_dc)

            plt.pause(5)

        print("final error: %f total time:--- %s seconds ---" % (err, time.time() - start_time))
        total_time += (time.time() - start_time)
        start_time = time.time()


    print("total time --- %s seconds ---" % (total_time))
    Mesh.save_to_obj('result/V_row_opm_allpars.obj', V_row, row_mesh.f)
