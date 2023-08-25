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
import os, glob
import argparse


def get_sample_pts(contour, colormap=None, t_id=None, part=0, numT = 14):
    # decrease the size
    # contour1 = scipy.misc.imresize(contour, 0.5)
    # Sample points
    t_sample_pts = []
    f_sample_pts = []
    # print contour1.size

    f_sample_pts = np.argwhere(contour[:, :, 0] > 0)

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


def check_around(colormap, t_id, checkpixel, part = 0, numT = 14):
    for i in range(-1, 1, 1):
        for j in range(-1, 1, 1):
            # if part == 0:
            if np.rint(colormap[checkpixel[0] + i, checkpixel[1] + j][0] * 100 / 3) == t_id and \
                    colormap[checkpixel[0] + i, checkpixel[1] + j][1] > 0:
                return True
            # else:
            #     if colormap[checkpixel[0] + i, checkpixel[1] + j][0] * 100 / 6 == float(t_id-numT)  and \
            #             colormap[checkpixel[0] + i, checkpixel[1] + j][1] > 0:
            #         return True
    return False


def get_pair_pts(gt_contour, sp_pts, ins_pts, pair_id=None):
    # sample ground truth contour points
    gt_pts = []
    index = 0
    gt_pts = np.argwhere(gt_contour[:, :, 0] > 0.5)
    # draw_pixel(np.array(gt_contour), gt_pts)

    # build KDTree for gt points
    gtpts_tree = scipy.spatial.KDTree(gt_pts)
    # get sample points which finds back projection 3D verts
    s_sp_pts = []
    if pair_id is not None:
        for i in range(pair_id.shape[0]):
            s_sp_pts.append(sp_pts[pair_id[i]])
    else:
        s_sp_pts = sp_pts

    # pairing
    pair_pts = []
    new_ins_pts = []
    pair_res = gtpts_tree.query(s_sp_pts)

    # trimming: set a threshhold to filter outliers
    tmp_pair = deepcopy(pair_res[0])
    tmp_pair.sort()
    mean_dis = np.mean(pair_res, axis=1)[0]
    print mean_dis, tmp_pair[len(tmp_pair) // 2]
    threshhold = np.min([mean_dis * 2, tmp_pair[len(tmp_pair) // 2] * 2, 12])

    for i in range(len(pair_res[1])):
        if pair_res[0][i] <= threshhold:
            pair_pts.append(gt_pts[pair_res[1][i]])
            new_ins_pts.append(ins_pts[i])
    print len(pair_pts), 'out of', len(pair_res[1])

    # draw_pixel(np.array(gt_contour), pair_pts, s_sp_pts)
    return pair_pts, new_ins_pts


# visualization for checking
def draw_pixel(contour, points1, points2=None):
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
    # cv2.imshow('pair points', img)
    plt.imshow(img)
    return 0


# residual for only translation
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
        pix_u = 320 * (tmp_v[0] / tmp_v[2]) + 320
        pix_v = 320 * (tmp_v[1] / tmp_v[2]) + 240
        residuals.append(pix_v - pair_pts[i][0])
        residuals.append(pix_u - pair_pts[i][1])
    residuals = np.vstack(residuals)

    return residuals


# residual for translation and rotation for one view
def residual_rtt(pars, w, h, f, offset, verts, Crt, Ct, pair_pts):
    tp = np.array([pars['tx'], pars['ty'], pars['tz']])
    rtp = np.array([pars['rtx'], pars['rty'], pars['rtz']])
    t_verts = (verts - offset).dot(Rodrigues(rtp)) + offset + tp
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
        pix_u = f * (tmp_v[0] / tmp_v[2]) + w / 2
        pix_v = f * (tmp_v[1] / tmp_v[2]) + h / 2
        residuals.append(pix_v - pair_pts[i][0])
        residuals.append(pix_u - pair_pts[i][1])
    residuals = np.vstack(residuals)

    return residuals


# residual for translation and rotation for all the views (individual tooth)
def residual_rtt_allview(pars, w, h, f, offset, verts1, verts2, verts3, verts4, verts5, Crt1, Ct1, Crt2, Ct2,
                         Crt3, Ct3, Crt4, Ct4, Crt5, Ct5, pair_pts1, pair_pts2, pair_pts3, pair_pts4,
                         pair_pts5):
    residuals = []
    tp = np.array([pars['tx'], pars['ty'], pars['tz']])
    rtp = np.array([pars['rtx'], pars['rty'], pars['rtz']])

    if len(pair_pts1) > 0:
        t_verts = (verts1 - offset).dot(Rodrigues(rtp)) + offset + tp
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
            pix_u = f * (tmp_v[0] / tmp_v[2]) + w / 2
            pix_v = f * (tmp_v[1] / tmp_v[2]) + h / 2
            residuals.append(pix_v - pair_pts1[i][0])
            residuals.append(pix_u - pair_pts1[i][1])

    if len(pair_pts2) > 0:
        t_verts = (verts2 - offset).dot(Rodrigues(rtp)) + offset + tp
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
            pix_u = f * (tmp_v[0] / tmp_v[2]) + w / 2
            pix_v = f * (tmp_v[1] / tmp_v[2]) + h / 2
            residuals.append(pix_v - pair_pts2[i][0])
            residuals.append(pix_u - pair_pts2[i][1])

    if len(pair_pts3) > 0:
        t_verts = (verts3 - offset).dot(Rodrigues(rtp)) + offset + tp
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
            pix_u = f * (tmp_v[0] / tmp_v[2]) + w / 2
            pix_v = f * (tmp_v[1] / tmp_v[2]) + h / 2
            residuals.append(pix_v - pair_pts3[i][0])
            residuals.append(pix_u - pair_pts3[i][1])

    if len(pair_pts4) > 0:
        t_verts = (verts4 - offset).dot(Rodrigues(rtp)) + offset + tp
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
            pix_u = f * (tmp_v[0] / tmp_v[2]) + w / 2
            pix_v = f * (tmp_v[1] / tmp_v[2]) + h / 2
            residuals.append(pix_v - pair_pts4[i][0])
            residuals.append(pix_u - pair_pts4[i][1])

    if len(pair_pts5) > 0:
        t_verts = (verts5 - offset).dot(Rodrigues(rtp)) + offset + tp
        Crt = np.array(Crt5.r)
        Ct = np.array(Ct5.r)
        tmpr = R.from_rotvec(Crt)
        r_mat = tmpr.as_dcm()
        t_vec = Ct.T
        cor_mtx = np.zeros((4, 4), dtype='float32')
        cor_mtx[0:3, 0:3] = r_mat
        cor_mtx[0:3, 3] = t_vec
        cor_mtx[3, 3] = 1
        for i in range(t_verts.size / 3):
            tmp_v = cor_mtx.dot(np.array([t_verts[i][0], t_verts[i][1], t_verts[i][2], 1]).T)
            pix_u = f * (tmp_v[0] / tmp_v[2]) + w / 2
            pix_v = f * (tmp_v[1] / tmp_v[2]) + h / 2
            residuals.append(pix_v - pair_pts5[i][0])
            residuals.append(pix_u - pair_pts5[i][1])

    # if len(pair_pts6) > 0:
    #     t_verts = (verts6 - offset).dot(Rodrigues(rtp)) + offset + tp
    #     Crt = np.array(Crt6.r)
    #     Ct = np.array(Ct6.r)
    #     tmpr = R.from_rotvec(Crt)
    #     r_mat = tmpr.as_dcm()
    #     t_vec = Ct.T
    #     cor_mtx = np.zeros((4, 4), dtype='float32')
    #     cor_mtx[0:3, 0:3] = r_mat
    #     cor_mtx[0:3, 3] = t_vec
    #     cor_mtx[3, 3] = 1
    #     for i in range(t_verts.size / 3):
    #         tmp_v = cor_mtx.dot(np.array([t_verts[i][0], t_verts[i][1], t_verts[i][2], 1]).T)
    #         pix_u = f * (tmp_v[0] / tmp_v[2]) + w / 2
    #         pix_v = f * (tmp_v[1] / tmp_v[2]) + h / 2
    #         residuals.append(pix_v - pair_pts6[i][0])
    #         residuals.append(pix_u - pair_pts6[i][1])
    #
    # if len(pair_pts7) > 0:
    #     t_verts = (verts7 - offset).dot(Rodrigues(rtp)) + offset + tp
    #     Crt = np.array(Crt7.r)
    #     Ct = np.array(Ct7.r)
    #     tmpr = R.from_rotvec(Crt)
    #     r_mat = tmpr.as_dcm()
    #     t_vec = Ct.T
    #     cor_mtx = np.zeros((4, 4), dtype='float32')
    #     cor_mtx[0:3, 0:3] = r_mat
    #     cor_mtx[0:3, 3] = t_vec
    #     cor_mtx[3, 3] = 1
    #     for i in range(t_verts.size / 3):
    #         tmp_v = cor_mtx.dot(np.array([t_verts[i][0], t_verts[i][1], t_verts[i][2], 1]).T)
    #         pix_u = f * (tmp_v[0] / tmp_v[2]) + w / 2
    #         pix_v = f * (tmp_v[1] / tmp_v[2]) + h / 2
    #         residuals.append(pix_v - pair_pts7[i][0])
    #         residuals.append(pix_u - pair_pts7[i][1])

    residuals = np.vstack(residuals)

    return residuals


def randome_deviation(rseed, rd_range, td_range):
    random.seed(rseed)
    rx = random.random()
    ry = random.uniform(0, 1 - rx)
    rz = random.uniform(0, 1 - rx - ry)
    randr = np.array([rx, ry, rz]) * np.pi / rd_range
    tx = random.random()
    ty = random.uniform(0, 1 - tx)
    tz = random.uniform(0, 1 - tx - ty)
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

def get_gt_images(folder):
    gt_files = glob.glob(os.path.join(folder, '*.jpg'))
    gt_files.sort()
    print gt_files
    return gt_files

def parsing_camera_pose(file_camera):
    file = open(file_camera, 'r')
    lines = file.readlines()
    cc = 0
    cpose = []
    for line in lines:
        if cc == 0:
            tmp = line.split(' ')
            pw = int(tmp[2])
            ph = int(tmp[3])
            pf = float(tmp[0]) / float(tmp[1]) * pw
            cc += 1
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
    prt5 = ch.array(cpose[8])
    pt5 = ch.array(cpose[9])
    return prt1, pt1, prt2, pt2, prt3, pt3, prt4, pt4, prt5, pt5, pf, pw, ph


def drawfig(iter):
    ob1_dc = deepcopy(observed1)
    ob2_dc = deepcopy(observed2)
    ob3_dc = deepcopy(observed3)
    ob4_dc = deepcopy(observed4)
    ob5_dc = deepcopy(observed5)
    # ob6_dc = deepcopy(observed6)
    # ob7_dc = deepcopy(observed7)

    ob1_dc[ob1_dc[:, :, 0] > 0] *= [0, 1, 0]
    ob2_dc[ob2_dc[:, :, 0] > 0] *= [0, 1, 0]
    ob3_dc[ob3_dc[:, :, 0] > 0] *= [0, 1, 0]
    ob4_dc[ob4_dc[:, :, 0] > 0] *= [0, 1, 0]
    ob5_dc[ob5_dc[:, :, 0] > 0] *= [0, 1, 0]
    # ob6_dc[ob6_dc[:, :, 0] > 0] *= [0, 1, 0]
    # ob7_dc[ob7_dc[:, :, 0] > 0] *= [0, 1, 0]

    rn1_dc = deepcopy(rn.r)
    rn2_dc = deepcopy(rn2.r)
    rn3_dc = deepcopy(rn3.r)
    rn4_dc = deepcopy(rn4.r)
    rn5_dc = deepcopy(rn5.r)
    # rn6_dc = deepcopy(rn6.r)
    # rn7_dc = deepcopy(rn7.r)

    # crn1_dc = deepcopy(crn.r)
    # crn2_dc = deepcopy(crn2.r)
    # crn3_dc = deepcopy(crn3.r)
    # crn4_dc = deepcopy(crn4.r)
    # crn5_dc = deepcopy(crn5.r)
    # crn6_dc = deepcopy(crn6.r)

    rn1_dc[rn1_dc[:, :, 0] > 0] *= [1, 0, 0]
    rn2_dc[rn2_dc[:, :, 0] > 0] *= [1, 0, 0]
    rn3_dc[rn3_dc[:, :, 0] > 0] *= [1, 0, 0]
    rn4_dc[rn4_dc[:, :, 0] > 0] *= [1, 0, 0]
    rn5_dc[rn5_dc[:, :, 0] > 0] *= [1, 0, 0]
    # rn6_dc[rn6_dc[:, :, 0] > 0] *= [1, 0, 0]
    # rn7_dc[rn7_dc[:, :, 0] > 0] *= [1, 0, 0]

    # ob1_dc[rn1_dc[:, :, 0] > 0] = crn1_dc[rn1_dc[:, :, 0] > 0]
    # ob2_dc[rn2_dc[:, :, 0] > 0] = crn2_dc[rn2_dc[:, :, 0] > 0]
    # ob3_dc[rn3_dc[:, :, 0] > 0] = crn3_dc[rn3_dc[:, :, 0] > 0]
    # ob4_dc[rn4_dc[:, :, 0] > 0] = crn4_dc[rn4_dc[:, :, 0] > 0]
    # ob5_dc[rn5_dc[:, :, 0] > 0] = crn5_dc[rn5_dc[:, :, 0] > 0]
    # ob6_dc[rn6_dc[:, :, 0] > 0] = crn6_dc[rn6_dc[:, :, 0] > 0]
    # draw_color(crn1_dc, rn1_dc, ob1_dc)
    # draw_color(crn2_dc, rn2_dc, ob2_dc)
    # draw_color(crn3_dc, rn3_dc, ob3_dc)
    # draw_color(crn4_dc, rn4_dc, ob4_dc)
    # draw_color(crn5_dc, rn5_dc, ob5_dc)
    # draw_color(crn6_dc, rn6_dc, ob6_dc)

    # scipy.misc.imsave('result/log2/fittingresult1_iter%02d.jpg'%(iter), ob1_dc)
    scipy.misc.imsave('result/log2/state/fittingresult1_iter{}a.jpg'.format(iter), rn1_dc)
    scipy.misc.imsave('result/log2/state/fittingresult1_iter{}b.jpg'.format(iter), ob1_dc)
    scipy.misc.imsave('result/log2/state/fittingresult1_iter{}.jpg'.format(iter), ob1_dc + rn1_dc)
    # scipy.misc.imsave('result/log2/fittingresult2_iter%02d.jpg'%(iter), ob2_dc)
    scipy.misc.imsave('result/log2/state/fittingresult2_iter{}a.jpg'.format(iter), rn2_dc)
    scipy.misc.imsave('result/log2/state/fittingresult2_iter{}b.jpg'.format(iter), ob2_dc)
    scipy.misc.imsave('result/log2/state/fittingresult2_iter{}.jpg'.format(iter), ob2_dc + rn2_dc)
    # scipy.misc.imsave('result/log2/fittingresult3_iter%02d.jpg'%(iter), ob3_dc)
    scipy.misc.imsave('result/log2/state/fittingresult3_iter{}a.jpg'.format(iter), rn3_dc)
    scipy.misc.imsave('result/log2/state/fittingresult3_iter{}b.jpg'.format(iter), ob3_dc)
    scipy.misc.imsave('result/log2/state/fittingresult3_iter{}.jpg'.format(iter), ob3_dc + rn3_dc)
    # scipy.misc.imsave('result/log2/fittingresult4_iter%02d.jpg'%(iter), ob4_dc)
    scipy.misc.imsave('result/log2/state/fittingresult4_iter{}a.jpg'.format(iter), rn4_dc)
    scipy.misc.imsave('result/log2/state/fittingresult4_iter{}b.jpg'.format(iter), ob4_dc)
    scipy.misc.imsave('result/log2/state/fittingresult4_iter{}.jpg'.format(iter), ob4_dc + rn4_dc)
    # scipy.misc.imsave('result/log2/fittingresult5_iter%02d.jpg'%(iter), ob5_dc)
    scipy.misc.imsave('result/log2/state/fittingresult5_iter{}a.jpg'.format(iter), rn5_dc)
    scipy.misc.imsave('result/log2/state/fittingresult5_iter{}b.jpg'.format(iter), ob5_dc)
    scipy.misc.imsave('result/log2/state/fittingresult5_iter{}.jpg'.format(iter), ob5_dc + rn5_dc)
    # scipy.misc.imsave('result/log2/fittingresult6_iter%02d.jpg'%(iter), ob6_dc)
    # scipy.misc.imsave('result/log2/state/fittingresult6_iter{}a.jpg'.format(iter), rn6_dc)
    # scipy.misc.imsave('result/log2/state/fittingresult6_iter{}b.jpg'.format(iter), ob6_dc)
    # scipy.misc.imsave('result/log2/state/fittingresult6_iter{}.jpg'.format(iter), ob6_dc + rn6_dc)
    # scipy.misc.imsave('result/log2/state/fittingresult7_iter{}a.jpg'.format(iter), rn7_dc)
    # scipy.misc.imsave('result/log2/state/fittingresult7_iter{}b.jpg'.format(iter), ob7_dc)
    # scipy.misc.imsave('result/log2/state/fittingresult7_iter{}.jpg'.format(iter), ob7_dc + rn7_dc)

    print('fig saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--t_model_u", help="upper tooth model file folder")
    parser.add_argument("--t_model_l", help="lower tooth model file folder")
    parser.add_argument("--gt_contour", help="ground truth contour images")
    parser.add_argument("--camera_pose", help="camera pose parameters file")
    args = parser.parse_args()

    ### default value ####
    teeth_file_folder = r'/home/sky7hate/Project/MultiviewFitting/data/seg/model1_u'
    teeth_file_folder_low = r'/home/sky7hate/Project/MultiviewFitting/data/seg/model1_l'
    img1_file_path = r'/home/sky7hate/Project/MultiviewFitting/data/observation/new_seg/bite/gt1n.jpg'
    img2_file_path = r'/home/sky7hate/Project/MultiviewFitting/data/observation/new_seg/bite/gt3n.jpg'
    img3_file_path = r'/home/sky7hate/Project/MultiviewFitting/data/observation/new_seg/bite/gt5n.jpg'
    img4_file_path = r'/home/sky7hate/Project/MultiviewFitting/data/observation/new_seg/bite/gt6n.jpg'
    img5_file_path = r'/home/sky7hate/Project/MultiviewFitting/data/observation/new_seg/bite/gt7n.jpg'
    gt_list = None
    # rt1 = ch.array([0.3, -1.6, -0.015]) * np.pi / 4
    # t1 = ch.array([0.557, -0.20, 3.1])
    # rt2 = ch.array([0.43, 0, 0.03]) * np.pi / 4
    # t2 = ch.array([0.08, 0.03, 2.9])
    # rt3 = ch.array([0.5, 1.78, 0.16]) * np.pi / 4
    # t3 = ch.array([-0.35, -0.28, 3.35])
    # rt4 = ch.array([-2.36, 0.05, 0.05]) * np.pi / 4
    # t4 = ch.array([-0.055, -0.39, 2.3])
    # rt5 = ch.array([1.55, -0.07, 0.05]) * np.pi / 4
    # t5 = ch.array([0.018, 0.07, 2.25])
    # w, h = (640, 480)
    # f = w * 4 / 4.8

    # parsing the input arguments
    if args.t_model_u is not None:
        teeth_file_folder = args.t_model_u
    if args.t_model_l is not None:
        teeth_file_folder_low = args.t_model_l
    if args.gt_contour is not None:
        gt_list = get_gt_images(args.gt_contour)
        print gt_list
    if args.camera_pose is not None:
        rt1, t1, rt2, t2, rt3, t3, rt4, t4, rt5, t5, f, w, h = parsing_camera_pose(args.camera_pose)

    # print(teeth_file_folder)

    teeth_row_mesh = Mesh.TeethRowMesh(teeth_file_folder, 0, False)
    row_mesh = teeth_row_mesh.row_mesh

    numTooth = len(teeth_row_mesh.mesh_list)

    # lower teeth
    teeth_row_mesh_l = Mesh.TeethRowMesh(teeth_file_folder_low, 1, numTooth, False)
    row_mesh_l = teeth_row_mesh_l.row_mesh

    numTooth_l = len(teeth_row_mesh_l.mesh_list)
    teeth_row_mesh_l.rotate(np.array([0, 0, np.pi]))
    teeth_row_mesh_l.translate(np.array([0.01, 0.13, -0.04]))

    Vi_list = [ch.array(teeth_row_mesh.mesh_list[i].v) for i in range(numTooth)]
    Vi_offset = [ch.mean(Vi_list[i], axis=0) for i in range(numTooth)]
    # print(Vi_offset)
    Vi_center = [(Vi_list[i] - Vi_offset[i]) for i in range(numTooth)]
    V_row = ch.vstack([Vi_list[i] for i in range(numTooth)])

    Vi_list_l = [ch.array(teeth_row_mesh_l.mesh_list[i].v) for i in range(numTooth_l)]
    Vi_offset_l = [ch.mean(Vi_list_l[i], axis=0) for i in range(numTooth_l)]
    # print(Vi_offset)
    Vi_center_l = [(Vi_list_l[i] - Vi_offset_l[i]) for i in range(numTooth_l)]
    V_row_l = ch.vstack([Vi_list_l[i] for i in range(numTooth_l)])

    merged_mesh = Mesh.merge_mesh(row_mesh, row_mesh_l)
    V_row_bite = ch.array(merged_mesh.v)


    rn = BoundaryRenderer()
    drn = DepthRenderer()
    crn = ColoredRenderer()

    rn.camera = ProjectPoints(v=V_row_bite, rt=rt1, t=t1, f=ch.array([f, f]),
                              c=ch.array([w, h]) / 2.,
                              k=ch.zeros(5))
    drn.camera = ProjectPoints(v=V_row_bite, rt=rt1, t=t1, f=ch.array([f, f]),
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))
    crn.camera = ProjectPoints(v=V_row_bite, rt=rt1, t=t1, f=ch.array([f, f]),
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))

    rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    drn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    drn.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    crn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    crn.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc, bgcolor=ch.zeros(3), num_channels=3)

    rn2 = BoundaryRenderer()
    drn2 = DepthRenderer()
    crn2 = ColoredRenderer()

    rn2.camera = ProjectPoints(v=V_row_bite, rt=rt2, t=t2, f=ch.array([f, f]),
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))
    drn2.camera = ProjectPoints(v=V_row_bite, rt=rt2, t=t2, f=ch.array([f, f]),
                                c=ch.array([w, h]) / 2.,
                                k=ch.zeros(5))
    crn2.camera = ProjectPoints(v=V_row_bite, rt=rt2, t=t2, f=ch.array([f, f]),
                                c=ch.array([w, h]) / 2.,
                                k=ch.zeros(5))

    rn2.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    rn2.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    drn2.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    drn2.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    crn2.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    crn2.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc, bgcolor=ch.zeros(3), num_channels=3)

    rn3 = BoundaryRenderer()
    drn3 = DepthRenderer()
    crn3 = ColoredRenderer()

    rn3.camera = ProjectPoints(v=V_row_bite, rt=rt3, t=t3, f=ch.array([f, f]),
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))
    drn3.camera = ProjectPoints(v=V_row_bite, rt=rt3, t=t3, f=ch.array([f, f]),
                                c=ch.array([w, h]) / 2.,
                                k=ch.zeros(5))
    crn3.camera = ProjectPoints(v=V_row_bite, rt=rt3, t=t3, f=ch.array([f, f]),
                                c=ch.array([w, h]) / 2.,
                                k=ch.zeros(5))

    rn3.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    rn3.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    drn3.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    drn3.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    crn3.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    crn3.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc, bgcolor=ch.zeros(3), num_channels=3)

    # rn4 = BoundaryRenderer()
    # drn4 = DepthRenderer()
    # crn4 = ColoredRenderer()
    # rn4.camera = ProjectPoints(v=V_row_bite, rt=rt4, t=t4, f=ch.array([f, f]),
    #                            c=ch.array([w, h]) / 2.,
    #                            k=ch.zeros(5))
    # drn4.camera = ProjectPoints(v=V_row_bite, rt=rt4, t=t4, f=ch.array([f, f]),
    #                             c=ch.array([w, h]) / 2.,
    #                             k=ch.zeros(5))
    # crn4.camera = ProjectPoints(v=V_row_bite, rt=rt4, t=t4, f=ch.array([f, f]),
    #                             c=ch.array([w, h]) / 2.,
    #                             k=ch.zeros(5))
    #
    # rn4.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    # rn4.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    # drn4.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    # drn4.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    # crn4.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    # crn4.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc, bgcolor=ch.zeros(3), num_channels=3)

    # rn5 = BoundaryRenderer()
    # drn5 = DepthRenderer()
    # crn5 = ColoredRenderer()
    #
    # rn5.camera = ProjectPoints(v=V_row_bite, rt=rt5, t=t5, f=ch.array([f, f]),
    #                            c=ch.array([w, h]) / 2.,
    #                            k=ch.zeros(5))
    # drn5.camera = ProjectPoints(v=V_row_bite, rt=rt5, t=t5, f=ch.array([f, f]),
    #                             c=ch.array([w, h]) / 2.,
    #                             k=ch.zeros(5))
    # crn5.camera = ProjectPoints(v=V_row_bite, rt=rt5, t=t5, f=ch.array([f, f]),
    #                             c=ch.array([w, h]) / 2.,
    #                             k=ch.zeros(5))
    #
    # rn5.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    # rn5.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    # drn5.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    # drn5.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    # crn5.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    # crn5.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc, bgcolor=ch.zeros(3), num_channels=3)

    rn4 = BoundaryRenderer()
    drn4 = DepthRenderer()
    crn4 = ColoredRenderer()

    rn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))
    drn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                                c=ch.array([w, h]) / 2.,
                                k=ch.zeros(5))
    crn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                                c=ch.array([w, h]) / 2.,
                                k=ch.zeros(5))

    rn4.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    rn4.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    drn4.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    drn4.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    crn4.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    crn4.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc, bgcolor=ch.zeros(3), num_channels=3)

    rn5 = BoundaryRenderer()
    drn5 = DepthRenderer()
    crn5 = ColoredRenderer()

    rn5.camera = ProjectPoints(v=V_row_l, rt=rt5, t=t5, f=ch.array([f, f]),
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))
    drn5.camera = ProjectPoints(v=V_row_l, rt=rt5, t=t5, f=ch.array([f, f]),
                                c=ch.array([w, h]) / 2.,
                                k=ch.zeros(5))
    crn5.camera = ProjectPoints(v=V_row_l, rt=rt5, t=t5, f=ch.array([f, f]),
                                c=ch.array([w, h]) / 2.,
                                k=ch.zeros(5))

    rn5.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    rn5.set(v=V_row_l, f=row_mesh_l.f, vc=row_mesh_l.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    drn5.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    drn5.set(v=V_row_l, f=row_mesh_l.f, vc=row_mesh_l.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    crn5.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    crn5.set(v=V_row_l, f=row_mesh_l.f, vc=row_mesh_l.vc, bgcolor=ch.zeros(3), num_channels=3)


    if gt_list is not None:
        observed1 = load_image(gt_list[0])
        observed2 = load_image(gt_list[1])
        observed3 = load_image(gt_list[2])
        observed4 = load_image(gt_list[3])
        observed5 = load_image(gt_list[4])
    else:
        observed1 = load_image(img1_file_path)
        observed2 = load_image(img2_file_path)
        observed3 = load_image(img3_file_path)
        observed4 = load_image(img4_file_path)
        observed5 = load_image(img5_file_path)
    # observed6 = load_image(img6_file_path)
    # observed7 = load_image(img7_file_path)

    # observed1 = scipy.misc.imresize(obs1, 0.25)
    # observed2 = scipy.misc.imresize(obs2, 0.25)
    # observed3 = scipy.misc.imresize(obs3, 0.25)
    # observed4 = scipy.misc.imresize(obs4, 0.25)

    # observed1 = cv2.resize(obs1, (obs1.shape[1]/4, obs1.shape[0]/4))
    # observed2 = cv2.resize(obs2, (obs2.shape[1] / 4, obs2.shape[0] / 4))
    # observed3 = cv2.resize(obs3, (obs3.shape[1] / 4, obs3.shape[0] / 4))
    # observed4 = cv2.resize(obs4, (obs4.shape[1] / 4, obs4.shape[0] / 4))
    # observed5 = cv2.resize(obs5, (obs5.shape[1] / 4, obs5.shape[0] / 4))
    # observed6 = cv2.resize(obs6, (obs6.shape[1] / 4, obs6.shape[0] / 4))

    # draw preparation
    plt.ion()
    fig, axarr = None, None
    ob1_dc = deepcopy(observed1)
    ob2_dc = deepcopy(observed2)
    ob3_dc = deepcopy(observed3)
    ob4_dc = deepcopy(observed4)
    ob5_dc = deepcopy(observed5)
    # ob6_dc = deepcopy(observed6)
    # ob7_dc = deepcopy(observed7)
    ob1_dc[ob1_dc[:, :, 0] > 0] *= [0, 1, 0]
    ob2_dc[ob2_dc[:, :, 0] > 0] *= [0, 1, 0]
    ob3_dc[ob3_dc[:, :, 0] > 0] *= [0, 1, 0]
    ob4_dc[ob4_dc[:, :, 0] > 0] *= [0, 1, 0]
    ob5_dc[ob5_dc[:, :, 0] > 0] *= [0, 1, 0]
    # ob6_dc[ob6_dc[:, :, 0] > 0] *= [0, 1, 0]
    # ob7_dc[ob7_dc[:, :, 0] > 0] *= [0, 1, 0]

    start_time = time.time()
    total_time = 0

    # drawfig(0)

    # package the info
    obs = []
    obs.append(observed1)
    obs.append(observed2)
    obs.append(observed3)
    obs.append(observed4)
    obs.append(observed5)
    # obs.append(observed6)
    # obs.append(observed7)
    cb_err = [100000, 100000, 100000, 100000, 100000, 100000, 100000]
    for iter in range(3):
        rn_contours = []
        rn_contours.append(rn)
        rn_contours.append(rn2)
        rn_contours.append(rn3)
        rn_contours.append(rn4)
        rn_contours.append(rn5)
        # rn_contours.append(rn6)
        # rn_contours.append(rn7)
        rn_depths = []
        rn_depths.append(drn)
        rn_depths.append(drn2)
        rn_depths.append(drn3)
        rn_depths.append(drn4)
        rn_depths.append(drn5)
        # rn_depths.append(drn6)
        # rn_depths.append(drn7)
        rn_rs = []
        rn_rs.append(rt1)
        rn_rs.append(rt2)
        rn_rs.append(rt3)
        rn_rs.append(rt4)
        rn_rs.append(rt5)
        # rn_rs.append(rt6)
        # rn_rs.append(rt7)
        rn_ts = []
        rn_ts.append(t1)
        rn_ts.append(t2)
        rn_ts.append(t3)
        rn_ts.append(t4)
        rn_ts.append(t5)
        # rn_ts.append(t6)
        # rn_ts.append(t7)
        # Camera pose calibration
        for i in range(5):
            err = 100000
            err_dif = 100
            iter = 0
            mean = np.mean(V_row.r, axis=0)
            while not (err < 1 or err_dif < 0.1 or iter > 20):
                cur_time = time.time()
                sample_pts = get_sample_pts(rn_contours[i].r)
                print('sample point time: %s s' % (time.time() - cur_time))
                cur_time = time.time()
                # print "a", cur_time
                # d = rn_depths[i].r
                # print "a1", time.time()
                # intersection_pts, index_ray, index_tri = pj.back_projection(sample_pts, rn_rs[i], rn_ts[i], V_row, row_mesh.f)
                # pair_pts = get_pair_pts(obs[i], sample_pts, index_ray)
                sp_ins_pts = pj.back_projection_depth(w, h, f, sample_pts, rn_rs[i], rn_ts[i], rn_depths[i].r)
                print('back projection time: %s s' % (time.time() - cur_time))
                cur_time = time.time()
                pair_pts, trim_ins_pts = get_pair_pts(obs[i], sample_pts, sp_ins_pts)  # get pairing points
                print('pairing time: %s s' % (time.time() - cur_time))

                pars = Parameters()
                pars.add('rtx', value=0)
                pars.add('rty', value=0)
                pars.add('rtz', value=0)
                pars.add('tx', value=0)
                pars.add('ty', value=0)
                pars.add('tz', value=0)
                out = lmfit.minimize(residual_rtt, pars,
                                     args=(w, h, f, mean, trim_ins_pts, rn_rs[i], rn_ts[i], pair_pts), method='leastsq')

                totol_num = len(pair_pts)
                err_dif = cb_err[i] - out.chisqr / totol_num
                # cb_err[i] = out.chisqr
                if (err_dif > 0):
                    err = out.chisqr / totol_num
                    cb_err[i] = out.chisqr / totol_num
                    # out.params.pretty_print()
                    tmprt = np.array([out.params['rtx'], out.params['rty'], out.params['rtz']])
                    tmpt = np.array([out.params['tx'], out.params['ty'], out.params['tz']])
                    # print tmprt, tmpt

                    tc_rt, tc_t = reverse_camerapose(rn_rs[i], rn_ts[i], tmprt, tmpt, mean)
                    rn_rs[i] = ch.array(tc_rt)
                    rn_ts[i] = ch.array(tc_t)
                    if i < 3:
                        rn_contours[i].camera = ProjectPoints(v=V_row_bite, rt=rn_rs[i], t=rn_ts[i], f=ch.array([f, f]),
                                                              c=ch.array([w, h]) / 2.,
                                                              k=ch.zeros(5))
                        rn_depths[i].camera = ProjectPoints(v=V_row_bite, rt=rn_rs[i], t=rn_ts[i], f=ch.array([f, f]),
                                                            c=ch.array([w, h]) / 2.,
                                                            k=ch.zeros(5))
                    if i == 3:
                        rn_contours[i].camera = ProjectPoints(v=V_row, rt=rn_rs[i], t=rn_ts[i], f=ch.array([f, f]),
                                                              c=ch.array([w, h]) / 2.,
                                                              k=ch.zeros(5))
                        rn_depths[i].camera = ProjectPoints(v=V_row, rt=rn_rs[i], t=rn_ts[i], f=ch.array([f, f]),
                                                            c=ch.array([w, h]) / 2.,
                                                            k=ch.zeros(5))
                    if i == 4:
                        rn_contours[i].camera = ProjectPoints(v=V_row_l, rt=rn_rs[i], t=rn_ts[i], f=ch.array([f, f]),
                                                              c=ch.array([w, h]) / 2.,
                                                              k=ch.zeros(5))
                        rn_depths[i].camera = ProjectPoints(v=V_row_l, rt=rn_rs[i], t=rn_ts[i], f=ch.array([f, f]),
                                                            c=ch.array([w, h]) / 2.,
                                                            k=ch.zeros(5))

                print out.message, out.chisqr, out.chisqr / totol_num

            print("View: %d error: %f --- %s seconds ---" % (i, cb_err[i], time.time() - start_time))
            total_time += (time.time() - start_time)
            start_time = time.time()

        rt1 = rn_rs[0]
        t1 = rn_ts[0]
        rn.camera = ProjectPoints(v=V_row_bite, rt=rt1, t=t1, f=ch.array([f, f]),
                                  c=ch.array([w, h]) / 2.,
                                  k=ch.zeros(5))
        drn.camera = ProjectPoints(v=V_row_bite, rt=rt1, t=t1, f=ch.array([f, f]),
                                   c=ch.array([w, h]) / 2.,
                                   k=ch.zeros(5))
        crn.camera = ProjectPoints(v=V_row_bite, rt=rt1, t=t1, f=ch.array([f, f]),
                                   c=ch.array([w, h]) / 2.,
                                   k=ch.zeros(5))
        rt2 = rn_rs[1]
        t2 = rn_ts[1]
        rn2.camera = ProjectPoints(v=V_row_bite, rt=rt2, t=t2, f=ch.array([f, f]),
                                   c=ch.array([w, h]) / 2.,
                                   k=ch.zeros(5))
        drn2.camera = ProjectPoints(v=V_row_bite, rt=rt2, t=t2, f=ch.array([f, f]),
                                    c=ch.array([w, h]) / 2.,
                                    k=ch.zeros(5))
        crn2.camera = ProjectPoints(v=V_row_bite, rt=rt2, t=t2, f=ch.array([f, f]),
                                    c=ch.array([w, h]) / 2.,
                                    k=ch.zeros(5))
        rt3 = rn_rs[2]
        t3 = rn_ts[2]
        rn3.camera = ProjectPoints(v=V_row_bite, rt=rt3, t=t3, f=ch.array([f, f]),
                                   c=ch.array([w, h]) / 2.,
                                   k=ch.zeros(5))
        drn3.camera = ProjectPoints(v=V_row_bite, rt=rt3, t=t3, f=ch.array([f, f]),
                                    c=ch.array([w, h]) / 2.,
                                    k=ch.zeros(5))
        crn3.camera = ProjectPoints(v=V_row_bite, rt=rt3, t=t3, f=ch.array([f, f]),
                                    c=ch.array([w, h]) / 2.,
                                    k=ch.zeros(5))
        # rt4 = rn_rs[3]
        # t4 = rn_ts[3]
        # rn4.camera = ProjectPoints(v=V_row_bite, rt=rt4, t=t4, f=ch.array([f, f]),
        #                            c=ch.array([w, h]) / 2.,
        #                            k=ch.zeros(5))
        # drn4.camera = ProjectPoints(v=V_row_bite, rt=rt4, t=t4, f=ch.array([f, f]),
        #                             c=ch.array([w, h]) / 2.,
        #                             k=ch.zeros(5))
        # crn4.camera = ProjectPoints(v=V_row_bite, rt=rt4, t=t4, f=ch.array([f, f]),
        #                             c=ch.array([w, h]) / 2.,
        #                             k=ch.zeros(5))
        # rt5 = rn_rs[4]
        # t5 = rn_ts[4]
        # rn5.camera = ProjectPoints(v=V_row_bite, rt=rt5, t=t5, f=ch.array([f, f]),
        #                            c=ch.array([w, h]) / 2.,
        #                            k=ch.zeros(5))
        # drn5.camera = ProjectPoints(v=V_row_bite, rt=rt5, t=t5, f=ch.array([f, f]),
        #                             c=ch.array([w, h]) / 2.,
        #                             k=ch.zeros(5))
        # crn5.camera = ProjectPoints(v=V_row_bite, rt=rt5, t=t5, f=ch.array([f, f]),
        #                             c=ch.array([w, h]) / 2.,
        #                             k=ch.zeros(5))
        rt4 = rn_rs[3]
        t4 = rn_ts[3]
        rn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                                   c=ch.array([w, h]) / 2.,
                                   k=ch.zeros(5))
        drn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                                    c=ch.array([w, h]) / 2.,
                                    k=ch.zeros(5))
        crn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                                    c=ch.array([w, h]) / 2.,
                                    k=ch.zeros(5))
        rt5 = rn_rs[4]
        t5 = rn_ts[4]
        rn5.camera = ProjectPoints(v=V_row_l, rt=rt5, t=t5, f=ch.array([f, f]),
                                   c=ch.array([w, h]) / 2.,
                                   k=ch.zeros(5))
        drn5.camera = ProjectPoints(v=V_row_l, rt=rt5, t=t5, f=ch.array([f, f]),
                                    c=ch.array([w, h]) / 2.,
                                    k=ch.zeros(5))
        crn5.camera = ProjectPoints(v=V_row_l, rt=rt5, t=t5, f=ch.array([f, f]),
                                    c=ch.array([w, h]) / 2.,
                                    k=ch.zeros(5))

        # individual tooth pose estimation
        for i in range(numTooth+numTooth_l):
            err = 100000
            err_dif = 100
            iter = 0
            part = 0
            molars = []
			# set molars[0] = 100 means no molars fixed
			molars.append(100)
			# set fixed molars ids
            # molars.append(numTooth/2 - 2)
            # molars.append(numTooth/2 - 1)
            # molars.append(numTooth - 2)
            # molars.append(numTooth - 1)
            # molars.append(numTooth + numTooth_l/2 - 2)
            # molars.append(numTooth + numTooth_l/2 - 1)
            # molars.append(numTooth + numTooth_l - 2)
            # molars.append(numTooth + numTooth_l - 1)

            #fixed molars
            if i == molars[0]:
                while not (err < 1 or err_dif < 0.1 or iter > 20):
                    pair_pts1_m = []
                    trim_ins_pts1_m = []
                    pair_pts2_m = []
                    trim_ins_pts2_m = []
                    pair_pts3_m = []
                    trim_ins_pts3_m = []
                    pair_pts4_m = []
                    trim_ins_pts4_m = []
                    pair_pts5_m = []
                    trim_ins_pts5_m = []
                    V_molars = []
                    for j in molars:
                        if j < numTooth:
                            V_molars.append(Vi_list[j])
                        else:
                            V_molars.append(Vi_list_l[j-numTooth])

                        curs_time = time.time()
                        sample_pts1 = get_sample_pts(rn.r, crn.r, j)  # first time using intial rn
                        print('sample point time: %s s' % (time.time() - curs_time))
                        cur_time = time.time()
                        if len(sample_pts1) < 2:
                            pair_pts1 = []
                            trim_ins_pts1 = []
                        else:
                            # get back projection 3D verts and id of 2D points which can find corresponding verts
                            sp_ins_pts1 = pj.back_projection_depth(w, h, f, sample_pts1, rt1, t1, drn.r)
                            print('back projection time: %s s' % (time.time() - cur_time))
                            # if iter == 0:
                            #     Mesh.save_to_obj('result/tooth{}_vertices.obj'.format(i), sp_ins_pts1)
                            cur_time = time.time()
                            pair_pts1, trim_ins_pts1 = get_pair_pts(observed1, sample_pts1,
                                                                    sp_ins_pts1)  # get pairing points
                            pair_pts1_m.extend(pair_pts1)
                            trim_ins_pts1_m.extend(trim_ins_pts1)
                            print('pairing time: %s s' % (time.time() - cur_time))

                        sample_pts2 = get_sample_pts(rn2.r, crn2.r, j)
                        if len(sample_pts2) < 2:
                            pair_pts2 = []
                            trim_ins_pts2 = []
                        else:
                            sp_ins_pts2 = pj.back_projection_depth(w, h, f, sample_pts2, rt2, t2, drn2.r)
                            pair_pts2, trim_ins_pts2 = get_pair_pts(observed2, sample_pts2, sp_ins_pts2)
                            pair_pts2_m.extend(pair_pts2)
                            trim_ins_pts2_m.extend(trim_ins_pts2)

                        sample_pts3 = get_sample_pts(rn3.r, crn3.r, j)
                        if len(sample_pts3) < 2:
                            pair_pts3 = []
                            trim_ins_pts3 = []
                        else:
                            sp_ins_pts3 = pj.back_projection_depth(w, h, f, sample_pts3, rt3, t3, drn3.r)
                            pair_pts3, trim_ins_pts3 = get_pair_pts(observed3, sample_pts3, sp_ins_pts3)
                            pair_pts3_m.extend(pair_pts3)
                            trim_ins_pts3_m.extend(trim_ins_pts3)

                        sample_pts4 = get_sample_pts(rn4.r, crn4.r, j)
                        if len(sample_pts4) < 2:
                            pair_pts4 = []
                            trim_ins_pts4 = []
                        else:
                            sp_ins_pts4 = pj.back_projection_depth(w, h, f, sample_pts4, rt4, t4, drn4.r)
                            pair_pts4, trim_ins_pts4 = get_pair_pts(observed4, sample_pts4, sp_ins_pts4)
                            pair_pts4_m.extend(pair_pts4)
                            trim_ins_pts4_m.extend(trim_ins_pts4)

                        sample_pts5 = get_sample_pts(rn5.r, crn5.r, j)
                        if len(sample_pts5) < 2:
                            pair_pts5 = []
                            trim_ins_pts5 = []
                        else:
                            sp_ins_pts5 = pj.back_projection_depth(w, h, f, sample_pts5, rt5, t5, drn5.r)
                            pair_pts5, trim_ins_pts5 = get_pair_pts(observed5, sample_pts5, sp_ins_pts5)
                            pair_pts5_m.extend(pair_pts5)
                            trim_ins_pts5_m.extend(trim_ins_pts5)

                    V_molars = ch.vstack(V_molars)
                    mean = np.mean(V_molars.r, axis=0)
                    print mean
                    # print pair_pts1_m.shape

                    cur_time = time.time()

                    # pars = np.array([0, 0, 0, 0, 0, 0], dtype='float32')
                    pars = Parameters()
                    pars.add('rtx', value=0)
                    pars.add('rty', value=0)
                    pars.add('rtz', value=0)
                    pars.add('tx', value=0)
                    pars.add('ty', value=0)
                    pars.add('tz', value=0)
                    out = lmfit.minimize(residual_rtt_allview, pars,
                                         args=(
                                         w, h, f, mean, trim_ins_pts1_m, trim_ins_pts2_m, trim_ins_pts3_m, trim_ins_pts4_m,
                                         trim_ins_pts5_m, rt1, t1, rt2, t2, rt3, t3, rt4, t4, rt5, t5,
                                         pair_pts1_m, pair_pts2_m, pair_pts3_m, pair_pts4_m, pair_pts5_m),
                                         method='leastsq')
                    print('optimization time: %s s' % (time.time() - cur_time))

                    total_pts = len(pair_pts1) + len(pair_pts2) + len(pair_pts3) + len(pair_pts4) + len(pair_pts5)
                    cur_time = time.time()
                    err_dif = err - out.chisqr / total_pts
                    if (err_dif > 0):
                        err = out.chisqr / total_pts
                        # out.params.pretty_print()
                        tmprt = np.array([out.params['rtx'], out.params['rty'], out.params['rtz']])
                        tmpt = np.array([out.params['tx'], out.params['ty'], out.params['tz']])
                        print tmprt, tmpt

                        for j in molars:
                            if j < numTooth:
                                teeth_row_mesh.rotate(tmprt, j, mean)
                                teeth_row_mesh.translate(tmpt, j)
                                # Vi_list[i] = (Vi_list[i] - mean).dot(Rodrigues(tmprt)) + mean + tmpt
                                # V_row = (V_row-mean).dot(Rodrigues(tmprt)) + mean + tmpt
                            else:
                                teeth_row_mesh_l.rotate(tmprt, j - numTooth, mean)
                                teeth_row_mesh_l.translate(tmpt, j - numTooth)
                                # Vi_list_l[i-numTooth] = (Vi_list_l[i-numTooth] - mean).dot(Rodrigues(tmprt)) + mean + tmpt
                        Vi_list = [ch.array(teeth_row_mesh.mesh_list[k].v) for k in range(numTooth)]
                        V_row = ch.vstack([Vi_list[k] for k in range(numTooth)])
                        Vi_list_l = [ch.array(teeth_row_mesh_l.mesh_list[k].v) for k in range(numTooth_l)]
                        V_row_l = ch.vstack([Vi_list_l[k] for k in range(numTooth_l)])
                        merged_mesh = Mesh.merge_mesh(row_mesh, row_mesh_l)
                        V_row_bite = ch.array(merged_mesh.v)

                    print out.message, out.chisqr, out.chisqr / total_pts

                    # reproject 2D contour
                    rn.camera = ProjectPoints(v=V_row_bite, rt=rt1, t=t1, f=ch.array([f, f]),
                                              c=ch.array([w, h]) / 2.,
                                              k=ch.zeros(5))
                    rn2.camera = ProjectPoints(v=V_row_bite, rt=rt2, t=t2, f=ch.array([f, f]),
                                               c=ch.array([w, h]) / 2.,
                                               k=ch.zeros(5))
                    rn3.camera = ProjectPoints(v=V_row_bite, rt=rt3, t=t3, f=ch.array([f, f]),
                                               c=ch.array([w, h]) / 2.,
                                               k=ch.zeros(5))
                    # rn4.camera = ProjectPoints(v=V_row_bite, rt=rt4, t=t4, f=ch.array([f, f]),
                    #                            c=ch.array([w, h]) / 2.,
                    #                            k=ch.zeros(5))
                    # rn5.camera = ProjectPoints(v=V_row_bite, rt=rt5, t=t5, f=ch.array([f, f]),
                    #                            c=ch.array([w, h]) / 2.,
                    #                            k=ch.zeros(5))
                    rn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                                               c=ch.array([w, h]) / 2.,
                                               k=ch.zeros(5))
                    rn5.camera = ProjectPoints(v=V_row_l, rt=rt5, t=t5, f=ch.array([f, f]),
                                               c=ch.array([w, h]) / 2.,
                                               k=ch.zeros(5))
                    drn.camera = ProjectPoints(v=V_row_bite, rt=rt1, t=t1, f=ch.array([f, f]),
                                               c=ch.array([w, h]) / 2.,
                                               k=ch.zeros(5))
                    drn2.camera = ProjectPoints(v=V_row_bite, rt=rt2, t=t2, f=ch.array([f, f]),
                                                c=ch.array([w, h]) / 2.,
                                                k=ch.zeros(5))
                    drn3.camera = ProjectPoints(v=V_row_bite, rt=rt3, t=t3, f=ch.array([f, f]),
                                                c=ch.array([w, h]) / 2.,
                                                k=ch.zeros(5))
                    # drn4.camera = ProjectPoints(v=V_row_bite, rt=rt4, t=t4, f=ch.array([f, f]),
                    #                             c=ch.array([w, h]) / 2.,
                    #                             k=ch.zeros(5))
                    # drn5.camera = ProjectPoints(v=V_row_bite, rt=rt5, t=t5, f=ch.array([f, f]),
                    #                             c=ch.array([w, h]) / 2.,
                    #                             k=ch.zeros(5))
                    drn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                                                c=ch.array([w, h]) / 2.,
                                                k=ch.zeros(5))
                    drn5.camera = ProjectPoints(v=V_row_l, rt=rt5, t=t5, f=ch.array([f, f]),
                                                c=ch.array([w, h]) / 2.,
                                                k=ch.zeros(5))
                    crn.camera = ProjectPoints(v=V_row_bite, rt=rt1, t=t1, f=ch.array([f, f]),
                                               c=ch.array([w, h]) / 2.,
                                               k=ch.zeros(5))
                    crn2.camera = ProjectPoints(v=V_row_bite, rt=rt2, t=t2, f=ch.array([f, f]),
                                                c=ch.array([w, h]) / 2.,
                                                k=ch.zeros(5))
                    crn3.camera = ProjectPoints(v=V_row_bite, rt=rt3, t=t3, f=ch.array([f, f]),
                                                c=ch.array([w, h]) / 2.,
                                                k=ch.zeros(5))
                    # crn4.camera = ProjectPoints(v=V_row_bite, rt=rt4, t=t4, f=ch.array([f, f]),
                    #                             c=ch.array([w, h]) / 2.,
                    #                             k=ch.zeros(5))
                    # crn5.camera = ProjectPoints(v=V_row_bite, rt=rt5, t=t5, f=ch.array([f, f]),
                    #                             c=ch.array([w, h]) / 2.,
                    #                             k=ch.zeros(5))
                    crn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                                                c=ch.array([w, h]) / 2.,
                                                k=ch.zeros(5))
                    crn5.camera = ProjectPoints(v=V_row_l, rt=rt5, t=t5, f=ch.array([f, f]),
                                                c=ch.array([w, h]) / 2.,
                                                k=ch.zeros(5))
                    print('rerendering time: %s s' % (time.time() - cur_time))
                    if (iter == 0):
                        print('one step time: %s s' % (time.time() - curs_time))
                    # Mesh.save_to_obj('result/V_row_op.obj', V_row, row_mesh.f)
                    # rn_c2 = deepcopy(rn.r)
                    # rn_c2[rn_c2[:, :, 0] > 0] *= [0, 1, 0]
                    #
                    # plt.imshow(rn_c2 + rn_c)
                    # plt.show()
                    # break
                    iter += 1

                continue
            else:
                if i in molars:
                    continue

            # print V_row.shape
            # cur_tooth = V_row[teeth_row_mesh.start_idx_list[i]:teeth_row_mesh.start_idx_list[i+1], 0:3]
            # print cur_tooth.shape
            while not (err < 1 or err_dif < 0.1 or iter > 20):
                if i < numTooth:
                    mean = np.mean(Vi_list[i].r, axis=0)
                else:
                    mean = np.mean(Vi_list_l[i-numTooth].r, axis=0)
                    part = 1
                # print mean
                curs_time = time.time()
                sample_pts1 = get_sample_pts(rn.r, crn.r, i)  # first time using intial rn
                print('sample point time: %s s' % (time.time() - curs_time))
                cur_time = time.time()
                if len(sample_pts1) < 2:
                    pair_pts1 = []
                    trim_ins_pts1 = []
                else:
                    # get back projection 3D verts and id of 2D points which can find corresponding verts
                    sp_ins_pts1 = pj.back_projection_depth(w, h, f, sample_pts1, rt1, t1, drn.r)
                    print('back projection time: %s s' % (time.time() - cur_time))
                    # if iter == 0:
                    #     Mesh.save_to_obj('result/tooth{}_vertices.obj'.format(i), sp_ins_pts1)
                    cur_time = time.time()
                    pair_pts1, trim_ins_pts1 = get_pair_pts(observed1, sample_pts1, sp_ins_pts1)  # get pairing points
                    print('pairing time: %s s' % (time.time() - cur_time))

                # intersection_pts1, index_ray1, index_tri1 = pj.back_projection(sample_pts1, rt1, t1, V_row, row_mesh.f)
                # get separate tooth's corresponding verts and pair points
                # sp_ins_pts1 = []
                # sp_index_ray1 = []
                # cc = 0
                # for j in range(index_tri1.size):
                #     if teeth_row_mesh.faces_num[i] <= index_tri1[j] and index_tri1[j] < teeth_row_mesh.faces_num[i+1]:
                #         sp_ins_pts1.append(intersection_pts1[j])
                #         sp_index_ray1.append(index_ray1[j])
                #         cc += 1
                # if cc < 2:
                #     pair_pts1 = []
                # else:
                #     sp_ins_pts1 = np.vstack(sp_ins_pts1)
                #     sp_index_ray1 = np.squeeze(sp_index_ray1)
                #     pair_pts1 = get_pair_pts(observed1, sample_pts1, sp_index_ray1) #find pair points (only those getting 3D verts)

                sample_pts2 = get_sample_pts(rn2.r, crn2.r, i)
                if len(sample_pts2) < 2:
                    pair_pts2 = []
                    trim_ins_pts2 = []
                else:
                    sp_ins_pts2 = pj.back_projection_depth(w, h, f, sample_pts2, rt2, t2, drn2.r)
                    pair_pts2, trim_ins_pts2 = get_pair_pts(observed2, sample_pts2, sp_ins_pts2)

                # intersection_pts2, index_ray2, index_tri2 = pj.back_projection(sample_pts2, rt2, t2, V_row, row_mesh.f)
                # sp_ins_pts2 = []
                # sp_index_ray2 = []
                # cc = 0
                # for j in range(index_tri2.size):
                #     if teeth_row_mesh.faces_num[i] <= index_tri2[j] and index_tri2[j] < teeth_row_mesh.faces_num[i + 1]:
                #         sp_ins_pts2.append(intersection_pts2[j])
                #         sp_index_ray2.append(index_ray2[j])
                #         cc += 1
                # if cc < 2:
                #     pair_pts2 = []
                # else:
                #     sp_ins_pts2 = np.vstack(sp_ins_pts2)
                #     sp_index_ray2 = np.squeeze(sp_index_ray2)
                #     pair_pts2 = get_pair_pts(observed2, sample_pts2, sp_index_ray2)

                sample_pts3 = get_sample_pts(rn3.r, crn3.r, i)
                if len(sample_pts3) < 2:
                    pair_pts3 = []
                    trim_ins_pts3 = []
                else:
                    sp_ins_pts3 = pj.back_projection_depth(w, h, f, sample_pts3, rt3, t3, drn3.r)
                    pair_pts3, trim_ins_pts3 = get_pair_pts(observed3, sample_pts3, sp_ins_pts3)

                # intersection_pts3, index_ray3, index_tri3 = pj.back_projection(sample_pts3, rt3, t3, V_row, row_mesh.f)
                # sp_ins_pts3 = []
                # sp_index_ray3 = []
                # cc = 0
                # for j in range(index_tri3.size):
                #     if teeth_row_mesh.faces_num[i] <= index_tri3[j] and index_tri3[j] < teeth_row_mesh.faces_num[i + 1]:
                #         sp_ins_pts3.append(intersection_pts3[j])
                #         sp_index_ray3.append(index_ray3[j])
                #         cc += 1
                # if cc < 2:
                #     pair_pts3 = []
                # else:
                #     sp_ins_pts3 = np.vstack(sp_ins_pts3)
                #     sp_index_ray3 = np.squeeze(sp_index_ray3)
                #     pair_pts3 = get_pair_pts(observed3, sample_pts3, sp_index_ray3)

                sample_pts4 = get_sample_pts(rn4.r, crn4.r, i)
                if len(sample_pts4) < 2:
                    pair_pts4 = []
                    trim_ins_pts4 = []
                else:
                    sp_ins_pts4 = pj.back_projection_depth(w, h, f, sample_pts4, rt4, t4, drn4.r)
                    pair_pts4, trim_ins_pts4 = get_pair_pts(observed4, sample_pts4, sp_ins_pts4)

                # intersection_pts4, index_ray4, index_tri4 = pj.back_projection(sample_pts4, rt4, t4, V_row, row_mesh.f)
                # sp_ins_pts4 = []
                # sp_index_ray4 = []
                # cc = 0
                # for j in range(index_tri4.size):
                #     if teeth_row_mesh.faces_num[i] <= index_tri4[j] and index_tri4[j] < teeth_row_mesh.faces_num[i + 1]:
                #         sp_ins_pts4.append(intersection_pts4[j])
                #         sp_index_ray4.append(index_ray4[j])
                #         cc += 1
                # if cc <2:
                #     pair_pts4 = []
                # else:
                #     # print cc
                #     sp_ins_pts4 = np.vstack(sp_ins_pts4)
                #     sp_index_ray4 = np.squeeze(sp_index_ray4)
                #     # print sp_index_ray4.shape
                #     pair_pts4 = get_pair_pts(observed4, sample_pts4, sp_index_ray4)

                sample_pts5 = get_sample_pts(rn5.r, crn5.r, i)
                if len(sample_pts5) < 2:
                    pair_pts5 = []
                    trim_ins_pts5 = []
                else:
                    sp_ins_pts5 = pj.back_projection_depth(w, h, f, sample_pts5, rt5, t5, drn5.r)
                    pair_pts5, trim_ins_pts5 = get_pair_pts(observed5, sample_pts5, sp_ins_pts5)

                # sample_pts6 = get_sample_pts(rn6.r, crn6.r, i)
                # if len(sample_pts6) < 2:
                #     pair_pts6 = []
                #     trim_ins_pts6 = []
                # else:
                #     sp_ins_pts6 = pj.back_projection_depth(w, h, f, sample_pts6, rt6, t6, drn6.r)
                #     pair_pts6, trim_ins_pts6 = get_pair_pts(observed6, sample_pts6, sp_ins_pts6)
                #
                # sample_pts7 = get_sample_pts(rn7.r, crn7.r, i)
                # if len(sample_pts7) < 2:
                #     pair_pts7 = []
                #     trim_ins_pts7 = []
                # else:
                #     sp_ins_pts7 = pj.back_projection_depth(w, h, f, sample_pts7, rt7, t7, drn7.r)
                #     pair_pts7, trim_ins_pts7 = get_pair_pts(observed7, sample_pts7, sp_ins_pts7)

                cur_time = time.time()

                # pars = np.array([0, 0, 0, 0, 0, 0], dtype='float32')
                pars = Parameters()
                pars.add('rtx', value=0)
                pars.add('rty', value=0)
                pars.add('rtz', value=0)
                pars.add('tx', value=0)
                pars.add('ty', value=0)
                pars.add('tz', value=0)
                out = lmfit.minimize(residual_rtt_allview, pars,
                                     args=(w, h, f, mean, trim_ins_pts1, trim_ins_pts2, trim_ins_pts3, trim_ins_pts4,
                                           trim_ins_pts5, rt1, t1, rt2, t2, rt3, t3, rt4, t4, rt5, t5,
                                            pair_pts1, pair_pts2, pair_pts3, pair_pts4, pair_pts5),
                                     method='leastsq')
                print('optimization time: %s s' % (time.time() - cur_time))

                total_pts = len(pair_pts1) + len(pair_pts2) + len(pair_pts3) + len(pair_pts4) + len(pair_pts5)
                cur_time = time.time()
                err_dif = err - out.chisqr / total_pts
                if (err_dif > 0):
                    err = out.chisqr / total_pts
                    # out.params.pretty_print()
                    tmprt = np.array([out.params['rtx'], out.params['rty'], out.params['rtz']])
                    tmpt = np.array([out.params['tx'], out.params['ty'], out.params['tz']])
                    print tmprt, tmpt

                    if part == 0:
                        teeth_row_mesh.rotate(tmprt, i)
                        teeth_row_mesh.translate(tmpt, i)
                        # Vi_list[i] = (Vi_list[i] - mean).dot(Rodrigues(tmprt)) + mean + tmpt
                        Vi_list = [ch.array(teeth_row_mesh.mesh_list[k].v) for k in range(numTooth)]
                        V_row = ch.vstack([Vi_list[k] for k in range(numTooth)])
                        # V_row = (V_row-mean).dot(Rodrigues(tmprt)) + mean + tmpt
                    else:
                        teeth_row_mesh_l.rotate(tmprt, i-numTooth)
                        teeth_row_mesh_l.translate(tmpt, i-numTooth)
                        # Vi_list_l[i-numTooth] = (Vi_list_l[i-numTooth] - mean).dot(Rodrigues(tmprt)) + mean + tmpt
                        Vi_list_l = [ch.array(teeth_row_mesh_l.mesh_list[k].v) for k in range(numTooth_l)]
                        V_row_l = ch.vstack([Vi_list_l[k] for k in range(numTooth_l)])
                    merged_mesh = Mesh.merge_mesh(row_mesh, row_mesh_l)
                    V_row_bite = ch.array(merged_mesh.v)

                print out.message, out.chisqr, out.chisqr / total_pts

                # reproject 2D contour
                rn.camera = ProjectPoints(v=V_row_bite, rt=rt1, t=t1, f=ch.array([f, f]),
                                          c=ch.array([w, h]) / 2.,
                                          k=ch.zeros(5))
                rn2.camera = ProjectPoints(v=V_row_bite, rt=rt2, t=t2, f=ch.array([f, f]),
                                           c=ch.array([w, h]) / 2.,
                                           k=ch.zeros(5))
                rn3.camera = ProjectPoints(v=V_row_bite, rt=rt3, t=t3, f=ch.array([f, f]),
                                           c=ch.array([w, h]) / 2.,
                                           k=ch.zeros(5))
                # rn4.camera = ProjectPoints(v=V_row_bite, rt=rt4, t=t4, f=ch.array([f, f]),
                #                            c=ch.array([w, h]) / 2.,
                #                            k=ch.zeros(5))
                # rn5.camera = ProjectPoints(v=V_row_bite, rt=rt5, t=t5, f=ch.array([f, f]),
                #                            c=ch.array([w, h]) / 2.,
                #                            k=ch.zeros(5))
                rn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                                           c=ch.array([w, h]) / 2.,
                                           k=ch.zeros(5))
                rn5.camera = ProjectPoints(v=V_row_l, rt=rt5, t=t5, f=ch.array([f, f]),
                                           c=ch.array([w, h]) / 2.,
                                           k=ch.zeros(5))
                drn.camera = ProjectPoints(v=V_row_bite, rt=rt1, t=t1, f=ch.array([f, f]),
                                           c=ch.array([w, h]) / 2.,
                                           k=ch.zeros(5))
                drn2.camera = ProjectPoints(v=V_row_bite, rt=rt2, t=t2, f=ch.array([f, f]),
                                            c=ch.array([w, h]) / 2.,
                                            k=ch.zeros(5))
                drn3.camera = ProjectPoints(v=V_row_bite, rt=rt3, t=t3, f=ch.array([f, f]),
                                            c=ch.array([w, h]) / 2.,
                                            k=ch.zeros(5))
                # drn4.camera = ProjectPoints(v=V_row_bite, rt=rt4, t=t4, f=ch.array([f, f]),
                #                             c=ch.array([w, h]) / 2.,
                #                             k=ch.zeros(5))
                # drn5.camera = ProjectPoints(v=V_row_bite, rt=rt5, t=t5, f=ch.array([f, f]),
                #                             c=ch.array([w, h]) / 2.,
                #                             k=ch.zeros(5))
                drn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                                            c=ch.array([w, h]) / 2.,
                                            k=ch.zeros(5))
                drn5.camera = ProjectPoints(v=V_row_l, rt=rt5, t=t5, f=ch.array([f, f]),
                                            c=ch.array([w, h]) / 2.,
                                            k=ch.zeros(5))
                crn.camera = ProjectPoints(v=V_row_bite, rt=rt1, t=t1, f=ch.array([f, f]),
                                           c=ch.array([w, h]) / 2.,
                                           k=ch.zeros(5))
                crn2.camera = ProjectPoints(v=V_row_bite, rt=rt2, t=t2, f=ch.array([f, f]),
                                            c=ch.array([w, h]) / 2.,
                                            k=ch.zeros(5))
                crn3.camera = ProjectPoints(v=V_row_bite, rt=rt3, t=t3, f=ch.array([f, f]),
                                            c=ch.array([w, h]) / 2.,
                                            k=ch.zeros(5))
                # crn4.camera = ProjectPoints(v=V_row_bite, rt=rt4, t=t4, f=ch.array([f, f]),
                #                             c=ch.array([w, h]) / 2.,
                #                             k=ch.zeros(5))
                # crn5.camera = ProjectPoints(v=V_row_bite, rt=rt5, t=t5, f=ch.array([f, f]),
                #                             c=ch.array([w, h]) / 2.,
                #                             k=ch.zeros(5))
                crn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([f, f]),
                                            c=ch.array([w, h]) / 2.,
                                            k=ch.zeros(5))
                crn5.camera = ProjectPoints(v=V_row_l, rt=rt5, t=t5, f=ch.array([f, f]),
                                            c=ch.array([w, h]) / 2.,
                                            k=ch.zeros(5))
                print('rerendering time: %s s' % (time.time() - cur_time))
                if (iter == 0):
                    print('one step time: %s s' % (time.time() - curs_time))
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
            rn5_dc = deepcopy(rn5.r)
            # rn6_dc = deepcopy(rn6.r)
            # rn7_dc = deepcopy(rn7.r)

            rn1_dc[rn1_dc[:, :, 0] > 0] *= [1, 0, 0]
            rn2_dc[rn2_dc[:, :, 0] > 0] *= [1, 0, 0]
            rn3_dc[rn3_dc[:, :, 0] > 0] *= [1, 0, 0]
            rn4_dc[rn4_dc[:, :, 0] > 0] *= [1, 0, 0]
            rn5_dc[rn5_dc[:, :, 0] > 0] *= [1, 0, 0]
            # rn6_dc[rn6_dc[:, :, 0] > 0] *= [1, 0, 0]
            # rn7_dc[rn7_dc[:, :, 0] > 0] *= [1, 0, 0]

            axarr[0, 0].imshow(rn1_dc + ob1_dc)
            axarr[0, 1].imshow(rn2_dc + ob2_dc)
            # axarr[0, 2].imshow(rn4_dc + ob4_dc)
            axarr[1, 0].imshow(rn4_dc + ob4_dc)
            axarr[1, 1].imshow(rn5_dc + ob5_dc)
            # axarr[1, 2].imshow(rn7_dc + ob7_dc)

            scipy.misc.imsave('result/log_bite/fittingresult1_iter{}.jpg'.format(i), rn1_dc + ob1_dc)
            scipy.misc.imsave('result/log_bite/fittingresult1_iter{}a.jpg'.format(i), rn1_dc)
            scipy.misc.imsave('result/log_bite/fittingresult1_iter{}b.jpg'.format(i), ob1_dc)
            scipy.misc.imsave('result/log_bite/fittingresult2_iter{}.jpg'.format(i), rn2_dc + ob2_dc)
            scipy.misc.imsave('result/log_bite/fittingresult2_iter{}a.jpg'.format(i), rn2_dc)
            scipy.misc.imsave('result/log_bite/fittingresult2_iter{}b.jpg'.format(i), ob2_dc)
            scipy.misc.imsave('result/log_bite/fittingresult3_iter{}.jpg'.format(i), rn3_dc + ob3_dc)
            scipy.misc.imsave('result/log_bite/fittingresult3_iter{}a.jpg'.format(i), rn3_dc)
            scipy.misc.imsave('result/log_bite/fittingresult3_iter{}b.jpg'.format(i), ob3_dc)
            scipy.misc.imsave('result/log_bite/fittingresult4_iter{}.jpg'.format(i), rn4_dc + ob4_dc)
            scipy.misc.imsave('result/log_bite/fittingresult4_iter{}a.jpg'.format(i), rn4_dc)
            scipy.misc.imsave('result/log_bite/fittingresult4_iter{}b.jpg'.format(i), ob4_dc)
            scipy.misc.imsave('result/log_bite/fittingresult5_iter{}.jpg'.format(i), rn5_dc + ob5_dc)
            scipy.misc.imsave('result/log_bite/fittingresult5_iter{}a.jpg'.format(i), rn5_dc)
            scipy.misc.imsave('result/log_bite/fittingresult5_iter{}b.jpg'.format(i), ob5_dc)
            # scipy.misc.imsave('result/log_bite/fittingresult6_iter{}.jpg'.format(i), rn6_dc + ob6_dc)
            # scipy.misc.imsave('result/log_bite/fittingresult6_iter{}a.jpg'.format(i), rn6_dc)
            # scipy.misc.imsave('result/log_bite/fittingresult6_iter{}b.jpg'.format(i), ob6_dc)
            # scipy.misc.imsave('result/log_bite/fittingresult7_iter{}.jpg'.format(i), rn7_dc + ob7_dc)
            # scipy.misc.imsave('result/log_bite/fittingresult7_iter{}a.jpg'.format(i), rn7_dc)
            # scipy.misc.imsave('result/log_bite/fittingresult7_iter{}b.jpg'.format(i), ob7_dc)

            print("tooth id: %d error: %f --- %s seconds ---" % (i, err, time.time() - start_time))
            total_time += (time.time() - start_time)
            start_time = time.time()

            plt.pause(2)

    print("total time --- %s seconds ---" % (total_time))
    Mesh.save_to_obj('result/bite/V_row_bite_u2.obj', V_row, row_mesh.f)
    Mesh.save_to_obj('result/bite/V_row_bite_l2.obj', V_row_l, row_mesh_l.f)
    Mesh.save_to_obj('result/bite/V_row_bite2.obj', V_row_bite, merged_mesh.f)

    for i in range(numTooth+numTooth_l):
        if i < numTooth:
            Mesh.save_to_obj('result/individual_tooth_bite/V_row{}.obj'.format(i), Vi_list[i], teeth_row_mesh.mesh_list[i].f)
        else:
            Mesh.save_to_obj('result/individual_tooth_bite/V_row{}.obj'.format(i), Vi_list_l[i-numTooth],
                             teeth_row_mesh_l.mesh_list[i-numTooth].f)
