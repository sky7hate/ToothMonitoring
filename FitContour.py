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
from scipy.spatial.transform import Rotation as R
import cv2
from config import cfg
import os

def get_sample_pts(contour):
    #decrease the size
    contour1 = scipy.misc.imresize(contour, 0.5)
    #Sample points
    t_sample_pts = []
    f_sample_pts = []
    # print contour1.size

    cc = 0
    index = 0
    for i in range(320):
        for j in range(240):
            if contour1[j, i, 0] > 0:
                t_sample_pts.append([j, i])
                index += 1
            cc += 1
            # if (i == 0) | (j == 0) | (i == 639) | (j == 479):
            #     frame.append([j, i])
    for i in range(index):
        if contour[t_sample_pts[i][0]*2, t_sample_pts[i][1]*2, 0] > 0:
            f_sample_pts.append([t_sample_pts[i][0]*2, t_sample_pts[i][1]*2])
        elif contour[t_sample_pts[i][0]*2, t_sample_pts[i][1]*2+1, 0] > 0:
            f_sample_pts.append([t_sample_pts[i][0]*2, t_sample_pts[i][1]*2+1])
        elif contour[t_sample_pts[i][0]*2+1, t_sample_pts[i][1]*2, 0] > 0:
            f_sample_pts.append([t_sample_pts[i][0]*2+1, t_sample_pts[i][1]*2])
        elif contour[t_sample_pts[i][0]*2+1, t_sample_pts[i][1]*2+1, 0] > 0:
            f_sample_pts.append([t_sample_pts[i][0]*2+1, t_sample_pts[i][1]*2+1])
    # print index, cc
    return f_sample_pts

def get_pair_pts(gt_contour, sp_pts, pair_id):
    gt_pts = []
    index = 0
    for i in range(640):
        for j in range(480):
            if gt_contour[j, i, 0] > 0:
                gt_pts.append([j, i])
                index += 1

    gtpts_tree = scipy.spatial.KDTree(gt_pts)
    s_sp_pts = []
    if pair_id != None:
        for i in range(pair_id.shape[0]):
            s_sp_pts.append(sp_pts[pair_id[i]])
    else:
        s_sp_pts = sp_pts
    pair_pts = []
    pair_res = gtpts_tree.query(s_sp_pts)
    for i in range(len(pair_res[1])):
        pair_pts.append(gt_pts[pair_res[1][i]])
    # print pair_id.shape[0]
    # for i in range(pair_id.shape[0]):
    #     tmp_dis = 1000000
    #     min_index = 0
    #     for j in range(index):
    #         cur_dis = (sp_pts[pair_id[i]][0]-gt_pts[j][0])*(sp_pts[pair_id[i]][0]-gt_pts[j][0]) \
    #                   + (sp_pts[pair_id[i]][1]-gt_pts[j][1])*(sp_pts[pair_id[i]][1]-gt_pts[j][1])
    #         if cur_dis < tmp_dis:
    #             tmp_dis = cur_dis
    #             min_index = j
    #     pair_pts.append(gt_pts[min_index])
    return pair_pts

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

def residual_rtt(pars, verts, Crt, Ct, pair_pts):
    tp = np.array([pars['tx'], pars['ty'], pars['tz']])
    rtp = np.array([pars['rtx'], pars['rty'], pars['rtz']])
    t_verts = verts.dot(Rodrigues(rtp)) + tp
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


if __name__ == '__main__':

    teeth_file_folder = '/home/jiaming/MultiviewFitting/data/upper_segmented/HBF_12681/before'
    # teeth_file_folder = 'data/from GuMin/seg/model_0'
    moved_mesh_folder = '/home/jiaming/MultiviewFitting/data/observation/12681/movedRow_real1.obj'
    img1_file_path = '/home/jiaming/MultiviewFitting/data/observation/12681/real_rc1.jpg'
    img2_file_path = '/home/jiaming/MultiviewFitting/data/observation/12681/real_rc2.jpg'
    img3_file_path = '/home/jiaming/MultiviewFitting/data/observation/12681/real_rc3.jpg'
    img4_file_path = '/home/jiaming/MultiviewFitting/data/observation/12681/real_rc4.jpg'
    # img5_file_path = 'data/observation/contour_4.jpg'
    

    print(teeth_file_folder)

    teeth_row_mesh = Mesh.TeethRowMesh(teeth_file_folder, False)
    row_mesh = teeth_row_mesh.row_mesh

    moved_mesh = Mesh.TeethRowMesh(moved_mesh_folder, True)
    # t0 = ch.asarray(teeth_row_mesh.positions_in_row)

    numTooth = len(teeth_row_mesh.mesh_list)
    Ri_list = [ch.zeros(3) for i in range(numTooth)]
    ti_list = [ch.zeros(3) for i in range(numTooth)]
    # R_row = ch.zeros(3)
    # t_row = ch.zeros(3)
    # R_row = ch.array([0, 0, 0.06])
    # t_row = ch.array([0, 0.06, 0])
    R_row = ch.array([0, 0, 0.06])
    t_row = ch.array([0, 0.06, 0])
    teeth_row_mesh.rotate(R_row)
    teeth_row_mesh.translate(t_row)

    Vi_list = [ch.array(teeth_row_mesh.mesh_list[i].v) for i in range(numTooth)]
    Vi_offset = [ch.mean(Vi_list[i], axis=0) for i in range(numTooth)]
    # print(Vi_offset)

    Vi_center = [(Vi_list[i] - Vi_offset[i]) for i in range(numTooth)]

    # V_row = t_row + ch.vstack([ti_list[i] + Vi_offset[i] + Vi_center[i].dot(Rodrigues(Ri_list[i])) for i in range(numTooth)]).dot(Rodrigues(R_row))
    V_row = ch.vstack([Vi_list[i] for i in range(numTooth)])
    # V_comb = ch.vstack([ti_list[i] + Vi_list[i].mean(axis=0) + (Vi_list[i] - Vi_list[i].mean(axis=0)).dot(Rodrigues(Ri_list[i])) for i in range(numTooth)])
    # V_row = t_row + V_comb.mean(axis=0) + (V_comb - V_comb.mean(axis=0)).dot(Rodrigues(R_row))

    # Mesh.save_to_obj('result/V_row_initial.obj', V_row.r, row_mesh.f)

    w, h = (640, 480)

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # vw = cv2.VideoWriter('result/optimRecord.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (2*w, 2*h))

    # rn = _create_colored_renderer()
    # imtmp = simple_renderer(rn, m.v, m.f)

    rn = BoundaryRenderer()
    drn = DepthRenderer()
    # rn.camera = ProjectPoints(v=V_row, rt=ch.zeros(3), t=ch.array([0, 0, 0]), f=ch.array([w, w]) / 2.,
    #                           c=ch.array([w, h]) / 2.,
    #                           k=ch.zeros(5))
    # rt = ch.array([0, 1, 0]) * np.pi / 4
    # rn.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([-1.5, 0, 0.5]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    #12681
    rt1 = ch.array([0, -0.3, 0]) * np.pi/2
    t1 = ch.array([1.2, 0.2, 0])
    rn.camera = ProjectPoints(v=V_row, rt=rt1, t=t1, f=ch.array([w, w]) / 2.,
                                   c=ch.array([w, h]) / 2.,
                                   k=ch.zeros(5))
    drn.camera = ProjectPoints(v=V_row, rt=rt1, t=t1, f=ch.array([w, w]) / 2.,
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

    rn2 = BoundaryRenderer()
    drn2 = DepthRenderer()
    # rt = ch.array([0, 1, 0]) * np.pi / 2
    # rn2.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([-2, 0, 2]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    # rt = ch.array([-1, 0, 0]) * np.pi / 4
    # rn2.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([0, -1.5, 0.4]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    #12681
    rt2 = ch.array([0.08, 0, 0]) * np.pi / 2
    t2 = ch.array([-0.05, 0.2, -0.25])
    rn2.camera = ProjectPoints(v=V_row, rt=rt2, t=t2, f=ch.array([w, w]) / 2.,
                                   c=ch.array([w, h]) / 2.,
                                   k=ch.zeros(5))
    drn2.camera = ProjectPoints(v=V_row, rt=rt2, t=t2, f=ch.array([w, w]) / 2.,
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

    rn3 = BoundaryRenderer()
    drn3 = DepthRenderer()
    # rt = ch.array([0, -1, 0]) * np.pi/2
    # rn3.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([2, 0, 2]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    # rt = ch.array([-0.3, -1, 0]) * np.pi / 4
    # rn3.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([1.5, -0.5, 0.8]), f=ch.array([w, w]) / 2.,
    #                           c=ch.array([w, h]) / 2.,
    #                           k=ch.zeros(5))
    #12681
    rt3 = ch.array([-0.9, 0, 0]) * np.pi / 3
    t3 = ch.array([0, -1.5, 0.2])
    rn3.camera = ProjectPoints(v=V_row, rt=rt3, t=t3, f=ch.array([w, w]) / 2.,
                              c=ch.array([w, h]) / 2.,
                              k=ch.zeros(5))
    drn3.camera = ProjectPoints(v=V_row, rt=rt3, t=t3, f=ch.array([w, w]) / 2.,
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


    rn4 = BoundaryRenderer()
    drn4 = DepthRenderer()
    # rt = ch.array([0, -1, 0]) * np.pi/4
    # rn4.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([1.5, 0, 0.5]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    # rt = ch.array([0, -1, 0]) * np.pi / 6
    # rn4.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([0.8, 0, 0.6]), f=ch.array([w, w]) / 2.,
    #                                c=ch.array([w, h]) / 2.,
    #                                k=ch.zeros(5))
    #12681
    rt4 = ch.array([0.1, 0.4, 0]) * np.pi/2
    t4 = ch.array([-1.4, 0.3, 0.2])
    rn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([w, w]) / 2.,
                                   c=ch.array([w, h]) / 2.,
                                   k=ch.zeros(5))
    drn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([w, w]) / 2.,
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
    #individual tooth pose estimation
    for i in range(numTooth):
        err = 100000
        err_dif = 100
        iter = 0
        # print V_row.shape
        # cur_tooth = V_row[teeth_row_mesh.start_idx_list[i]:teeth_row_mesh.start_idx_list[i+1], 0:3]
        # print cur_tooth.shape
        while not(err < 10 or err_dif < 1 or iter > 20):
            mean = np.mean(Vi_list[i].r, axis=0)
            # print mean
            curs_time = time.time()
            sample_pts1 = get_sample_pts(rn.r) #first time using intial rn
            print('sample point time: %s s' % (time.time() - curs_time))
            cur_time = time.time()
            # get back projection 3D verts and id of 2D points which can find corresponding verts
            sp_ins_pts1 = pj.back_projection_depth(sample_pts1, rt1, t1, drn.r)
            # intersection_pts1, index_ray1, index_tri1 = pj.back_projection(sample_pts1, rt1, t1, V_row, row_mesh.f)
            print('back projection time: %s s' % (time.time() - cur_time))
            cur_time = time.time()
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
            pair_pts1 = get_pair_pts(observed1, sample_pts1, None) #get pairing points
            print('pairing time: %s s' % (time.time() - cur_time))

            sample_pts2 = get_sample_pts(rn2.r)
            sp_ins_pts2 = pj.back_projection_depth(sample_pts2, rt2, t2, drn2.r)
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
            pair_pts2 = get_pair_pts(observed2, sample_pts2, None)

            sample_pts3 = get_sample_pts(rn3.r)
            sp_ins_pts3 = pj.back_projection_depth(sample_pts3, rt3, t3, drn3.r)
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
            pair_pts3 = get_pair_pts(observed3, sample_pts3, None)

            sample_pts4 = get_sample_pts(rn4.r)
            sp_ins_pts4 = pj.back_projection_depth(sample_pts4, rt4, t4, drn4.r)
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
            pair_pts4 = get_pair_pts(observed4, sample_pts4, None)

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
                                args=(mean, sp_ins_pts1, sp_ins_pts2, sp_ins_pts3, sp_ins_pts4, rt1, t1, rt2, t2, rt3, t3, rt4, t4, pair_pts1, pair_pts2, pair_pts3, pair_pts4),
                                method='leastsq')
            print('optimization time: %s s' % (time.time() - cur_time))
            cur_time = time.time()
            err_dif = err - out.chisqr
            if (err_dif > 0):
                err = out.chisqr
                # out.params.pretty_print()
                tmprt = np.array([out.params['rtx'], out.params['rty'], out.params['rtz']])
                tmpt = np.array([out.params['tx'], out.params['ty'], out.params['tz']])
                print tmprt, tmpt
                Vi_list[i] = (Vi_list[i] - mean).dot(Rodrigues(tmprt)) + mean + tmpt
                V_row = ch.vstack([Vi_list[k] for k in range(numTooth)])
                # V_row = (V_row-mean).dot(Rodrigues(tmprt)) + mean + tmpt

            print out.message, out.chisqr

            #reproject 2D contour
            rn.camera = ProjectPoints(v=V_row, rt=rt1, t=t1, f=ch.array([w, w]) / 2.,
                                    c=ch.array([w, h]) / 2.,
                                    k=ch.zeros(5))
            rn2.camera = ProjectPoints(v=V_row, rt=rt2, t=t2, f=ch.array([w, w]) / 2.,
                                    c=ch.array([w, h]) / 2.,
                                    k=ch.zeros(5))
            rn3.camera = ProjectPoints(v=V_row, rt=rt3, t=t3, f=ch.array([w, w]) / 2.,
                                    c=ch.array([w, h]) / 2.,
                                    k=ch.zeros(5))
            rn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([w, w]) / 2.,
                                    c=ch.array([w, h]) / 2.,
                                    k=ch.zeros(5))
            drn.camera = ProjectPoints(v=V_row, rt=rt1, t=t1, f=ch.array([w, w]) / 2.,
                                      c=ch.array([w, h]) / 2.,
                                      k=ch.zeros(5))
            drn2.camera = ProjectPoints(v=V_row, rt=rt2, t=t2, f=ch.array([w, w]) / 2.,
                                       c=ch.array([w, h]) / 2.,
                                       k=ch.zeros(5))
            drn3.camera = ProjectPoints(v=V_row, rt=rt3, t=t3, f=ch.array([w, w]) / 2.,
                                       c=ch.array([w, h]) / 2.,
                                       k=ch.zeros(5))
            drn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([w, w]) / 2.,
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

        rn1_dc[rn1_dc[:, :, 0] > 0] *= [1, 0, 0]
        rn2_dc[rn2_dc[:, :, 0] > 0] *= [1, 0, 0]
        rn3_dc[rn3_dc[:, :, 0] > 0] *= [1, 0, 0]
        rn4_dc[rn4_dc[:, :, 0] > 0] *= [1, 0, 0]

        axarr[0, 0].imshow(rn1_dc + ob1_dc)
        axarr[0, 1].imshow(rn2_dc + ob2_dc)
        axarr[1, 0].imshow(rn3_dc + ob3_dc)
        axarr[1, 1].imshow(rn4_dc + ob4_dc)

        scipy.misc.imsave('result/log/fittingresult1_iter{}.jpg'.format(i), rn1_dc + ob1_dc)
        scipy.misc.imsave('result/log/fittingresult1_iter{}a.jpg'.format(i), rn1_dc)
        scipy.misc.imsave('result/log/fittingresult1_iter{}b.jpg'.format(i), ob1_dc)
        scipy.misc.imsave('result/log/fittingresult2_iter{}.jpg'.format(i), rn2_dc + ob2_dc)
        scipy.misc.imsave('result/log/fittingresult2_iter{}a.jpg'.format(i), rn2_dc)
        scipy.misc.imsave('result/log/fittingresult2_iter{}b.jpg'.format(i), ob2_dc)
        scipy.misc.imsave('result/log/fittingresult3_iter{}.jpg'.format(i), rn3_dc + ob3_dc)
        scipy.misc.imsave('result/log/fittingresult3_iter{}a.jpg'.format(i), rn3_dc)
        scipy.misc.imsave('result/log/fittingresult3_iter{}b.jpg'.format(i), ob3_dc)
        scipy.misc.imsave('result/log/fittingresult4_iter{}.jpg'.format(i), rn4_dc + ob4_dc)
        scipy.misc.imsave('result/log/fittingresult4_iter{}a.jpg'.format(i), rn4_dc)
        scipy.misc.imsave('result/log/fittingresult4_iter{}b.jpg'.format(i), ob4_dc)

        print("tooth id: %d error: %f --- %s seconds ---" % (i, err, time.time() - start_time))
        total_time += (time.time() - start_time)
        start_time = time.time()

        plt.pause(5)

    print("total time --- %s seconds ---" % (total_time))
    Mesh.save_to_obj('result/V_row_opm.obj', V_row, row_mesh.f)


    # observed5 = load_image(img5_file_path)
    # E_raw5 = rn5 - observed5
    # E_pyr5 = gaussian_pyramid(E_raw5, n_levels=n_level, normalization=normalizations[nth])  # , normalization='size'
    # objs['View5'] = E_pyr5

    # w_s = 1.0e-1
    # E_sparse = 0
    # for i in range(numTooth):
    #     E_sparse += ch.abs(Ri_list[i]).sum() + ch.abs(ti_list[i]).sum()
    # E_sparse = w_s*E_sparse/numTooth

    plt.ion()
    # ax1, ax2, ax3, ax4 = None, None, None, None
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
    fit_vis = []
    # ob5_dc[ob5_dc[:, :, 0] > 0] *= [0, 1, 0]
    iter = 0
    start_time = time.time()

    def cb(_):
        global ob1_dc, ob2_dc, ob3_dc, ob4_dc, fig, axarr, iter, start_time

        print("--- %s seconds ---" % (time.time() - start_time))

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
        # fig.subplots_adjust(hspace=0, wspace=0)

        # fit_vis.append(rn1_dc + ob1_dc)
        # fit_vis.append(rn2_dc + ob2_dc)
        # fit_vis.append(rn3_dc + ob3_dc)
        # fit_vis.append(rn4_dc + ob4_dc)

        scipy.misc.imsave('result/log/fittingresult1_iter{}.jpg'.format(iter), rn1_dc + ob1_dc)
        scipy.misc.imsave('result/log/fittingresult1_iter{}a.jpg'.format(iter), rn1_dc)
        scipy.misc.imsave('result/log/fittingresult1_iter{}b.jpg'.format(iter), ob1_dc)
        scipy.misc.imsave('result/log/fittingresult2_iter{}.jpg'.format(iter), rn2_dc + ob2_dc)
        scipy.misc.imsave('result/log/fittingresult2_iter{}a.jpg'.format(iter), rn2_dc)
        scipy.misc.imsave('result/log/fittingresult2_iter{}b.jpg'.format(iter), ob2_dc)
        scipy.misc.imsave('result/log/fittingresult3_iter{}.jpg'.format(iter), rn3_dc + ob3_dc)
        scipy.misc.imsave('result/log/fittingresult3_iter{}a.jpg'.format(iter), rn3_dc)
        scipy.misc.imsave('result/log/fittingresult3_iter{}b.jpg'.format(iter), ob3_dc)
        scipy.misc.imsave('result/log/fittingresult4_iter{}.jpg'.format(iter), rn4_dc + ob4_dc)
        scipy.misc.imsave('result/log/fittingresult4_iter{}a.jpg'.format(iter), rn4_dc)
        scipy.misc.imsave('result/log/fittingresult4_iter{}b.jpg'.format(iter), ob4_dc)
        iter += 1
        start_time = time.time()

        # vis_img = np.vstack([np.hstack([rn1_dc + ob1_dc, rn2_dc + ob2_dc]), np.hstack([rn3_dc + ob3_dc, rn4_dc + ob4_dc])])
        # vw.write(vis_img)

        plt.pause(5)

    # stages = 1
    # methods = ['dogleg', 'SLSQP', 'Newton-CG', 'BFGS']
    # method = 1 #'trust-ncg' #'newton-cg' #''BFGS' #'dogleg'
    # # option = {'maxiter': 1000, 'disp': 1, 'e_3': 1.0e-4}
    # option = {'disp': True}
    # # option = None
    # tol = 1e-15
    # ch.random.seed(19921122)
    # random_move = lambda x, order: (1.0 - 1.0 / np.e**order + 2*np.random.rand(3) / np.e**order)*x
    # start_time = time.time()
    # # todo: only a few teeth moved, so adding a sparse constraint to Ri_list and ti_list should make a better result.
    # print ('OPTIMIZING TRANSLATION, ROTATION: method=[{}]'.format(methods[method]))
    # for stage in range(stages):
    #     print('## Stage {} ##'.format(stage))
    #     # randomly jump around the solution to help get rid of local minimum,=
    #     #  as the stage increase, the movement should be smaller
    #     if stage != 0:
    #         #R_row[:] = random_move(R_row.r, stage)
    #         #t_row[:] = random_move(t_row.r, stage)
    #         for i in range(numTooth):
    #             Ri_list[i][:] = random_move(Ri_list[i].r, stage+4)
    #             ti_list[i][:] = random_move(ti_list[i].r, stage+4)
    #     # ch.minimize({'pyr1': E_pyr1}, x0=[t_row], callback=cb, method=method, options=option, tol=tol)
    #     # ch.minimize({'pyr2': E_pyr1}, x0=[R_row, t_row], callback=cb, method=method, options=option, tol=tol)
    #     # ch.minimize({'pyr3': E_pyr1}, x0=ti_list, callback=cb, method=method, options=option, tol=tol) #[R_row, t_row] +
    #     # ch.minimize({'pyr4': E_pyr1}, x0=Ri_list + ti_list, callback=cb, method=method, options=option, tol=tol) #[R_row, t_row] +
    #
    #     if stage == 0:
    #         print('Sub-stage {}-1: x0=[t_row]'.format(stage))
    #         op.minimize(residual, x0=[t_row], callback=cb, method=methods[method], options=option, tol=tol)
    #
    #         print('Sub-stage {}-2: x0=[R_row, t_row]'.format(stage))
    #         op.minimize(residual, x0=[R_row, t_row], callback=cb, method=methods[method], options=option, tol=tol)
    #
    #     # print('Sub-stage {}-3: x0=[R_row, t_row] + ti_list'.format(stage))
    #     # ch.minimize(objs, x0=[R_row, t_row] + ti_list, callback=cb, method=methods[method], options=option, tol=tol)  # [R_row, t_row] +
    #
    #     for k in range(3):
    #         for i in range(2):
    #             print('Sub-stage {}-3, round {}: x0=ti_list'.format(stage, i))
    #             for j in range(numTooth):
    #                 ch.minimize(residual, x0=[ti_list[j]], callback=cb, method=methods[method], options=option, tol=tol)
    #
    #         for i in range(3):
    #             print('Sub-stage {}-4, round {}: x0=Ri_list'.format(stage, i))
    #             for j in range(numTooth):
    #                 ch.minimize(residual, x0=[Ri_list[j]], callback=cb, method=methods[method], options=option, tol=tol)  # Ri_list +
    #
    #     for i in range(2):
    #         print('Sub-stage {}-5, round {}: x0=Ri_list+ti_list'.format(stage, i))
    #         for j in range(numTooth):
    #             ch.minimize(residual, x0=[Ri_list[j], ti_list[j]], callback=cb, method=methods[method], options=option, tol=tol)   #Ri_list + ti_list
    #
    #     # print('Sub-stage {}-6: x0=[R_row, t_row] + Ri_list + ti_list'.format(stage))
    #     # ch.minimize(objs, x0=[R_row, t_row] + Ri_list + ti_list, callback=cb, method=methods[method], options=option, tol=tol)  # [R_row, t_row] +
    #
    #     Mesh.save_to_obj('result/fittedRow_stage{}.obj'.format(stage), V_row.r, row_mesh.f)
    #     t_c = len(teeth_row_mesh.mesh_list[0].v)
    #     vert_t = [[] for i in range(numTooth)]
    #     mvert_t = [[] for i in range(numTooth)]
    #     idx = 0
    #     for i in range(V_row.shape[0]):
    #         if i < t_c:
    #             vert_t[idx].append(V_row[i].r)
    #         else:
    #             idx += 1
    #             vert_t[idx].append(V_row[i].r)
    #             t_c += len(teeth_row_mesh.mesh_list[idx].v)
    #         mvert_t[idx].append(moved_mesh.row_mesh.v[i])
    #
    #     for i in range(numTooth):
    #         #Mesh.save_to_obj('result/fittedRow_stage{}.obj'.format(stage), vert_t[i], teeth_row_mesh.mesh_list[i].f)
    #         #print(i)
    #         d, Z, tform = p.procrustes(np.array(mvert_t[i]), np.array(vert_t[i]), scaling=False)
    #         # new_mvert_t = mvert_t / 2.0
    #         # new_mvert_t *= moved_mesh.max_v
    #         # new_vert_t = vert_t / 2.0
    #         # new_vert_t *= teeth_row_mesh.max_v
    #         # d1, Z1, tform1 = p.procrustes(np.array(new_mvert_t[i]), np.array(new_vert_t[i]), scaling=False)
    #         #print ("--tooth_{}'s errors are: ".format(i), d, tform['translation'], p.rotationMatrixToEulerAngles(tform['rotation']), 'scale_reverse:', tform1['translation'], p.rotationMatrixToEulerAngles(tform1['rotation']))
    #         print ("--tooth_{}'s errors are: ".format(i), d, tform['translation'], p.rotationMatrixToEulerAngles(tform['rotation']))
    #     scipy.misc.imsave('result/fittingresult1_stage{}.jpg'.format(stage), rn.r)
    #     scipy.misc.imsave('result/fittingresult2_stage{}.jpg'.format(stage), rn2.r)
    #     scipy.misc.imsave('result/fittingresult3_stage{}.jpg'.format(stage), rn3.r)
    #     scipy.misc.imsave('result/fittingresult4_stage{}.jpg'.format(stage), rn4.r)
    #     # scipy.misc.imsave('result/fittingresult5_stage{}.jpg'.format(stage), rn5.r)
    #     # scipy.misc.imsave('result/fittingresult_diff1_stage{}.jpg'.format(stage), np.abs(E_raw1.r))
    #     # scipy.misc.imsave('result/fittingresult_diff2_stage{}.jpg'.format(stage), np.abs(E_raw2.r))
    #     # scipy.misc.imsave('result/fittingresult_diff3_stage{}.jpg'.format(stage), np.abs(E_raw3.r))
    #     # scipy.misc.imsave('result/fittingresult_diff4_stage{}.jpg'.format(stage), np.abs(E_raw4.r))
    #     # scipy.misc.imsave('result/fittingresult_diff5_stage{}.jpg'.format(stage), np.abs(E_raw5.r))
    #
    #
    # print("---Optimization takes %s seconds ---" % (time.time()-start_time))
    # #vw.release()