import Mesh
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
from opendr.camera import ProjectPoints
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.misc
import scipy.optimize as op
import time
import cv2
from config import cfg
import os
from scipy.spatial.transform import Rotation as R

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


    # Ri_list = [ch.zeros(3) for i in range(numTooth)]
    # ti_list = [ch.zeros(3) for i in range(numTooth)]
    # # R_row = ch.zeros(3)
    # # t_row = ch.zeros(3)
    R_row = ch.array([0, 0, 0.06])
    t_row = ch.array([0, 0.06, 0])
    # R_row = ch.array([0, 0, 0.01])
    # t_row = ch.array([0, 0, 0])
    # teeth_row_mesh.rotate(R_row)
    # teeth_row_mesh.translate(t_row)

    Vi_list = [ch.array(teeth_row_mesh.mesh_list[i].v) for i in range(numTooth)]
    Vi_offset = [ch.mean(Vi_list[i], axis=0) for i in range(numTooth)]
    Vi_center = [(Vi_list[i] - Vi_offset[i]) for i in range(numTooth)]

    V_row = ch.vstack([Vi_list[i] for i in range(numTooth)])

    teeth_row_mesh.rotate(R_row)
    teeth_row_mesh.translate(t_row)
    Vi_list1 = [ch.array(teeth_row_mesh.mesh_list[i].v) for i in range(numTooth)]
    V_row1 = ch.vstack([Vi_list1[i] for i in range(numTooth)])
    # V_row = t_row + ch.vstack(
    #     [ti_list[i] + Vi_offset[i] + Vi_center[i].dot(Rodrigues(Ri_list[i])) for i in range(numTooth)]).dot(
    #     Rodrigues(R_row))
    # V_comb = ch.vstack([ti_list[i] + Vi_list[i].mean(axis=0) + (Vi_list[i] - Vi_list[i].mean(axis=0)).dot(Rodrigues(Ri_list[i])) for i in range(numTooth)])
    # V_row = t_row + V_comb.mean(axis=0) + (V_comb - V_comb.mean(axis=0)).dot(Rodrigues(R_row))


    # create the Energy
    observed1 = load_image(img1_file_path)
    observed2 = load_image(img2_file_path)
    observed3 = load_image(img3_file_path)
    observed4 = load_image(img4_file_path)


    w, h = (640, 480)


    rn = BoundaryRenderer()
    rn1 = BoundaryRenderer()
    crn = ColoredRenderer()
    # 12681
    origin = np.array([0, 0, 0, 1]).T
    # X_o = np.array([1, 0, 0, 1]).T
    # Y_o = np.array([0, 1, 0, 1]).T
    # Z_o = np.array([0, 0, 1, 1]).T
    X_o = np.array([1, 0, 0]).T
    Y_o = np.array([0, 1, 0]).T
    Z_o = np.array([0, 0, 1]).T
    mean = np.mean(V_row.r, axis=0)
    rt = ch.array([0, -0.3, 0]) * np.pi / 2
    rtn = np.array(rt.r)
    tmprt = R.from_rotvec(rtn)
    rt_mat = tmprt.as_dcm()
    inv_rt = np.linalg.inv(rt_mat)
    t = np.array([1.2, 0.2, 0])
    cor_mtx = np.zeros((4, 4), dtype='float32')
    cor_mtx[0:3, 0:3] = rt_mat
    cor_mtx[0:3, 3] = t.T
    cor_mtx[3, 3] = 1
    # print cor_mtx
    inv_cormtx = np.linalg.inv(cor_mtx)

    Crt = np.array(R_row.r)
    Ct = np.array(t_row.r)
    tmpr = R.from_rotvec(Crt)
    r_mat = tmpr.as_dcm()
    inv_r = np.linalg.inv(r_mat)
    o_mtx = np.zeros((4, 4), dtype='float32')
    o_mtx[0:3, 0:3] = np.linalg.inv(r_mat)
    o_mtx[0:3, 3] = (mean - r_mat.dot(mean) + Ct).T
    o_mtx[3, 3] = 1
    # print cor_mtx
    inv_omtx = np.linalg.inv(o_mtx)

    # print inv_r.dot(r_mat.dot(Ct))

    # print inv_cormtx.dot(origin)[0:3], inv_cormtx.dot(origin)[0:3]-mean, mean
    # new_O = inv_r.dot(inv_cormtx.dot(origin)[0:3]-mean) + mean - inv_r.dot(Ct)
    new_O = inv_omtx.dot(inv_cormtx.dot(origin))
    # new_X = r_mat.dot(inv_cormtx.dot(X_o)[0:3] - mean) + r_mat.dot(mean) + Ct
    # new_Y = r_mat.dot(inv_cormtx.dot(Y_o)[0:3] - mean) + r_mat.dot(mean) + Ct
    # new_Z = r_mat.dot(inv_cormtx.dot(Z_o)[0:3] - mean) + r_mat.dot(mean) + Ct
    new_X = r_mat.dot(inv_rt.dot(X_o))
    new_Y = r_mat.dot(inv_rt.dot(Y_o))
    new_Z = r_mat.dot(inv_rt.dot(Z_o))

    # new_Mat = np.array([[new_X[0] - new_O[0], new_Y[0] - new_O[0], new_Z[0] - new_O[0], new_O[0]],
    #                     [new_X[1] - new_O[1], new_Y[1] - new_O[1], new_Z[1] - new_O[1], new_O[1]],
    #                     [new_X[2] - new_O[2], new_Y[2] - new_O[2], new_Z[2] - new_O[2], new_O[2]],
    #                     [0, 0, 0, 1]])
    # new_Mat = np.array([[new_X[0], new_Y[0], new_Z[0], new_O[0]],
    #                     [new_X[1], new_Y[1], new_Z[1], new_O[1]],
    #                     [new_X[2], new_Y[2], new_Z[2], new_O[2]],
    #                     [0, 0, 0, 1]])
    new_RT = cor_mtx.dot(o_mtx)
    # new_RT = np.linalg.inv(new_Mat)
    # print new_Mat, new_RT

    rc_mat = new_RT[0:3, 0:3]
    print np.linalg.det(rc_mat)
    tmprt = R.from_dcm(rc_mat)
    rt1 = tmprt.as_rotvec()
    t1 = new_RT[0:3, 3].T
    # t1 = t

    # rc_mat = np.linalg.inv(inv_r.dot(inv_rt))
    # t1 = -((-t.dot(rt_mat) - mean).dot(r_mat)+mean-Ct)
    # rc_mat = np.linalg.inv((R.from_quat(r_q*rt_q)).as_dcm())

    # tmprt = R.from_dcm(rc_mat)
    # rt1 = tmprt.as_rotvec()
    # rt1 = rt
    print rt1, t1

    rn.camera = ProjectPoints(v=V_row, rt=rt, t=t, f=ch.array([w, w]) / 2.,
                              c=ch.array([w, h]) / 2.,
                              k=ch.zeros(5))
    rn1.camera = ProjectPoints(v=V_row1, rt=rt, t=t, f=ch.array([w, w]) / 2.,
                              c=ch.array([w, h]) / 2.,
                              k=ch.zeros(5))
    # crn.camera = ProjectPoints(v=V_row, rt=rt, t=t, f=ch.array([w, w]) / 2.,
    #                           c=ch.array([w, h]) / 2.,
    #                           k=ch.zeros(5))
    rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    rn1.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn1.set(v=V_row1, f=row_mesh.f, vc=row_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
    # crn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    # crn.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc, bgcolor=ch.zeros(3), num_channels=3)

    # plt.imshow(crn.r)
    # plt.show()



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
    rn_dc = deepcopy(rn.r)
    rn1_dc = deepcopy(rn1.r)
    # print np.argwhere(rn1_dc[:, :, 0]>0)
    # print np.rint(rn1_dc[rn1_dc[:, :, 0] > 0][:, 0] * 20)
    rn_dc[rn_dc[:, :, 0] > 0] *= [1, 0, 0]
    rn1_dc[rn1_dc[:, :, 0] > 0] *= [0, 1, 1]
    plt.imshow(rn1_dc + rn_dc)
    # # plt.show()
    plt.pause(20)
    # scipy.misc.imsave('result/compareZ45.jpg', rn1_dc + rn_dc)
    # fit_vis = []
    # # ob5_dc[ob5_dc[:, :, 0] > 0] *= [0, 1, 0]
    # iter = 0
    # start_time = time.time()


