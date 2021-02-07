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
    # 12681
    mean = np.mean(V_row.r, axis=0)
    rt = ch.array([0, -0.3, 0]) * np.pi / 2
    rtn = np.array(rt.r)
    tmprt = R.from_rotvec(rtn)
    rt_mat = tmprt.as_dcm()
    inv_rt = np.linalg.inv(rt_mat)
    tmprt = R.from_dcm(inv_rt)
    rt_q = tmprt.as_quat()
    t = ch.array([1.2, 0.2, 0])
    Crt = np.array(R_row.r)
    Ct = np.array(t_row.r)
    tmpr = R.from_rotvec(Crt)
    r_mat = tmpr.as_dcm()
    # t_vec = Ct.T
    # cor_mtx = np.zeros((3, 3), dtype='float32')
    # cor_mtx[0:3, 0:3] = r_mat
    # cor_mtx[0:3, 3] = t_vec
    # cor_mtx[3, 3] = 1
    # print cor_mtx
    inv_r = np.linalg.inv(r_mat)
    tmpr = R.from_dcm(inv_r)
    r_q = tmpr.as_quat()
    # print inv_cormtx
    print r_q,rt_q
    t1 = -((-t - mean).dot(inv_r)+mean-Ct)
    rc_mat = np.linalg.inv((R.from_quat(r_q*rt_q)).as_dcm())
    # rc_mat = np.linalg.inv(inv_r.dot(inv_rt))
    tmprt = R.from_dcm(rc_mat)
    rt1 = tmprt.as_rotvec()
    # rt1 = rt
    print rt1, t1

    rn.camera = ProjectPoints(v=V_row, rt=rt1, t=t1, f=ch.array([w, w]) / 2.,
                              c=ch.array([w, h]) / 2.,
                              k=ch.zeros(5))
    rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)



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
    rn1_dc = deepcopy(rn.r)
    rn1_dc[rn1_dc[:, :, 0] > 0] *= [1, 0, 0]
    plt.imshow(rn1_dc + ob1_dc)
    # plt.show()
    plt.pause(20)
    scipy.misc.imsave('result/test1.jpg', rn1_dc + ob1_dc)
    # fit_vis = []
    # # ob5_dc[ob5_dc[:, :, 0] > 0] *= [0, 1, 0]
    # iter = 0
    # start_time = time.time()


    # def cb(_):
    #     global E_raw1, E_raw2, E_raw3, E_raw4, ob1_dc, ob2_dc, ob3_dc, ob4_dc, fig, axarr, iter, start_time
    #
    #     print("--- %s seconds ---" % (time.time() - start_time))
    #
    #     if axarr is None:
    #         fig, axarr = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'wspace': 0, 'hspace': 0})
    #         # fig.subplots_adjust(hspace=0, wspace=0)
    #     fig.patch.set_facecolor('grey')
    #
    #     rn1_dc = deepcopy(rn.r)
    #     rn2_dc = deepcopy(rn2.r)
    #     rn3_dc = deepcopy(rn3.r)
    #     rn4_dc = deepcopy(rn4.r)
    #
    #     rn1_dc[rn1_dc[:, :, 0] > 0] *= [1, 0, 0]
    #     rn2_dc[rn2_dc[:, :, 0] > 0] *= [1, 0, 0]
    #     rn3_dc[rn3_dc[:, :, 0] > 0] *= [1, 0, 0]
    #     rn4_dc[rn4_dc[:, :, 0] > 0] *= [1, 0, 0]
    #
    #     axarr[0, 0].imshow(rn1_dc + ob1_dc)
    #     axarr[0, 1].imshow(rn2_dc + ob2_dc)
    #     axarr[1, 0].imshow(rn3_dc + ob3_dc)
    #     axarr[1, 1].imshow(rn4_dc + ob4_dc)
    #     # fig.subplots_adjust(hspace=0, wspace=0)
    #
    #     # fit_vis.append(rn1_dc + ob1_dc)
    #     # fit_vis.append(rn2_dc + ob2_dc)
    #     # fit_vis.append(rn3_dc + ob3_dc)
    #     # fit_vis.append(rn4_dc + ob4_dc)
    #
    #     scipy.misc.imsave('result/log/fittingresult1_iter{}.jpg'.format(iter), rn1_dc + ob1_dc)
    #     scipy.misc.imsave('result/log/fittingresult1_iter{}a.jpg'.format(iter), rn1_dc)
    #     scipy.misc.imsave('result/log/fittingresult1_iter{}b.jpg'.format(iter), ob1_dc)
    #     scipy.misc.imsave('result/log/fittingresult2_iter{}.jpg'.format(iter), rn2_dc + ob2_dc)
    #     scipy.misc.imsave('result/log/fittingresult2_iter{}a.jpg'.format(iter), rn2_dc)
    #     scipy.misc.imsave('result/log/fittingresult2_iter{}b.jpg'.format(iter), ob2_dc)
    #     scipy.misc.imsave('result/log/fittingresult3_iter{}.jpg'.format(iter), rn3_dc + ob3_dc)
    #     scipy.misc.imsave('result/log/fittingresult3_iter{}a.jpg'.format(iter), rn3_dc)
    #     scipy.misc.imsave('result/log/fittingresult3_iter{}b.jpg'.format(iter), ob3_dc)
    #     scipy.misc.imsave('result/log/fittingresult4_iter{}.jpg'.format(iter), rn4_dc + ob4_dc)
    #     scipy.misc.imsave('result/log/fittingresult4_iter{}a.jpg'.format(iter), rn4_dc)
    #     scipy.misc.imsave('result/log/fittingresult4_iter{}b.jpg'.format(iter), ob4_dc)
    #     iter += 1
    #     start_time = time.time()
    #
    #     # vis_img = np.vstack([np.hstack([rn1_dc + ob1_dc, rn2_dc + ob2_dc]), np.hstack([rn3_dc + ob3_dc, rn4_dc + ob4_dc])])
    #     # vw.write(vis_img)
    #
    #     plt.pause(5)


    # stages = 1
    # methods = ['dogleg', 'SLSQP', 'Newton-CG', 'BFGS']
    # method = 3  # 'trust-ncg' #'newton-cg' #''BFGS' #'dogleg'
    # # option = {'maxiter': 1000, 'disp': 1, 'e_3': 1.0e-4}
    # option = {'disp': True}
    # # option = None
    # tol = 1e-15
    # ch.random.seed(19921122)
    # random_move = lambda x, order: (1.0 - 1.0 / np.e ** order + 2 * np.random.rand(3) / np.e ** order) * x
    # start_time = time.time()
    # # todo: only a few teeth moved, so adding a sparse constraint to Ri_list and ti_list should make a better result.
    # print ('OPTIMIZING TRANSLATION, ROTATION: method=[{}]'.format(methods[method]))
    # for stage in range(stages):
    #     print('## Stage {} ##'.format(stage))
    #     # randomly jump around the solution to help get rid of local minimum,=
    #     #  as the stage increase, the movement should be smaller
    #     # if stage != 0:
    #     #     # R_row[:] = random_move(R_row.r, stage)
    #     #     # t_row[:] = random_move(t_row.r, stage)
    #     #     for i in range(numTooth):
    #     #         Ri_list[i][:] = random_move(Ri_list[i].r, stage + 4)
    #     #         ti_list[i][:] = random_move(ti_list[i].r, stage + 4)
    #     # ch.minimize({'pyr1': E_pyr1}, x0=[t_row], callback=cb, method=method, options=option, tol=tol)
    #     # ch.minimize({'pyr2': E_pyr1}, x0=[R_row, t_row], callback=cb, method=method, options=option, tol=tol)
    #     # ch.minimize({'pyr3': E_pyr1}, x0=ti_list, callback=cb, method=method, options=option, tol=tol) #[R_row, t_row] +
    #     # ch.minimize({'pyr4': E_pyr1}, x0=Ri_list + ti_list, callback=cb, method=method, options=option, tol=tol) #[R_row, t_row] +
    #
    #     if stage == 0:
    #         print('Sub-stage {}-1: x0=[t_row]'.format(stage))
    #         op.minimize(residual, x0=[t_row], args=(observed1, Vi_list), method=methods[method], options=option, tol=tol)
    #
    #     #     print('Sub-stage {}-2: x0=[R_row, t_row]'.format(stage))
    #     #     op.minimize(objs, x0=[R_row, t_row], callback=cb, method=methods[method], options=option, tol=tol)
    #     #
    #     # # print('Sub-stage {}-3: x0=[R_row, t_row] + ti_list'.format(stage))
    #     # # ch.minimize(objs, x0=[R_row, t_row] + ti_list, callback=cb, method=methods[method], options=option, tol=tol)  # [R_row, t_row] +
    #     #
    #     # for k in range(3):
    #     #     for i in range(2):
    #     #         print('Sub-stage {}-3, round {}: x0=ti_list'.format(stage, i))
    #     #         for j in range(numTooth):
    #     #             ch.minimize(objs, x0=[ti_list[j]], callback=cb, method=methods[method], options=option, tol=tol)
    #     #
    #     #     for i in range(3):
    #     #         print('Sub-stage {}-4, round {}: x0=Ri_list'.format(stage, i))
    #     #         for j in range(numTooth):
    #     #             ch.minimize(objs, x0=[Ri_list[j]], callback=cb, method=methods[method], options=option,
    #     #                         tol=tol)  # Ri_list +
    #     #
    #     # for i in range(2):
    #     #     print('Sub-stage {}-5, round {}: x0=Ri_list+ti_list'.format(stage, i))
    #     #     for j in range(numTooth):
    #     #         ch.minimize(objs, x0=[Ri_list[j], ti_list[j]], callback=cb, method=methods[method], options=option,
    #     #                     tol=tol)  # Ri_list + ti_list
    #
    #     # print('Sub-stage {}-6: x0=[R_row, t_row] + Ri_list + ti_list'.format(stage))
    #     # ch.minimize(objs, x0=[R_row, t_row] + Ri_list + ti_list, callback=cb, method=methods[method], options=option, tol=tol)  # [R_row, t_row] +
    #
    # #     Mesh.save_to_obj('result/fittedRow_stage{}.obj'.format(stage), V_row.r, row_mesh.f)
    # #     t_c = len(teeth_row_mesh.mesh_list[0].v)
    # #     vert_t = [[] for i in range(numTooth)]
    # #     mvert_t = [[] for i in range(numTooth)]
    # #     idx = 0
    # #     for i in range(V_row.shape[0]):
    # #         if i < t_c:
    # #             vert_t[idx].append(V_row[i].r)
    # #         else:
    # #             idx += 1
    # #             vert_t[idx].append(V_row[i].r)
    # #             t_c += len(teeth_row_mesh.mesh_list[idx].v)
    # #         mvert_t[idx].append(moved_mesh.row_mesh.v[i])
    # #
    # #     for i in range(numTooth):
    # #         # Mesh.save_to_obj('result/fittedRow_stage{}.obj'.format(stage), vert_t[i], teeth_row_mesh.mesh_list[i].f)
    # #         # print(i)
    # #         d, Z, tform = p.procrustes(np.array(mvert_t[i]), np.array(vert_t[i]), scaling=False)
    # #         # new_mvert_t = mvert_t / 2.0
    # #         # new_mvert_t *= moved_mesh.max_v
    # #         # new_vert_t = vert_t / 2.0
    # #         # new_vert_t *= teeth_row_mesh.max_v
    # #         # d1, Z1, tform1 = p.procrustes(np.array(new_mvert_t[i]), np.array(new_vert_t[i]), scaling=False)
    # #         # print ("--tooth_{}'s errors are: ".format(i), d, tform['translation'], p.rotationMatrixToEulerAngles(tform['rotation']), 'scale_reverse:', tform1['translation'], p.rotationMatrixToEulerAngles(tform1['rotation']))
    # #         print ("--tooth_{}'s errors are: ".format(i), d, tform['translation'],
    # #                p.rotationMatrixToEulerAngles(tform['rotation']))
    # #     scipy.misc.imsave('result/fittingresult1_stage{}.jpg'.format(stage), rn.r)
    # #     scipy.misc.imsave('result/fittingresult2_stage{}.jpg'.format(stage), rn2.r)
    # #     scipy.misc.imsave('result/fittingresult3_stage{}.jpg'.format(stage), rn3.r)
    # #     scipy.misc.imsave('result/fittingresult4_stage{}.jpg'.format(stage), rn4.r)
    # #     # scipy.misc.imsave('result/fittingresult5_stage{}.jpg'.format(stage), rn5.r)
    # #     scipy.misc.imsave('result/fittingresult_diff1_stage{}.jpg'.format(stage), np.abs(E_raw1.r))
    # #     scipy.misc.imsave('result/fittingresult_diff2_stage{}.jpg'.format(stage), np.abs(E_raw2.r))
    # #     scipy.misc.imsave('result/fittingresult_diff3_stage{}.jpg'.format(stage), np.abs(E_raw3.r))
    # #     scipy.misc.imsave('result/fittingresult_diff4_stage{}.jpg'.format(stage), np.abs(E_raw4.r))
    # #     # scipy.misc.imsave('result/fittingresult_diff5_stage{}.jpg'.format(stage), np.abs(E_raw5.r))
    # #
    # # print("---Optimization takes %s seconds ---" % (time.time() - start_time))
    # # vw.release()