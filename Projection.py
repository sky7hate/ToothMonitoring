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
from opendr.renderer import DepthRenderer
from opendr.camera import ProjectPoints
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import scipy.misc
import scipy.optimize as op
import time
import cv2
from config import cfg
import os
import trimesh
from scipy.spatial.transform import Rotation as R


def back_projection(sample_pts, rt, t, cur_Vrow, faces):
    Crt = np.array(rt.r)
    Ct = np.array(t.r)
    reversed_imgpts = []
    index = len(sample_pts)
    for i in range(index):
        # print sample_pts[i]
        tmp = [float(sample_pts[i][1] - 320) / 320, float(sample_pts[i][0] - 240) / 320, 1]
        # print tmp
        reversed_imgpts.append(tmp)
    # print index

    # Calculate Vertices coordinates in Camera coordinate system
    V_row1 = deepcopy(cur_Vrow.r)
    tmpr = R.from_rotvec(Crt)
    r_mat = tmpr.as_dcm()
    t_vec = Ct.T
    cor_mtx = np.zeros((4, 4), dtype='float32')
    cor_mtx[0:3, 0:3] = r_mat
    cor_mtx[0:3, 3] = t_vec
    cor_mtx[3, 3] = 1
    # print cor_mtx
    inv_cormtx = np.linalg.inv(cor_mtx)
    # print inv_cormtx

    V_row_camera = []
    for i in range(V_row1.size / 3):
        # tmp_v = np.array([V_row1[i][0], V_row1[i][1], V_row1[i][2], 1]).dot(inv_cormtx)
        # tmp_v = np.matmul(np.array([V_row1[i][0], V_row1[i][1], V_row1[i][2], 1]), cor_mtx)
        tmp_v = cor_mtx.dot(np.array([V_row1[i][0], V_row1[i][1], V_row1[i][2], 1]).T)
        V_row_camera.append([tmp_v[0], tmp_v[1], tmp_v[2]])
    V_row_camera = np.vstack(V_row_camera)
    # print V_row_camera.size

    # Mesh.save_to_obj('result/V_row_camera3_R.obj', V_row_camera, row_mesh.f)

    # Ray tracing to find back-projection 3D vertices
    mesh_camera = trimesh.Trimesh(vertices=V_row_camera, faces=faces)
    # print mesh_camera.is_empty
    origins = np.zeros((index, 3), dtype='float32')
    dirs = np.zeros((index, 3), dtype='float32')
    dirs += reversed_imgpts
    intersection_pts, index_ray, index_tri = mesh_camera.ray.intersects_location(origins, dirs, multiple_hits=False)

    intersection_pts_world = []
    for i in range(intersection_pts.shape[0]):
        tmp_v = inv_cormtx.dot(np.array([intersection_pts[i][0], intersection_pts[i][1], intersection_pts[i][2], 1]).T)
        intersection_pts_world.append([tmp_v[0], tmp_v[1], tmp_v[2]])
    intersection_pts_world = np.vstack(intersection_pts_world)

    return intersection_pts_world, index_ray, index_tri

def back_projection_depth(sample_pts, rt, t, dmap):
    Crt = np.array(rt.r)
    Ct = np.array(t.r)
    reversed_imgpts = []
    index = len(sample_pts)
    for i in range(index):
        # print sample_pts[i]
        tmp = [float(sample_pts[i][1] - 320) / 320, float(sample_pts[i][0] - 240) / 320, 1]
        # print tmp
        reversed_imgpts.append(tmp)
    # print index

    Verts_d = []
    for i in range(index):
        # print reversed_imgpts[i][0], reversed_imgpts[i][1]
        d = dmap[sample_pts[i][0], sample_pts[i][1]]
        tmp = [reversed_imgpts[i][0] * d, reversed_imgpts[i][1] * d, d]
        Verts_d.append(tmp)
    Verts_d = np.vstack(Verts_d)

    tmpr = R.from_rotvec(Crt)
    r_mat = tmpr.as_dcm()
    t_vec = Ct.T
    cor_mtx = np.zeros((4, 4), dtype='float32')
    cor_mtx[0:3, 0:3] = r_mat
    cor_mtx[0:3, 3] = t_vec
    cor_mtx[3, 3] = 1
    # print cor_mtx
    inv_cormtx = np.linalg.inv(cor_mtx)

    Verts_d_world = []
    for i in range(Verts_d.shape[0]):
        tmp_v = inv_cormtx.dot(np.array([Verts_d[i][0], Verts_d[i][1], Verts_d[i][2], 1]).T)
        Verts_d_world.append([tmp_v[0], tmp_v[1], tmp_v[2]])
    Verts_d_world = np.vstack(Verts_d_world)

    return Verts_d_world


# if __name__ == '__main__':
#
#     teeth_file_folder = '/home/jiaming/MultiviewFitting/data/upper_segmented/HBF_12681/before'
#     # teeth_file_folder = 'data/from GuMin/seg/model_0'
#     moved_mesh_folder = '/home/jiaming/MultiviewFitting/data/observation/12681/movedRow_real1.obj'
#     img1_file_path = '/home/jiaming/MultiviewFitting/data/observation/12681/real_rc1.jpg'
#     img2_file_path = '/home/jiaming/MultiviewFitting/data/observation/12681/real_rc2.jpg'
#     img3_file_path = '/home/jiaming/MultiviewFitting/data/observation/12681/real_rc3.jpg'
#     img4_file_path = '/home/jiaming/MultiviewFitting/data/observation/12681/real_rc4.jpg'
#     # img5_file_path = 'data/observation/contour_4.jpg'
#
#     print(teeth_file_folder)
#
#     #read tooth mesh
#     teeth_row_mesh = Mesh.TeethRowMesh(teeth_file_folder, False)
#     row_mesh = teeth_row_mesh.row_mesh
#
#
#     moved_mesh = Mesh.TeethRowMesh(moved_mesh_folder, True)
#     # t0 = ch.asarray(teeth_row_mesh.positions_in_row)
#
#     numTooth = len(teeth_row_mesh.mesh_list)
#     Vi_list = [ch.array(teeth_row_mesh.mesh_list[i].v) for i in range(numTooth)]
#     Vi_offset = [ch.mean(Vi_list[i], axis=0) for i in range(numTooth)]
#     # print(Vi_offset)
#
#     Vi_center = [(Vi_list[i] - Vi_offset[i]) for i in range(numTooth)]
#
#     Ri_list = [ch.zeros(3) for i in range(numTooth)]
#     ti_list = [ch.zeros(3) for i in range(numTooth)]
#     R_row = ch.zeros(3)
#     t_row = ch.zeros(3)
#     # R_row = ch.array([0, 0, 0.06])
#     # t_row = ch.array([0, 0.06, 0])
#     # R_row = ch.array([0, 0, 0.01])
#     # t_row = ch.array([0, 0.01, 0])
#     V_row = t_row + ch.vstack(
#         [ti_list[i] + Vi_offset[i] + Vi_center[i].dot(Rodrigues(Ri_list[i])) for i in range(numTooth)]).dot(
#         Rodrigues(R_row))
#     # V_comb = ch.vstack([ti_list[i] + Vi_list[i].mean(axis=0) + (Vi_list[i] - Vi_list[i].mean(axis=0)).dot(Rodrigues(Ri_list[i])) for i in range(numTooth)])
#     # V_row = t_row + V_comb.mean(axis=0) + (V_comb - V_comb.mean(axis=0)).dot(Rodrigues(R_row))
#
#     #Projection by OpenDR
#     w, h = (640, 480)
#
#     # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#     # vw = cv2.VideoWriter('result/optimRecord.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (2*w, 2*h))
#
#     # rn = _create_colored_renderer()
#     # imtmp = simple_renderer(rn, m.v, m.f)
#
#     rn = BoundaryRenderer()
#     drn = DepthRenderer()
#     # rn.camera = ProjectPoints(v=V_row, rt=ch.zeros(3), t=ch.array([0, 0, 0]), f=ch.array([w, w]) / 2.,
#     #                           c=ch.array([w, h]) / 2.,
#     #                           k=ch.zeros(5))
#     # rt = ch.array([0, 1, 0]) * np.pi / 4
#     # rn.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([-1.5, 0, 0.5]), f=ch.array([w, w]) / 2.,
#     #                                c=ch.array([w, h]) / 2.,
#     #                                k=ch.zeros(5))
#     # 12681
#     # rt = ch.array([0, -0.3, 0]) * np.pi / 2
#     # rn.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([1.2, 0.2, 0]), f=ch.array([w, w]) / 2.,
#     #                           c=ch.array([w, h]) / 2.,
#     #                           k=ch.zeros(5))
#     # rt = ch.array([0, 0, 0]) * np.pi / 2
#     # rn.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([-0.05, 0.2, -0.25]), f=ch.array([w, w]) / 2.,
#     #                            c=ch.array([w, h]) / 2.,
#     #                            k=ch.zeros(5))
#     # rt = ch.array([0.1, 0.4, 0]) * np.pi / 2
#     # rn.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([-1.4, 0.3, 0.2]), f=ch.array([w, w]) / 2.,
#     #                            c=ch.array([w, h]) / 2.,
#     #                            k=ch.zeros(5))
#     rt = ch.array([-0.9, 0, 0]) * np.pi / 3
#     rn.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([0, -1.5, 0.2]), f=ch.array([w, w]) / 2.,
#                                c=ch.array([w, h]) / 2.,
#                                k=ch.zeros(5))
#     drn.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([0, -1.5, 0.2]), f=ch.array([w, w]) / 2.,
#                               c=ch.array([w, h]) / 2.,
#                               k=ch.zeros(5))
#     # 13282
#     # rt = ch.array([0, -0.25, 0.1]) * np.pi/2
#     # rn.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([0.7, 0.5, 0]), f=ch.array([w, w]) / 2.,
#     #                                c=ch.array([w, h]) / 2.,
#     #                                k=ch.zeros(5))
#     # 13157
#     # rt = ch.array([0, -0.4, 0]) * np.pi/2
#     # rn.camera = ProjectPoints(v=V_row, rt=rt, t=ch.array([1.6, 0.2, 0.3]), f=ch.array([w, w]) / 2.,
#     #                                c=ch.array([w, h]) / 2.,
#     #                                k=ch.zeros(5))
#     # from GuMin_1
#     # rn.camera = ProjectPoints(v=V_row, rt=ch.array([0,0,0]), t=ch.array([0, 0, 0]), f=ch.array([w, w]) / 2.,
#     #                               c=ch.array([w, h]) / 2.,
#     #                               k=ch.zeros(5))
#     rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
#     rn.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
#     drn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
#     drn.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)
#
#     # plt.imshow(drn.r, cmap='gray')
#     # plt.show()
#     #
#     # print drn.r
#
#
#     #Sample points
#     contour1 = deepcopy(rn.r)
#     dmap = deepcopy(drn.r)
#     sample_pts = []
#     # print contour1.size
#
#     cc = 0
#     index = 0
#     frame = []
#     for i in range(640):
#         for j in range(480):
#             if contour1[j, i, 0] > 0 & (cc % 3 == 0):
#                 sample_pts.append([j, i])
#                 contour1[j, i] = [1, 0, 0]
#                 index += 1
#             cc += 1
#             # if (i == 0) | (j == 0) | (i == 639) | (j == 479):
#             #     frame.append([j, i])
#     print index, cc
#
#     # reversed_frame = []
#     # for i in range(2236):
#     #     tmp = [float(frame[i][1] - 320) / 320, float(frame[i][0] - 240) / 320, 1]
#     #     reversed_frame.append(tmp)
#     #
#     # Mesh.save_to_obj('result/frame.obj', reversed_frame, None)
#
#     # camera_view = []
#     # for i in range(10):
#     #     camera_view.append(np.array([0, 0, 1]) * 0.2 * i)
#     # camera_view = np.vstack(camera_view)
#
#     # plt.imshow(contour1)
#     # plt.show()
#     # scipy.misc.imsave('result/contour3.jpg', contour1)
#
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111, projection='3d')
#
#     #Calculate sample points' image coordinates (3D coordinates in projection plane)
#     reversed_imgpts = []
#     for i in range(index):
#         # print sample_pts[i]
#         tmp = [float(sample_pts[i][1]-320)/320, float(sample_pts[i][0]-240)/320, 1]
#         # print tmp
#         reversed_imgpts.append(tmp)
#         # ax.scatter(tmp[0], tmp[1], tmp[2], marker= 'o')
#
#     # plt.show()
#
#     #get depth of reversed image points, calc 3D points
#     Verts_d = []
#     for i in range(index):
#         # print reversed_imgpts[i][0], reversed_imgpts[i][1]
#         d = dmap[sample_pts[i][0], sample_pts[i][1]]
#         tmp = [reversed_imgpts[i][0]*d, reversed_imgpts[i][1]*d, d]
#         Verts_d.append(tmp)
#     Mesh.save_to_obj('result/Verts_depth.obj', Verts_d, None)
#
#
#     #Calculate Vertices coordinates in Camera coordinate system
#     V_row1 = deepcopy(V_row.r)
#     # rt = np.array([0, -0.3, 0]) * np.pi / 2
#     # rt = np.array([0, 0, 0])
#     # rt = np.array([0.1, 0.4, 0]) * np.pi / 2
#     rt = np.array([-0.9, 0, 0]) * np.pi / 3
#     tmpr = R.from_rotvec(rt)
#     # W_to_Cr = np.array([[-1, 0, 0],
#     #                     [0, -1, 0],
#     #                     [0, 0, -1]])
#     # r_mat = tmpr.as_dcm().dot(W_to_Cr)
#     r_mat = tmpr.as_dcm()
#     # t = np.array([1.2, 0.2, 0])
#     # t = np.array([-0.05, 0.2, -0.25]).T
#     # t = np.array([-1.4, 0.3, 0.2])
#     t = np.array([0, -1.5, 0.2]).T
#     cor_mtx = np.zeros((4, 4), dtype='float32')
#     cor_mtx[0:3, 0:3] = r_mat
#     cor_mtx[0:3, 3] = t
#     cor_mtx[3, 3] = 1
#     print cor_mtx
#     inv_cormtx = np.linalg.inv(cor_mtx)
#     print inv_cormtx
#     # wa = np.array([2, 2, 2, 1])
#     # la = wa.dot(cor_mtx)
#     # wa = la.dot(inv_cormtx)
#     # print wa, la
#     V_row_camera = []
#     for i in range(V_row1.size/3):
#         # tmp_v = np.array([V_row1[i][0], V_row1[i][1], V_row1[i][2], 1]).dot(inv_cormtx)
#         # tmp_v = np.matmul(np.array([V_row1[i][0], V_row1[i][1], V_row1[i][2], 1]), cor_mtx)
#         tmp_v = cor_mtx.dot(np.array([V_row1[i][0], V_row1[i][1], V_row1[i][2], 1]).T)
#         V_row_camera.append([tmp_v[0], tmp_v[1], tmp_v[2]])
#     V_row_camera = np.vstack(V_row_camera)
#     # V_row_camera = V_row1.dot(Rodrigues(rt)) + t.T
#     # print V_row_camera.size
#
#     # camera_view_w = []
#     # for i in range(10):
#     #     tmp_c = np.array([camera_view[i][0], camera_view[i][1], camera_view[i][2], 1]).dot(cor_mtx)
#     #     # tmp_c = cor_mtx.dot(np.array([camera_view[i][0], camera_view[i][1], camera_view[i][2], 1]).T)
#     #     camera_view_w.append([tmp_c[0], tmp_c[1], tmp_c[2]])
#     # camera_view_w = np.vstack(camera_view_w)
#     # Mesh.save_to_obj('result/view2_R.obj', camera_view_w, None)
#     # Mesh.save_to_obj('result/view0.obj', camera_view, None)
#
#     # Mesh.save_to_obj('result/V_row_camera3_Rod.obj', V_row_camera, row_mesh.f)
#
#     #Ray tracing to find back-projection 3D vertices
#     mesh_camera = trimesh.Trimesh(vertices=V_row_camera, faces=row_mesh.f)
#     # mesh_camera = trimesh.Trimesh(vertices=V_row1, faces=row_mesh.f)
#     # print mesh_camera.is_empty
#     origins = np.zeros((index, 3), dtype='float32')
#     dirs = np.zeros((index, 3), dtype='float32')
#     dirs += reversed_imgpts
#     intersection_pts, index_ray, index_tri = mesh_camera.ray.intersects_location(origins, dirs, multiple_hits = False)
#
#     # ray_pts = []
#     # for k in range(10):
#     #     cur_pts = deepcopy(origins[k*160])
#     #     for i in range(10):
#     #         tmp = deepcopy(cur_pts)
#     #         ray_pts.append(tmp)
#     #         cur_pts += 0.2*dirs[k*160]
#     # for i in range(10):
#     #     ray_pts.append(np.array([0, 0, 1]) * 0.2 * i)
#     #     ray_pts.append(np.array([0, 1, 0]) * 0.2 * i)
#     #     ray_pts.append(np.array([1, 0, 0]) * 0.2 * i)
#     #
#     # # ray_pts = np.vstack(ray_pts)
#     # # print ray_pts.shape[0]
#     # # print ray_pts
#     # Mesh.save_to_obj('result/ray_pts3.obj', ray_pts, None)
#
#     # print intersection_pts, index_ray, index_tri
#     print origins.shape[0], dirs.shape[0], intersection_pts.shape[0], index_ray.size, index_tri.size
#
#     # for i in range()
#
#     # observed1 = load_image(img1_file_path)
#     # observed2 = load_image(img2_file_path)
#     # observed3 = load_image(img3_file_path)
#     # observed4 = load_image(img4_file_path)
