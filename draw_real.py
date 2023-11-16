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
import matplotlib.gridspec as gridspec
import scipy.misc
import time
import cv2
from config import cfg
import os
from scipy.spatial.transform import Rotation as R
from PIL import Image
from opendr.lighting import LambertianPointLight



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
    img1_file_path = 'data/observation/real/front_l.jpg'
    img2_file_path = 'data/observation/real/right_l.jpg'
    img3_file_path = 'data/observation/real/left_l.jpg'
    img4_file_path = 'data/observation/real/crop_upper.jpg'
    img5_file_path = 'data/observation/real/crop_lower.jpg'
	gt_list = None
	rt1 = ch.array([0.3, -1.6, -0.015]) * np.pi / 4
    t1 = ch.array([0.557, -0.20, 3.1])
    rt2 = ch.array([0.43, 0, 0.03]) * np.pi / 4
    t2 = ch.array([0.08, 0.03, 2.9])
    rt3 = ch.array([0.5, 1.78, 0.16]) * np.pi / 4
    t3 = ch.array([-0.35, -0.28, 3.35])
    rt4 = ch.array([-2.36, 0.05, 0.05]) * np.pi / 4
    t4 = ch.array([-0.055, -0.39, 2.3])
    rt5 = ch.array([1.55, -0.07, 0.05]) * np.pi / 4
    t5 = ch.array([0.018, 0.07, 2.25])
    w, h = (640, 480)
    f = w * 4 / 4.8
	
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

    print(teeth_file_folder)

    teeth_row_mesh = Mesh.TeethRowMesh(teeth_file_folder, False)
    row_mesh = teeth_row_mesh.row_mesh
    # merged_mesh = teeth_row_mesh.row_mesh

    # lower teeth
    teeth_row_mesh_l = Mesh.TeethRowMesh(teeth_file_folder_low, False)
    row_mesh_l = teeth_row_mesh_l.row_mesh
    numTooth_l = len(teeth_row_mesh_l.mesh_list)
    V_row_l = ch.array(row_mesh_l.v)

    teeth_row_mesh_l.rotate(np.array([0, 0, np.pi]))
    teeth_row_mesh_l.translate(np.array([0.01, 0.13, -0.04]))

    merged_mesh = Mesh.merge_mesh(row_mesh, row_mesh_l)
    V_row_bite = ch.array(merged_mesh.v)

    numTooth = len(teeth_row_mesh.mesh_list)
    Ri_list = [ch.zeros(3) for i in range(numTooth)]
    ti_list = [ch.zeros(3) for i in range(numTooth)]
   

    Vi_list = [ch.array(teeth_row_mesh.mesh_list[i].v) for i in range(numTooth)]
    Vi_offset = [ch.mean(Vi_list[i], axis=0) for i in range(numTooth)]
    # print(Vi_offset)

    Vi_center = [(Vi_list[i] - Vi_offset[i]) for i in range(numTooth)]

    V_row = ch.vstack([Vi_list[i] for i in range(numTooth)])

    # Mesh.save_to_obj('result/bite/upper_initial.obj', V_row.r, row_mesh.f)
    # print 0

    # f = 320
    w, h = (640, 480)
    # w, h = (4032, 3024)
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # vw = cv2.VideoWriter('result/optimRecord.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (2*w, 2*h))

    # rn = _create_colored_renderer()
    # imtmp = simple_renderer(rn, m.v, m.f)

    rn = BoundaryRenderer()
    # rn = ColoredRenderer()
    # drn = DepthRenderer()
    rn.camera = ProjectPoints(v=V_row_bite, rt=rt, t=t, f=ch.array([w, w]) * 4 / 4.8,
                              c=ch.array([w, h]) / 2.,
                              k=ch.zeros(5))
    
    rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc, bgcolor=ch.zeros(3), num_channels=3)

    # rn.vc = LambertianPointLight(
    #     v=V_row_bite,
    #     f=merged_mesh.f,
    #     num_verts=len(V_row),
    #     light_pos=ch.array([-1000, 0, -1000]),
    #     vc=merged_mesh.vc,
    #     light_color=ch.array([1., 1., 1.]))

    rn2 = BoundaryRenderer()
    # rn2 = ColoredRenderer()
    rn2.camera = ProjectPoints(v=V_row_bite, rt=rt2, t=t2, f=ch.array([w, w]) * 4 / 4.8,
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))
    rn2.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn2.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc, bgcolor=ch.zeros(3), num_channels=3)
    # rn2.vc = LambertianPointLight(
    #     v=V_row_bite,
    #     f=merged_mesh.f,
    #     num_verts=len(V_row),
    #     light_pos=ch.array([0, 0, -1000]),
    #     vc=merged_mesh.vc,
    #     light_color=ch.array([1., 1., 1.]))

    rn3 = BoundaryRenderer()
    # rn3 = ColoredRenderer()
    rn3.camera = ProjectPoints(v=V_row_bite, rt=rt3, t=t3, f=ch.array([w, w]) * 4 / 4.8,
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))
   
    rn3.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn3.set(v=V_row_bite, f=merged_mesh.f, vc=merged_mesh.vc, bgcolor=ch.zeros(3), num_channels=3)
    # rn3.vc = LambertianPointLight(
    #     v=V_row,
    #     f=row_mesh.f,
    #     num_verts=len(V_row),
    #     light_pos=ch.array([0, 1000, -1000]),
    #     vc=row_mesh.vc,
    #     light_color=ch.array([1., 1., 1.]))

    rn4 = BoundaryRenderer()
    # rn4 = ColoredRenderer()
    rn4.camera = ProjectPoints(v=V_row, rt=rt4, t=t4, f=ch.array([w, w]) * 4 / 4.8,
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))

    rn4.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn4.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc, bgcolor=ch.zeros(3), num_channels=3)
    # rn4.vc = LambertianPointLight(
    #     v=V_row_bite,
    #     f=merged_mesh.f,
    #     num_verts=len(V_row),
    #     light_pos=ch.array([1000, 0, -1000]),
    #     vc=merged_mesh.vc,
    #     light_color=ch.array([1., 1., 1.]))

    rn5 = BoundaryRenderer()
    # rn5 = ColoredRenderer()
    rn5.camera = ProjectPoints(v=V_row_l, rt=rt5, t=t5, f=ch.array([w, w]) * 4 / 4.8,
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))

    rn5.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn5.set(v=V_row_l, f=row_mesh_l.f, vc=row_mesh_l.vc, bgcolor=ch.zeros(3), num_channels=3)
    # rn5.vc = LambertianPointLight(
    #     v=V_row,
    #     f=row_mesh.f,
    #     num_verts=len(V_row),
    #     light_pos=ch.array([1000, 0, -1000]),
    #     vc=row_mesh.vc,
    #     light_color=ch.array([1., 1., 1.]))

    # rn6 = BoundaryRenderer()
    # rn6 = ColoredRenderer()
    # rn6.camera = ProjectPoints(v=V_row, rt=rt6, t=t6, f=ch.array([w, w]) * 4 / 4.8,
    #                            c=ch.array([w, h]) / 2.,
    #                            k=ch.zeros(5))
    # rn6.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    # rn6.set(v=V_row, f=row_mesh.f, vc=row_mesh.vc, bgcolor=ch.zeros(3), num_channels=3)
    # rn6.vc = LambertianPointLight(
    #     v=V_row,
    #     f=row_mesh.f,
    #     num_verts=len(V_row),
    #     light_pos=ch.array([0, 1000, 0]),
    #     vc=row_mesh.vc,
    #     light_color=ch.array([1., 1., 1.]))

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

    def drawcolor():
        scipy.misc.imsave('result/real_pose1/color_%05d.png' % (1), rn.r)
        scipy.misc.imsave('result/real_pose1/color_%05d.png' % (2), rn2.r)
        scipy.misc.imsave('result/real_pose1/color_%05d.png' % (3), rn3.r)
        # scipy.misc.imsave('result/real_pose1/color_%05d.png' % (4), rn4.r)
        # scipy.misc.imsave('result/real_pose1/color_%05d.png' % (5), rn5.r)
        # scipy.misc.imsave('result/real_pose1/color_%05d.png' % (6), rn6.r)

    # drawcolor()


    def drawfig(iter):

        ob1_dc = deepcopy(observed1)
        ob2_dc = deepcopy(observed2)
        ob3_dc = deepcopy(observed3)
        ob4_dc = deepcopy(observed4)
        ob5_dc = deepcopy(observed5)
        # ob6_dc = deepcopy(observed6)

        rn1_dc = deepcopy(rn.r)
        rn2_dc = deepcopy(rn2.r)
        rn3_dc = deepcopy(rn3.r)
        rn4_dc = deepcopy(rn4.r)
        rn5_dc = deepcopy(rn5.r)
        # rn6_dc = deepcopy(rn6.r)

        ob1_dc[rn1_dc[:, :, 0] > 0] *= [1, 0, 0]
        ob2_dc[rn2_dc[:, :, 0] > 0] *= [1, 0, 0]
        ob3_dc[rn3_dc[:, :, 0] > 0] *= [1, 0, 0]
        ob4_dc[rn4_dc[:, :, 0] > 0] *= [1, 0, 0]
        ob5_dc[rn5_dc[:, :, 0] > 0] *= [1, 0, 0]
        # ob6_dc[rn6_dc[:, :, 0] > 0] *= [1, 0, 0]

        # scipy.misc.imsave('result/log1/fittingresult1_iter{}.jpg'.format(iter), rn1_dc + ob1_dc)
        # scipy.misc.imsave('result/real_pose1/fittingresult1_iter{}a.jpg'.format(iter), rn1_dc)
        scipy.misc.imsave('result/render_data/prediction/fittingresult1_iter{}t.jpg'.format(iter), ob1_dc)
        # scipy.misc.imsave('result/log1/fittingresult2_iter{}.jpg'.format(iter), rn2_dc + ob2_dc)
        # scipy.misc.imsave('result/real_pose1/fittingresult2_iter{}a.jpg'.format(iter), rn2_dc)
        scipy.misc.imsave('result/render_data/prediction/fittingresult2_iter{}t.jpg'.format(iter), ob2_dc)
        # scipy.misc.imsave('result/log1/fittingresult3_iter{}.jpg'.format(iter), rn3_dc + ob3_dc)
        # scipy.misc.imsave('result/real_pose1/fittingresult3_iter{}a.jpg'.format(iter), rn3_dc)
        scipy.misc.imsave('result/render_data/prediction/fittingresult3_iter{}t.jpg'.format(iter), ob3_dc)
        # scipy.misc.imsave('result/log1/fittingresult4_iter{}.jpg'.format(iter), rn4_dc + ob4_dc)
        # scipy.misc.imsave('result/real_pose1/fittingresult4_iter{}a.jpg'.format(iter), rn4_dc)
        scipy.misc.imsave('result/render_data/prediction/fittingresult4_iter{}t.jpg'.format(iter), ob4_dc)
        # scipy.misc.imsave('result/log1/fittingresult5_iter{}.jpg'.format(iter), rn5_dc + ob5_dc)
        # scipy.misc.imsave('result/log1/fittingresult5_iter{}a.jpg'.format(iter), rn5_dc)
        scipy.misc.imsave('result/render_data/prediction/fittingresult5_iter{}t.jpg'.format(iter), ob5_dc)
        # scipy.misc.imsave('result/log1/fittingresult6_iter{}.jpg'.format(iter), rn6_dc + ob6_dc)
        # scipy.misc.imsave('result/log1/fittingresult6_iter{}a.jpg'.format(iter), rn6_dc)
        # scipy.misc.imsave('result/real_pose1/fittingresult6_iter{}b.jpg'.format(iter), ob6_dc)

        print('fig saved')



    print 1
    drawfig(0)
    print 2




