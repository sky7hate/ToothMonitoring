import opendr
import chumpy as ch
from opendr.everything import *
from opendr.renderer import ColoredRenderer
from opendr.renderer import BoundaryRenderer
from opendr.renderer import TexturedRenderer
import Mesh
from copy import deepcopy
import numpy as np
import scipy.misc


def check_around(colormap, checkpixel):
    if colormap[checkpixel[0], checkpixel[1]][1]>0:
        return True
    return False

# rn = ColoredRenderer()
rn = BoundaryRenderer()
# rn = TexturedRenderer()
# mesh = Mesh.TeethRowMesh(r'/home/jiaming/MultiviewFitting/data/teeth_gum', False)
# mesh = Mesh.TeethRowMesh(r'/home/jiaming/MultiviewFitting/data/ForJiaming(upper segmented)/HBF_13282/before', False)
# mesh_t = Mesh.TeethRowMesh(r'/home/jiaming/MultiviewFitting/data/upper_teeth', False)
mesh = Mesh.TeethRowMesh(r'/home/jiaming/MultiviewFitting/data/seg/newU_before', False)
# m_t = mesh_t.row_mesh
# v_t = ch.array(m_t.v)
# f_t = m_t.f
#mesh = Mesh.TeethRowMesh(r'/home/jiaming/Teeth Monitoring/gcn/data/Teeth/teeth_gum.ply', True, False)
#mesh = Mesh.TeethRowMesh('/home/jiaming/MultiviewFitting/result/observation/movedRow.obj', True)
m = mesh.row_mesh
v = ch.array(m.v)
vc = ch.array(m.vc) # albedo
f = m.f
# Mesh.save_to_obj('data/observation/from GuMin/movedRow_real1.obj', v, f)
from opendr.camera import ProjectPoints
w, h = (640, 480)

##### contour 0 initial camera pose ######
# rt = ch.array([0, -3.6, -0.3]) * np.pi/4
# rn.camera = ProjectPoints(v=v, rt=rt, t=ch.array([1.25, -0.35, 5]), f=ch.array([w, w])*4/4.8,
#                                c=ch.array([w, h]) / 2.,
#                                k=ch.zeros(5))
# ##### contour 1 initial camera pose ######
# rt = ch.array([0.3, -2, -0.1]) * np.pi/4
# rn.camera = ProjectPoints(v=v, rt=rt, t=ch.array([2.1, 0.3, 3]), f=ch.array([w, w])*4/4.8,
#                                c=ch.array([w, h]) / 2.,
#                                k=ch.zeros(5))
# ##### contour 2 initial camera pose ######
# rt = ch.array([0.4, -0.9, 0.2]) * np.pi/4
# rn.camera = ProjectPoints(v=v, rt=rt, t=ch.array([1.2, 0.9, 2]), f=ch.array([w, w])*4/4.8,
#                                c=ch.array([w, h]) / 2.,
#                                k=ch.zeros(5))
# ##### contour 3 initial camera pose ######
# rt = ch.array([1.5, -2, -1.3]) * np.pi/4
# rn.camera = ProjectPoints(v=v, rt=rt, t=ch.array([2.1, 0.2, 3.5]), f=ch.array([w, w])*4/4.8,
#                                c=ch.array([w, h]) / 2.,
#                                k=ch.zeros(5))
###### new c0 ######
# rt = ch.array([0.3, 0, 0]) * np.pi/4
# rn.camera = ProjectPoints(v=v, rt=rt, t=ch.array([0, 0, 3]), f=ch.array([w, w])*4/4.8,
#                                c=ch.array([w, h]) / 2.,
#                                k=ch.zeros(5))
###### new c1 ######
# rt = ch.array([0.2, -1.55, 0.1]) * np.pi/4
# rn.camera = ProjectPoints(v=v, rt=rt, t=ch.array([0.5, 0, 3]), f=ch.array([w, w])*4/4.8,
#                                c=ch.array([w, h]) / 2.,
#                                k=ch.zeros(5))
###### new c2 ######
# rt = ch.array([0.3, 1.75, 0]) * np.pi/4
# rn.camera = ProjectPoints(v=v, rt=rt, t=ch.array([-0.3, 0, 3]), f=ch.array([w, w])*4/4.8,
#                                c=ch.array([w, h]) / 2.,
#                                k=ch.zeros(5))
###### new c3 ######
# rt = ch.array([0.3, 0.2, 0.1]) * np.pi/4
# rn.camera = ProjectPoints(v=v, rt=rt, t=ch.array([-0.1, 0, 3]), f=ch.array([w, w])*4/4.8,
#                                c=ch.array([w, h]) / 2.,
#                                k=ch.zeros(5))
###### new c4 ######
# rt = ch.array([0.4, -0.45, 0]) * np.pi/4
# rn.camera = ProjectPoints(v=v, rt=rt, t=ch.array([0.3, 0, 3]), f=ch.array([w, w])*4/4.8,
#                                c=ch.array([w, h]) / 2.,
#                                k=ch.zeros(5))
###### new c5 ######
rt = ch.array([-2.5, 0, 0]) * np.pi/4
rn.camera = ProjectPoints(v=v, rt=rt, t=ch.array([0, 0, 3]), f=ch.array([w, w])*4/4.8,
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))

# rt = ch.array([-0.9, 0, 0]) * np.pi / 3
# rn.camera = ProjectPoints(v=v, rt=rt, t=ch.array([0, -1.5, 0.2]), f=ch.array([w, w]) / 2.,
#                           c=ch.array([w, h]) / 2.,
#                           k=ch.zeros(5))

rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn.set(v=v, f=f, vc=vc, bgcolor=ch.zeros(3), num_channels=3)
# rn.v = v
# rn.f = f
# rn.bgcolor = ch.array([0, 0, 0])
# rn.num_channels=3
#rn.vc = m.vc

# from opendr.lighting import LambertianPointLight
# rn.vc = LambertianPointLight(
#     f=f,
#     v=v,
#     num_verts=len(v),
#     # light_pos=ch.array([0,0,-1000]),
#     # light_pos=ch.array([-1000,0,0]),
#     light_pos=ch.array([0,1000,0]),
#     # light_pos=ch.array([1000,0,-1000]),
#     # light_pos=ch.array([1000,0,-1000]),
#     # light_pos=ch.array([0,-1000,0]),
#     vc=vc,
#     light_color=ch.array([1., 1., 1.]))

color = load_image(r'result/colormap2.jpg')
newc = np.zeros([480,640,3])

f_sample_pts = []
contour = deepcopy(rn.r)

f_sample_pts = np.argwhere(contour[:, :, 0] > 0)

f_sample_pts = np.vstack(f_sample_pts)

sp_pts = []
for i in range(f_sample_pts.shape[0]):
    if check_around(color, f_sample_pts[i]):
        for j in range(-1, 1, 1):
            for k in range(-1, 1, 1):
                contour[f_sample_pts[i][0]+j, f_sample_pts[i][1]+k] = color[f_sample_pts[i][0], f_sample_pts[i][1]]
    else:
        contour[f_sample_pts[i][0], f_sample_pts[i][1]] = [0, 0, 0]

newc[contour[:, :, 0] > 0] = color[contour[:, :, 0] > 0]



# contour = deepcopy(rn.r)
scipy.misc.imsave('result/colored_contour2.jpg', contour)
scipy.misc.imsave('result/colored_contour3.jpg', newc)



# fc = [1, 1, 1] - color_contour

# observed = load_image(r'data/observation/real_contour3_f.jpg')
#
# ob1_dc = deepcopy(observed)
# ob1_dc[ob1_dc[:, :, 0] > 0] *= [0, 1, 0]
# rn1_dc = deepcopy(rn.r)
# rn1_dc[rn1_dc[:, :, 0] > 0] *= [1, 0, 0]



import matplotlib.pyplot as plt
# plt.ion()
# plt.imshow(rn1_dc+ob1_dc)
# plt.imshow(contour)
# plt.draw()
# plt.show()