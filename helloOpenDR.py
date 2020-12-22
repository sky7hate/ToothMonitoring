# Create renderer
from opendr.everything import *
import chumpy as ch
from opendr.renderer import BoundaryRenderer
from opendr.camera import ProjectPoints
import matplotlib.pyplot as plt

# def mesh_opt(mesh, trans, rotation):
#     from chumpy.utils import row, col
#     import numpy as np
#     from copy import deepcopy
#     import cv2
#     # mesh.v = np.asarray(mesh.v, order='C')
#     mesh.vc = mesh.v * 0 + 1
#     mesh.v -= row(np.mean(mesh.v, axis=0))
#     mesh.v /= np.max(mesh.v)
#     mesh.v *= 2.0
#
#     mesh.v = mesh.v.dot(cv2.Rodrigues(np.asarray(np.array(rotation), np.float64))[0])
#     mesh.v = mesh.v + row(np.asarray(trans))
#     return mesh
#
# rn = BoundaryRenderer()
#
# # Assign attributes to renderer
# from opendr.util_tests import get_earthmesh
# # m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
# m = load_mesh('upperdeform_simplified.obj')
# m = mesh_opt(m, trans=ch.array([0, 0, 3.5]), rotation=ch.zeros(3))
# print(m.v)
#
# w, h = (800, 300)
# rn.camera = ProjectPoints(v=m.v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,w])/2., c=ch.array([w,h])/2., k=ch.zeros(5))
# rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
# # rn.set(v=m.v, f=m.f, vc=m.vc*0+1, bgcolor=ch.zeros(3), num_channels=3)
# rn.set(v=m.v, f=m.f, vc=m.vc*0+1, bgcolor=ch.zeros(3), num_channels=3)
#
# # Show it
# plt.ion()
# plt.imshow(rn.r)
# plt.show()
# plt.pause(1000)
#
# dr = rn.dr_wrt(rn.v) # or rn.vc, or rn.camera.rt, rn.camera.t, rn.camera.f, rn.camera.c, etc


###demo: silhouette

# # Create renderer
# import chumpy as ch
# from opendr.renderer import ColoredRenderer
# rn = ColoredRenderer()

# # Assign attributes to renderer
# from opendr.util_tests import get_earthmesh
# # m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
# m = load_mesh('/home/wgd/PycharmProjects/MultiviewFitting/upperdeform_simplified.obj')
# print(m.v.shape)
# m = mesh_opt(m, trans=ch.array([0, 0, 4]), rotation=ch.zeros(3))
# print(m.v.shape)
#
# w, h = (320, 240)
# from opendr.camera import ProjectPoints
# rn.camera = ProjectPoints(v=m.v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,w])/2., c=ch.array([w,h])/2., k=ch.zeros(5))
# rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
# rn.set(v=m.v, f=m.f, vc=m.vc*0+1, bgcolor=ch.zeros(3))
#
# # Show it
# import matplotlib.pyplot as plt
# plt.ion()
# plt.imshow(rn.r)
# plt.show()
#
# dr = rn.dr_wrt(rn.v) # or rn.vc, or rn.camera.rt, rn.camera.t, rn.camera.f, rn.camera.c, etc
# plt.waitforbuttonpress()



###demos['optimization'] = """
from opendr.simple import *
import numpy as np
import matplotlib.pyplot as plt
w, h = 320, 240

try:
    m = load_mesh('earth.obj')
except:
    from opendr.util_tests import get_earthmesh
    m = get_earthmesh(trans=ch.array([0,0,0]), rotation=ch.zeros(3))

# Create V, A, U, f: geometry, brightness, camera, renderer
V = ch.array(m.v)
A = SphericalHarmonics(vn=VertNormals(v=V, f=m.f),
                       components=[3.,2.,0.,0.,0.,0.,0.,0.,0.],
                       light_color=ch.ones(3))
U = ProjectPoints(v=V, f=[w,w], c=[w/2.,h/2.], k=ch.zeros(5),
                  t=ch.zeros(3), rt=ch.zeros(3))
f = TexturedRenderer(vc=A, camera=U, f=m.f, bgcolor=[0.,0.,0.],
                     texture_image=m.texture_image, vt=m.vt, ft=m.ft,
                     frustum={'width':w, 'height':h, 'near':1,'far':20})


# Parameterize the vertices
translation, rotation = ch.array([0,0,8]), ch.zeros(3)
f.v = translation + V.dot(Rodrigues(rotation))

observed = f.r
np.random.seed(1)
translation[:] = translation.r + np.random.rand(3)
rotation[:] = rotation.r + np.random.rand(3) *.2
A.components[1:] = 0

# Create the energy
E_raw = f - observed
E_pyr = gaussian_pyramid(E_raw, n_levels=6, normalization='size')

def cb(_):
    import cv2
    global E_raw
    cv2.imshow('Absolute difference', np.abs(E_raw.r))
    cv2.waitKey(1)

print 'OPTIMIZING TRANSLATION, ROTATION, AND LIGHT PARMS'
free_variables=[translation, rotation, A.components]
ch.minimize({'pyr': E_pyr}, x0=free_variables, callback=cb)
ch.minimize({'raw': E_raw}, x0=free_variables, callback=cb)