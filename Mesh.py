from opendr.everything import *
import numpy as np
from copy import deepcopy
import cv2
from chumpy.utils import row, col
import chumpy as ch
import vtk



class Mesh(object):
    def __init__(self, filename):
        self.mesh = self.load(filename, trans=ch.array([0, 0, 4]), rotation=ch.zeros(3))  # only support .obj and .ply file
        self.staticMesh = deepcopy(self.mesh)  # the mesh may be changed, so store the original mesh

    @staticmethod
    def load(filename, trans=ch.zeros(3), rotation=ch.zeros(3)):
        mesh = load_mesh(filename)
        # mesh.v = np.asarray(mesh.v, order='C')
        mesh.vc = mesh.v * 0 + 1
        mesh.v -= row(np.mean(mesh.v, axis=0))
        mesh.v /= np.max(mesh.v)
        mesh.v *= 2.0

        mesh.v = mesh.v.dot(cv2.Rodrigues(np.asarray(np.array(rotation), np.float64))[0])
        mesh.v = mesh.v + row(np.asarray(trans))
        return mesh

    def translation(self, rt):
        self.mesh.v = self.mesh.v + row(np.asarray(rt))

    def rotation(self, rv):
        self.mesh.v = self.mesh.v.dot(cv2.Rodrigues(np.asarray(np.array(rv), np.float64))[0])


import glob
import os
from opendr.dummy import Minimal

class TeethRowMesh(object):
    def __init__(self, folder, moved, normalized = True):
        if moved == True:
            self.file_name_list = folder
            self.row_mesh = self.load(self.file_name_list)
            self.max_v = np.max(self.row_mesh.v)
            if not normalized:
                self.normalize()
        else:
            self.file_name_list = glob.glob(folder+'/*.ply')
            self.file_name_list.sort()
            self.mesh_list = [self.load(f) for f in self.file_name_list]
            self.row_mesh = self.get_teethrow_mesh()
            self.start_idx_list = []
            self.faces_num = []
            self.max_v = np.max(self.row_mesh.v)
        # can be calculated as the center of each tooth
        #self.positions_in_row = self.read_tooth_origin(glob.glob(folder+'/*.txt')[0])

            numVerts = 0
            numFaces = 0
            for i, m in enumerate(self.mesh_list):
                self.start_idx_list.append(numVerts)
                numVerts += m.v.shape[0]
                self.faces_num.append(numFaces)
                numFaces += m.f.shape[0]
            self.start_idx_list.append(numVerts)
            self.faces_num.append(numFaces)

            self.view_init()

    def view_init(self):
        mean = row(np.mean(self.row_mesh.v, axis=0))
        max_v = np.max(np.abs(self.row_mesh.v))
        # mean = [[-7.44505049, -2.88865201, 5.98988344]]
        # max_v = 39.6236457824707
        print mean
        print max_v
        # apply the same operation to the teeth_row, each individual tooth and their origin positions
        for m in self.mesh_list:
            m.v -= mean
            m.v /= max_v
            # m.v *= 2.0

        self.row_mesh.v -= mean
        self.row_mesh.v /= max_v
        # self.row_mesh.v *= 10.0

        #13282
        self.rotate(ch.array([-np.pi/2, 0, 0]))
        self.rotate(ch.array([0, -np.pi/2, 0]))
        self.translate(ch.array([0, 0, 2]))
        #from GuMin
        # self.rotate(ch.array([0, 0, -np.pi]))
        # self.rotate(ch.array([0, -np.pi, 0]))
        # self.translate(ch.array([0, 0, 1.5]))

        # self.rotate(ch.array([0, 0, 0]))
        # self.rotate(ch.array([0, 0, np.pi / 2 * 0.98]))
        # self.translate(ch.array([0, 0, 3]))

    def normalize(self):
        mean = row(np.mean(self.row_mesh.v, axis=0))
        max_v = np.max(np.abs(self.row_mesh.v))
        print max_v
        # m.v -= mean
        # m.v /= max_v
        # m.v *= 2.0

        self.row_mesh.v -= mean
        self.row_mesh.v /= max_v
        #self.row_mesh.v *= 2.0

        # self.rotate(ch.array([0, 0, 0]))
        # self.translate(ch.array([0, 0, 4]))
        self.row_mesh.v = self.row_mesh.v.dot(cv2.Rodrigues(np.asarray(np.array(ch.array([np.pi, 0, 0])), np.float64))[0])
        self.row_mesh.v += row(np.asarray(ch.array([0, 0, 4])))

    def reverse_viewinit(self):
        for m in self.mesh_list:
            m.v /= 2.0
            m.v *= self.max_v

        self.row_mesh.v /= 2.0
        self.row_mesh.v *= self.max_v

    def get_teethrow_mesh(self):
        mesh = Minimal()
        # mesh.v = np.asarray(verts, order='C')
        # mesh.v = np.vstack([m.v for i, m in enumerate(self.mesh_list)])

        numVerts = 0
        faces_list = []
        verts_list = []
        vc_list = []

        i = 0

        for m in self.mesh_list:
            # if i == 0:
            #     m.vc = m.v * 0 + [1, 0.85, 0.85]
            #     i = 1
            # else:
            #     m.vc = m.v * 0. + 1
            m.vc = m.v * 0 + [0.05*i, 1, 1]
            i += 1
            verts_list.append(m.v)
            faces_list.append(m.f + numVerts)
            vc_list.append(m.vc)
            numVerts += m.v.shape[0]

        mesh.v = np.vstack(verts_list)
        mesh.f = np.vstack(faces_list)
        mesh.vc = np.vstack(vc_list)

        print('TeethRow with #V={}, #F={}, #vc={}'.format(mesh.v.shape, mesh.f.shape, mesh.vc.shape))
        return mesh

    @staticmethod
    def read_tooth_origin(filename):
        import pandas as pd
        # print(filename)
        data = pd.read_csv(filename, sep=" ", header=None).values[:, 1::].astype(np.float)
        # print(data)
        return data

    @staticmethod
    def load(filename):
        """
          Support all file format by using vtk file reader.
        """
        polyData = ReadPolyData(filename)
        verts = []
        for i in range(polyData.GetNumberOfPoints()):
            verts.append(np.array(polyData.GetPoint(i), dtype=np.float))
        verts = np.vstack(verts)
        # print(verts)
        # print(verts.shape)

        faces = []
        triangles = polyData.GetPolys().GetData()
        for i in range(polyData.GetNumberOfCells()):
            faces.append(np.array([int(triangles.GetValue(j)) for j in range(4 * i + 1, 4 * i + 4)]))
        faces = np.vstack(faces)
        # print(polyData.GetNumberOfCells())
        # print(faces)

        mesh = Minimal()
        # mesh.v = np.asarray(verts, order='C')
        mesh.v = verts
        mesh.f = faces
        mesh.vc = verts * 0. + 1


        # print('Loaded file: {}\n#Vertices={}, #Faces={}'.format(filename, verts.shape[0], faces.shape[0]))
        return mesh

    def translate(self, rt, which=None):
        if which is not None:
            self.mesh_list[which].v += row(np.asarray(rt))
            # update the individual change in teeth row
            self.row_mesh.v[self.start_idx_list[which]:self.start_idx_list[which + 1], :] = self.mesh_list[which].v
            return

        self.row_mesh.v += row(np.asarray(rt))
        for m in self.mesh_list:
            m.v += row(np.asarray(rt))

    def rotate(self, rv, which=None):
        if which is not None:
            mean = np.mean(self.mesh_list[which].v, axis=0, keepdims=True)
            self.mesh_list[which].v -= mean
            self.mesh_list[which].v = self.mesh_list[which].v.dot(Rodrigues(np.array(rv)))
            # self.mesh_list[which].v = Rodrigues(np.array(rv)).dot(self.mesh_list[which].v)
            self.mesh_list[which].v += mean
            # update the individual change in teeth row
            self.row_mesh.v[self.start_idx_list[which]:self.start_idx_list[which + 1], :] = self.mesh_list[which].v
            return

        mean = np.mean(self.row_mesh.v, axis=0)
        # print mean, self.row_mesh.v.shape
        self.row_mesh.v = (self.row_mesh.v-mean).dot(Rodrigues(np.array(rv)))+mean
        # self.row_mesh.v = Rodrigues(np.array(rv)).dot((self.row_mesh.v - mean)) + mean
        # print self.row_mesh.v.shape
        for m in self.mesh_list:
            m.v = (m.v-mean).dot(Rodrigues(np.array(rv))) + mean
            # m.v = Rodrigues(np.array(rv)).dot((m.v - mean)) + mean


def ReadPolyData(file_name):
    import os
    path, extension = os.path.splitext(file_name)
    extension = extension.lower()
    if extension == ".ply":
        reader = vtk.vtkPLYReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".vtp":
        reader = vtk.vtkXMLpoly_dataReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".obj":
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".vtk":
        reader = vtk.vtkpoly_dataReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".g":
        reader = vtk.vtkBYUReader()
        reader.SetGeometryFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    else:
        # Return a None if the extension is unknown.
        poly_data = None
    return poly_data


def save_to_obj(filename, verts, faces=None):
        """ write the verts and faces on file."""
        # print('#Vertices={}, #Faces={}'.format(verts.shape[0], faces.shape[0]))
        # print('#Faces = {}'.format(faces.shape[0]))

        with open(filename, 'w') as f:
            # write vertices
            f.write('g\n# %d vertex\n' % len(verts))
            for vert in verts:
                f.write('v %f %f %f\n' % tuple(vert))

            # write faces
            if faces is not None:
                f.write('# %d faces\n' % len(faces))
                for face in faces:
                    f.write('f %d %d %d\n' % tuple(face + 1))
            # if (faces != None):
            #     # write faces
            #     f.write('# %d faces\n' % len(faces))
            #     for face in faces:
            #         f.write('f %d %d %d\n' % tuple(face+1))



############################################################################################################

colors = {
    'pink': [.7, .7, .9],
    'neutral': [.9, .9, .8],
    'capsule': [.7, .75, .5],
    'yellow': [.5, .7, .75],
}


def _create_colored_renderer(w=640,
                     h=480,
                     rt=np.zeros(3),
                     t=np.zeros(3),
                     f=None,
                     c=None,
                     k=None,
                     near=.5,
                     far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn


def simple_renderer(rn, verts, faces, yrot=np.radians(120)):

    # Rendered model color
    color = colors['pink']

    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))

    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))

    return rn.r


def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)

from opendr.renderer import BoundaryRenderer
from opendr.renderer import ColoredRenderer
from opendr.camera import ProjectPoints
import matplotlib.pyplot as plt
import scipy.misc
if __name__ == '__main__':

    teeth_row_mesh = TeethRowMesh('/home/jiaming/Teeth Monitoring/gcn/data/Teeth/upper', False)
    # m = teeth_row_mesh.mesh_list[2] # render an individual tooth
    # m.v -= row(np.mean(m.v, axis=0))
    # m.v += row(np.asarray([0, 0, 4]))

    m = teeth_row_mesh.row_mesh
    # save_to_obj('result/observation/oriRow.obj', m.v, m.f)

    teeth_row_mesh.translate(ch.array([0, 0.1, 0]), 6)  # translate an individual tooth
    teeth_row_mesh.translate(ch.array([0, 0.1, 0]), 10)  # translate an individual tooth
    teeth_row_mesh.rotate(ch.array([0, 0.1, 0]), 7)  # rotate 1 means 60 degree
    teeth_row_mesh.rotate(ch.array([0, -0.1, 0]), 3)  # rotate 1 means 60 degree


    w, h = (640, 480)
    rn = BoundaryRenderer()
    rn.camera = ProjectPoints(v=m.v, rt=ch.zeros(3), t=ch.array([0, 0, 0]), f=ch.array([w, w]) / 2., c=ch.array([w, h]) / 2.,
                              k=ch.zeros(5))
    rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn.set(v=m.v, f=m.f, vc=m.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)

    rn2 = BoundaryRenderer()
    rt = ch.array([0, 1, 0]) * np.pi/1.5
    rn2.camera = ProjectPoints(v=m.v, rt=rt, t=ch.array([-2.5, -0., 5]), f=ch.array([w, w]) / 2.,
                              c=ch.array([w, h]) / 2.,
                              k=ch.zeros(5))
    rn2.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    rn2.set(v=m.v, f=m.f, vc=m.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)

    rn3 = BoundaryRenderer()
    rn3.camera = ProjectPoints(v=m.v, rt=-ch.array([0, 1, 0]) * np.pi / 1.5, t=ch.array([2.5, 0., 5]), f=ch.array([w, w]) / 2.,
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))
    rn3.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    rn3.set(v=m.v, f=m.f, vc=m.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)

    rn4 = BoundaryRenderer()
    rn4.camera = ProjectPoints(v=m.v, rt=ch.array([-0.1, 0.8, 0.2]) * np.pi, t=ch.array([-1., -1.5, 6.]),
                               f=ch.array([w, w]) / 2.,
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))
    rn4.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    rn4.set(v=m.v, f=m.f, vc=m.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)

    rn5 = BoundaryRenderer()
    rn5.camera = ProjectPoints(v=m.v, rt=-ch.array([-0.1, 0.8, 0.2]) * np.pi/1.2, t=ch.array([2.5, -1.5, 5.]),
                               f=ch.array([w, w]) / 2.,
                               c=ch.array([w, h]) / 2.,
                               k=ch.zeros(5))
    rn5.frustum = {'near': 1., 'far': 100., 'width': w, 'height': h}
    rn5.set(v=m.v, f=m.f, vc=m.vc * 0 + 1, bgcolor=ch.zeros(3), num_channels=3)

    # rn = _create_colored_renderer()
    # imtmp = simple_renderer(rn, m.v, m.f)

    save_to_obj('result/observation/movedRow.obj', m.v, m.f)
    scipy.misc.imsave('result/observation/movedRow1.jpg', rn.r)
    scipy.misc.imsave('result/observation/movedRow2.jpg', rn2.r)
    scipy.misc.imsave('result/observation/movedRow3.jpg', rn3.r)
    scipy.misc.imsave('result/observation/movedRow4.jpg', rn4.r)
    scipy.misc.imsave('result/observation/movedRow5.jpg', rn5.r)

    # Show it
    # plt.ion()
    # plt.imshow(rn4.r)
    # plt.show()
    # plt.pause(1000)
