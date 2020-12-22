import numpy as np
from open3d import *
from glob import glob
import os
import seaborn as sns
import copy
import vtk

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
    elif extension=='.off':
        return OFFReader(file_name)
    else:
        # Return a None if the extension is unknown.
        print 'Warning: Unsupport file extension.'
        poly_data = None

    verts = []
    faces = []
    for i in range(poly_data.GetNumberOfPoints()):
        verts.append(np.array(poly_data.GetPoint(i), dtype=np.float))
    verts = np.vstack(verts)

    triangles = poly_data.GetPolys().GetData()
    for i in range(poly_data.GetNumberOfCells()):
        faces.append(np.array([int(triangles.GetValue(j)) for j in range(4 * i + 1, 4 * i + 4)]))
    faces = np.vstack(faces)
    return verts, faces

def OFFReader(file_name):
    file = open(file_name, 'r')
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    verts = np.vstack(verts)
    faces = np.vstack(faces)
    return verts, faces

def read_mesh(file_name):
    '''
    Read a 3d model file,
    output an Open3D TriangleMesh object.
    '''
    v, f = ReadPolyData(file_name)
    trimesh = TriangleMesh()
    trimesh.vertices = Vector3dVector(v)
    trimesh.triangles = Vector3iVector(f)
    trimesh.compute_vertex_normals()
    trimesh.normalize_normals()
    return trimesh

def read_labels(labelpath):
    labelfile = open(labelpath, 'r')
    labels_stream = labelfile.readlines()
    n = -1
    for i, data in enumerate(labels_stream):
        if i == 0:
            line_data = data.replace('\n', '').split(' ')
            if len(line_data) >= 2:
                n = int(line_data[1])
            elif len(line_data) == 1:
                n = int(line_data[0])
            labels = np.zeros((n,))
        else:
            label = int(data)
            labels[i - 1] = label
    return n, labels

def read_off(filepath, labelpath=None, palette=None):
    '''
    Read OFF file with vertex labels.
    Output:
    n_vertices,
    labels,
    pcd
    '''
    file = open(filepath, 'r')
    data = file.readlines()
    n_vertices = 0
    vertices = None
    pcd = None
    labels = None
    if data[0] != 'OFF\n':
        print 'File type mismatch.'
        return None
    else:
        n_vertices = int(data[1].replace('\n', '').split(' ')[0])
        # print 'Num of vertices: ', n_vertices

        pcd = PointCloud()
        vertices = np.zeros((n_vertices, 3))
        for i in range(2, n_vertices):
            vertices[i - 2] = np.array(data[i].replace('\n', '').split(' '))[0:3].astype(float)
        pcd.points = Vector3dVector(vertices)

        if labelpath is not None:
            n_labels, labels = read_labels(labelpath)
            assert n_labels == n_vertices, 'Vertices num mismatch: {}, {}'.format(n_labels, n_vertices)
            if palette is not None:
                update_pcd_color(pcd, labels, palette)
    return n_vertices, labels, pcd


def read_template(data_dir, part=None, palette=None):
    tooth_files = glob(os.path.join(data_dir, '*.off'))
    tooth_files.sort()
    if part == 'UpperJaw':
        tooth_files = tooth_files[0:16]
    elif part == 'LowerJaw':
        tooth_files = tooth_files[16:32]

    file_names = [os.path.basename(x).split('.')[0] for x in tooth_files]
    print file_names
    print len(file_names)

    # pcd_list = []
    n_vertices_list = []
    labels_list = []
    combine = PointCloud()
    for i, path in enumerate(tooth_files):
        # print path
        n_vertices, _, pcd = read_off(path)

        # if palette is not None:
        #     n_labels, lables = read_labels(os.path.join(data_dir, part + '_vClassLabels.txt')) #data/template/individual/UpperJaw_vClassLabels.txt
        #     colors = np.zeros((n_vertices, 3))
        #     colors[:] = palette[i%len(palette)]
        #     pcd.colors = Vector3dVector(copy.deepcopy(colors))

        combine += pcd
        # label = np.ndarray(n_vertices).astype(int)
        # label[:] = int(os.path.basename(path).split('.')[0])
        n_vertices_list.append(n_vertices)
        # labels_list.append(label)
    # draw_geometries([combine])
    # labels = np.hstack(labels_list)

    if palette is not None:
        n_labels, labels = read_labels(
            os.path.join(data_dir, part + '_vClassLabels.txt'))  # data/template/individual/UpperJaw_vClassLabels.txt
            # os.path.join(data_dir, '*_vClassLabels.txt'))
        update_pcd_color(combine, labels, palette)

    # if part=='UpperJaw':
    #     cos_theta = np.cos(np.deg2rad(180))
    #     sin_theta = np.sin(np.deg2rad(180))
    #     trans_params = np.array([[cos_theta, -sin_theta, 0.0, 0.0],
    #                              [sin_theta, cos_theta, 0.0, 0.0],
    #                              [0.0, 0.0, 1.0, 0.0],
    #                              [0.0, 0.0, 0.0, 1.0]]
    #                             )
    #     combine.transform(trans_params)

    # label_fo = open('data/template/individual/UpperJaw_vClassLabels.txt', 'w')
    # label_fo.write('%d\n' % len(labels))
    # for i in range(len(labels)):
    #     label_fo.write('%d\n' % labels[i])
    # label_fo.close()

    return len(labels), labels, combine

def update_pcd_color(pcd, labels, palette):
    colors = np.zeros((len(labels), 3))
    npal = len(palette)
    classes = np.unique(labels)
    nclasses = len(classes)
    for i in range(nclasses):
        colors[np.where(labels == classes[i])] = palette[i % npal]
    pcd.colors = Vector3dVector(colors)
    return pcd

def create_pointcloud(V, colors=None):
    pcd = PointCloud()
    pcd.points = Vector3dVector(V)
    if colors is not None:
        pcd.colors = Vector3dVector(colors)
    return pcd

def create_trianglemesh(V, F, colors=None, compute_normals=True):
    mesh = TriangleMesh()
    mesh.triangles = Vector3iVector(F)
    mesh.vertices = Vector3dVector(V)
    if colors is not None:
        mesh.vertex_colors = Vector3dVector(colors)
    if compute_normals:
        mesh.compute_vertex_normals()
        mesh.normalize_normals()
    return mesh

def crop_mesh_with_labels_fast(mesh, labels, save_dir):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    classes = np.unique(labels).astype(np.int)
    iidx = {}
    ivertices = {}
    itriangles = {}
    for iclass in classes:
        # itooth = models[np.where(models_labels == classes[i])]
        if iclass < 0:
            continue
        iidx[iclass] = np.where(labels == iclass)[0]
        ivertices[iclass] = vertices[iidx[iclass], :]
        itriangles[iclass] = []

    for face in triangles:
        fvlabels = labels[face]
        ulabels = np.unique(fvlabels)
        if len(ulabels)== 1 and ulabels[0] > 0:
            label = ulabels[0]
            newface = np.zeros(3)
            newface[0] = np.where(iidx[label] == face[0])[0]
            newface[1] = np.where(iidx[label] == face[1])[0]
            newface[2] = np.where(iidx[label] == face[2])[0]
            itriangles[label].append(newface)

    for iclass in classes:
        if iclass < 0 or len(itriangles[iclass])==0:
            continue
        itriangles[iclass] = np.vstack(itriangles[iclass])
        print ivertices[iclass].shape, itriangles[iclass].shape
        imesh = create_trianglemesh(ivertices[iclass], itriangles[iclass], True)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        print 'save to', os.path.join(save_dir, '%d.ply' % iclass)
        write_triangle_mesh(os.path.join(save_dir, '%d.ply' % iclass), imesh) #write_ascii=True

# also crop out the teeth bed
def crop_mesh_with_labels_fast_2(mesh, labels, save_dir):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    classes = np.unique(labels).astype(np.int)
    iidx = {}
    ivertices = {}
    itriangles = {}
    for iclass in classes:
        # if iclass < 0:
        #     continue
        iidx[iclass] = np.where(labels == iclass)[0]
        ivertices[iclass] = vertices[iidx[iclass], :]
        itriangles[iclass] = []

    for face in triangles:
        fvlabels = labels[face]
        ulabels = np.unique(fvlabels)
        # if len(ulabels)== 1 and ulabels[0] > 0:
        if len(ulabels)== 1:
            label = ulabels[0]
            newface = np.zeros(3)
            newface[0] = np.where(iidx[label] == face[0])[0]
            newface[1] = np.where(iidx[label] == face[1])[0]
            newface[2] = np.where(iidx[label] == face[2])[0]
            itriangles[label].append(newface)

    for iclass in classes:
        # if iclass < 0 or len(itriangles[iclass])==0:
        if len(itriangles[iclass])==0:
            continue
        itriangles[iclass] = np.vstack(itriangles[iclass])
        print ivertices[iclass].shape, itriangles[iclass].shape
        imesh = create_trianglemesh(ivertices[iclass], itriangles[iclass], compute_normals=True)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        print 'save to', os.path.join(save_dir, '%d.ply' % iclass)
        write_triangle_mesh(os.path.join(save_dir, '%d.ply' % iclass), imesh) #write_ascii=True

if __name__ == '__main__':
    path = 'data/template/individual'
    palette = np.array(sns.color_palette('coolwarm', 16))
    read_template(path, part='UpperJaw', palette=palette)

