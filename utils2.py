import numpy as np
# from open3d import *
import open3d as o3d
from glob import glob
import os
#import seaborn as sns
import copy
import vtk
from vtk.util import numpy_support

def ReadPolyData(file_name):
    '''
    @ Guodong Wei.
    '''
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
        print ('Warning: Unsupport file extension.')
        poly_data = None

    verts = []
    faces = []
    # print poly_data.GetPointData() #GetPointData().GetArray("Colors")
    for i in range(poly_data.GetNumberOfPoints()):
        verts.append(np.array(poly_data.GetPoint(i), dtype=np.float))
    verts = np.vstack(verts)

    triangles = poly_data.GetPolys().GetData()
    for i in range(poly_data.GetNumberOfCells()):
        faces.append(np.array([int(triangles.GetValue(j)) for j in range(4 * i + 1, 4 * i + 4)]))
    if len(faces)>0:
        faces = np.vstack(faces)
    else:
        faces=None
    return verts, faces

def createVtkPolyData(verts, tris):
    """Create and return a vtkPolyData.

    verts is a (N, 3) numpy array of float vertices

    tris is a (N, 1) numpy array of int64 representing the triangles
    (cells) we create from the verts above.  The array contains
    groups of 4 integers of the form: 3 A B C
    Where 3 is the number of points in the cell and A B C are indexes
    into the verts array.
    """

    # save, we share memory with the numpy arrays
    # so they can't get deleted

    poly = vtk.vtkPolyData()

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(verts))
    poly.SetPoints(points)

    cells = vtk.vtkCellArray()
    cells.SetCells(len(tris) / 4, numpy_support.numpy_to_vtkIdTypeArray(tris))
    poly.SetPolys(cells)

    return poly

def OFFReader(file_name):
    file = open(file_name, 'r')
    # print file.readline().strip()
    header = file.readline().strip()
    if header not in ['OFF', 'COFF']:
        print  ('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')[0:3]] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:4] for i_face in range(n_faces)]
    verts = np.vstack(verts)
    faces = np.vstack(faces)
    # print verts[0]
    return verts, faces

def read_mesh(file_name):
    '''
    Read a 3d model file,
    output an Open3D TriangleMesh object.
    '''
    v, f = ReadPolyData(file_name)
    trimesh = o3d.geometry.TriangleMesh()
    trimesh.vertices = o3d.utility.Vector3dVector(v)
    trimesh.triangles = o3d.utility.Vector3iVector(f)
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
        print ('File type mismatch.')
        return None
    else:
        n_vertices = int(data[1].replace('\n', '').split(' ')[0])
        print ('Num of vertices: ', n_vertices)

        pcd = o3d.geometry.PointCloud()
        vertices = np.zeros((n_vertices, 3))
        for i in range(2, n_vertices):
            vertices[i - 2] = np.array(data[i].replace('\n', '').split(' '))[0:3].astype(float)
        pcd.points = o3d.utility.Vector3dVector(vertices)

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
    print (file_names)
    print (len(file_names))

    # pcd_list = []
    n_vertices_list = []
    labels_list = []
    combine = o3d.geometry.PointCloud()
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
        update_pcd_color(combine, labels, palette)

    if part=='UpperJaw':
        cos_theta = np.cos(np.deg2rad(180))
        sin_theta = np.sin(np.deg2rad(180))
        trans_params = np.array([[cos_theta, -sin_theta, 0.0, 0.0],
                                 [sin_theta, cos_theta, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]]
                                )
        combine.transform(trans_params)

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
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def create_pointcloud(V, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(V)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def create_trianglemesh(V, F, colors=None, compute_normals=True):
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(F)
    mesh.vertices = o3d.utility.Vector3dVector(V)
    if colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    if compute_normals:
        mesh.compute_vertex_normals()
        mesh.normalize_normals()
    return mesh

def draw_geometries(geo_list, win_width=1200, win_height=800, rand_color=False):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=win_width, height=win_height)
    for mesh in geo_list:
        if rand_color:
            mesh.paint_uniform_color(np.random.rand(3))
        vis.add_geometry(mesh)
    vis.get_render_option().load_from_json('renderoption.json')
    vis.run()

def detecteEdge(F):
    '''
    Find vertex indices on mesh boundary.
    '''
    fk1 = F[:, 0]
    fk2 = F[:, 1]
    fk3 = F[:, 2]

    # sort the vertex indices of each edge so that those edge with same vertex can be found
    ed1 = np.sort(np.vstack([fk1, fk2]).transpose(), axis=1)
    ed2 = np.sort(np.vstack([fk1, fk3]).transpose(), axis=1)
    ed3 = np.sort(np.vstack([fk2, fk3]).transpose(), axis=1)

    # single edges
    ed = np.vstack([ed1, ed2, ed3])
    # print ed.shape, ed1.shape, fk1.shape
    esingle, unique_indices, unique_inverse, unique_counts = np.unique(ed, axis=0, return_index=True, return_inverse=True, return_counts=True)
    C = esingle[np.where(unique_counts == 1)]
    Indces_edges = np.reshape(C, (len(C)*2, 1))
    return Indces_edges


def generate4x4TransformationMatrix(r=np.eye(3), t=np.zeros(3)):
    transformation = np.zeros((4, 4))
    transformation[0:3, 0:3] = r
    transformation[0:3, 3] = t
    transformation[3, 3] = 1
    return transformation

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
        print (ivertices[iclass].shape, itriangles[iclass].shape)
        imesh = create_trianglemesh(ivertices[iclass], itriangles[iclass], compute_normals=True)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        print ('save to', os.path.join(save_dir, '%d.ply' % iclass))
        o3d.io.write_triangle_mesh(os.path.join(save_dir, '%d.ply' % iclass), imesh) #write_ascii=True

if __name__ == '__main__':
    import seaborn as sns
    path = 'data/template/individual'
    palette = np.array(sns.color_palette('coolwarm', 16))
    read_template(path, part='UpperJaw', palette=palette)

