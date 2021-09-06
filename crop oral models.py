import open3d as o3d
import utils2 as utils
import os, glob
import numpy as np


def crop(mesh_file, label_dir=None, save_dir=None):
    trimeshes = o3d.geometry.TriangleMesh()
    models = o3d.geometry.PointCloud()
    models_labels = []

    try:
        if label_dir==None: # in the same folder with the mesh file
            label_file = os.path.join(os.path.dirname(mesh_file),
                                      os.path.basename(os.path.dirname(mesh_file)) + '_vClassLabels.txt')
                                      # os.path.basename(mesh_file).split('.')[0] + '_vClassLabels.txt')
                                      # os.path.basename(mesh_file).split('.')[0] + '_seg.adjusted')
                                      # 'seg.adjusted2')
        else:
            label_file = os.path.join(label_dir, os.path.basename(os.path.dirname(mesh_file)) + '_vClassLabels.txt')
            # label_file = os.path.join(label_dir, os.path.basename(mesh_file).split('.')[0] + '_vClassLabels.txt')
            # label_file = os.path.join(label_dir, os.path.basename(mesh_file).split('.')[0] + '_seg.adjusted')
            # label_file = os.path.join(label_dir, 'seg.adjusted2')
        print (mesh_file, label_file)
        _, model_label, pcd = utils.read_off(mesh_file, label_file, palette=None)
        print(model_label.shape)
        models_labels.append(model_label)
        # utils.update_pcd_color(pcd, label_file)

        trimesh = utils.read_mesh(mesh_file)
    except:
        print ('Error.')

    models = models + pcd
    trimeshes = trimeshes+trimesh
    if len(models_labels) > 0:
        models_labels = np.hstack(models_labels)
        # adjust the tooth IDs' order
        temp = models_labels.copy()
        midx = (temp>30) & (temp<39)
        models_labels[midx] += 10 
        midx = (temp>40) & (temp<49)
        models_labels[midx] -= 10 
        utils.crop_mesh_with_labels_fast_2(trimeshes, models_labels, save_dir)#os.path.join(save_dir, os.path.basename(os.path.dirname(mesh_file))))

if __name__ == '__main__':
    root_dir = r'G:\10Sets_results'
    label_dir = r'G:\10Sets_results_TID\LowerJaw'
    # save_dir = 'F:\TeethDataset\Gordon_Crown_U'
    save_dir = r'G:\10Sets_results_U'

    file_list = os.listdir(root_dir)
    file_list.sort()
    file_list = np.sort(file_list)

    part = 'LowerJaw' #'L'# 'U' #''UpperJaw' #'LowerJaw'

    for i, file in enumerate(file_list):
        print '{}/{}'.format(i, len(file_list))
        # meshes_files = glob.glob(os.path.join(root_dir, file, '*'+ part + '*.off'))
        meshes_files = glob.glob(os.path.join(root_dir,file,part,  '*.off'))
        model_label_dir = None
        if label_dir is not None:
            model_label_dir = os.path.join(label_dir, file)
            print 'model_label_dir:', model_label_dir
            # model_label_dir = os.path.join(label_dir, file)
        crop(meshes_files, label_dir=model_label_dir, save_dir=save_dir)