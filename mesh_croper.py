import utils
import os
import numpy as np
import glob
from open3d import *
import seaborn as sns
import copy
# import interative_teeth_labeling

# palette = np.array(sns.color_palette("RdBu_r", 17))  #coolwarm Paired cubehelix RdBu_r husl bright
palette = np.array(sns.color_palette("bright", 17))  #coolwarm Paired cubehelix RdBu_r husl bright
palette[0] = 0.7

need_modify = 0

def toggle_modify():
    global need_modify
    need_modify = not need_modify
    print 'need_modify:', need_modify

def draw(pcds):
    # vis = Visualizer()
    # vis.create_window(window_name=u'Interactive Tooth Labeling.', width=1200, height=800)
    # for pcd in pcds:
    #     vis.add_geometry(pcd)
    # vis.run()

    def toggle_modify_callback(vis):
        toggle_modify()
        return False

    key_to_callback = {}
    key_to_callback[ord("M")] = toggle_modify_callback
    draw_geometries_with_key_callbacks(pcds, key_to_callback, window_name=u'Model vClass Viewer.', width=1200, height=800)


def crop(mesh_file, label_dir=None, save_dir=None):
    trimeshes = TriangleMesh()
    models = PointCloud()
    models_labels = []
    for mesh in mesh_file:
        try:
            if label_dir==None:
                label_file = os.path.join(os.path.dirname(mesh),
                                          os.path.basename(mesh).split('.')[0] + '_vClassLabels.txt')
            else:
                # label_file = os.path.join(label_dir, os.path.basename(mesh).split('.')[0] + '_vClassLabels.txt')
                label_file = os.path.join(label_dir, part + '_vClassLabels.txt')
            print mesh, label_file
            _, model_label, pcd = utils.read_off(mesh, label_file, palette=palette)
            models_labels.append(model_label)
            # utils.update_pcd_color(pcd, label_file)

            trimesh = utils.read_mesh(mesh)
        except:
            print 'Error.'
            continue

        models = models + pcd
        trimeshes = trimeshes+trimesh
    if len(models_labels) > 0:
        models_labels = np.hstack(models_labels)
        print 'save to:', os.path.join(save_dir, os.path.basename(os.path.dirname(os.path.dirname(mesh))),os.path.basename(os.path.dirname(mesh)))
        # utils.crop_mesh_with_labels_fast_2(trimeshes, models_labels, os.path.join(save_dir, os.path.basename(os.path.dirname(mesh))))
        utils.crop_mesh_with_labels_fast_2(trimeshes, models_labels, os.path.join(save_dir, os.path.basename(os.path.dirname(os.path.dirname(mesh))),os.path.basename(os.path.dirname(mesh))))

    # draw([models] + [trimeshes])

if __name__ == '__main__':
    # root_dir = 'E:\HKU_Project\Dataset\ModelsForGordon-rearranged-cleaned'
    # label_dir = 'E:\HKU_Project\Dataset\TeethNotation'
    # save_dir = 'E:\HKU_Project\Dataset\ModelsForGordon-rearranged-cleaned-crownNbed-binaryEncoded'

    # root_dir = 'E:\HKU_Project\Dataset\ModelsForGordon-rearranged-cleaned'
    # label_dir = 'E:\HKU_Project\Dataset\ModelsForGordon-rearranged-cleaned_lowerJaw'
    # save_dir = 'E:\HKU_Project\Dataset\ModelsForGordon_LowerJaw-crownNbed'

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
