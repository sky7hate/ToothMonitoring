import cv2
import scipy.misc
from matplotlib import pyplot as plt
import numpy as np
from opendr.everything import *
import argparse
import chumpy as ch
from scipy.spatial.transform import Rotation as R

# img = cv2.imread(r'/home/jiaming/MultiviewFitting/data/observation/12681/real_rc1.jpg')
# # plt.imshow(img)
# img2 = scipy.misc.imresize(img, 0.5)
# plt.imshow(img2)
# plt.show()
# index = 0
# for i in range(640):
#     for j in range(480):
#         if img[j, i, 0] > 0:
#             index += 1
# print index
# index = 0
# for i in range(320):
#     for j in range(240):
#         if img2[j, i, 0] > 0:
#             index += 1
# print index
#
# print img.size, img2.size

# a = [[1, 2, 3, 4], [5, 6, 7, 8], [10, 11, 12, 13]]
# m = np.mean(a, axis=0)
# m1 = np.mean(a, axis=1)
# # m2 = np.mean(a, axis=2)
# print m, m1

# a = np.array([1, 2, 3])
# r = np.array([0, 0, np.pi/4])
# print a.dot(Rodrigues(r)).shape, Rodrigues(r).dot(a).shape

# img1_file_path = '/home/jiaming/MultiviewFitting/data/observation/new_seg/gtcontour_0.jpg'
# observed1 = load_image(img1_file_path)
# print observed1.shape, observed1.dtype
#
# contour1 = (scipy.misc.imresize(observed1, 0.25)/255).astype(np.float64)
# print contour1.shape, contour1.dtype
#
# contour2 = cv2.resize(observed1, (observed1.shape[1]/4, observed1.shape[0]/4))
# print contour2.shape, contour2.dtype
#
# # contour3 = scipy.ndimage.interpolation.zoom(observed1, (0.25, 0.25, 1.0))
# # print contour3.shape, contour3.dtype
#
# plt.imshow(contour1)
# plt.show()


def parsing_camera_pose(file_camera):
    file = open(file_camera, 'r')
    lines = file.readlines()
    cc = 0
    cpose = []
    for line in lines:
        if cc == 0:
            tmp = line.split(' ')
            pw = float(tmp[2])
            ph = float(tmp[3])
            pf = float(tmp[0]) / float(tmp[1]) * pw
            cc +=1
            continue
        tmp = line.split(', ')
        # print tmp
        cpose.append([float(tmp[0]), float(tmp[1]), float(tmp[2])])
        # cpose.append([float(x) for x in line.split(',')])
    prt1 = ch.array(cpose[0])
    pt1 = ch.array(cpose[1])
    prt2 = ch.array(cpose[2])
    pt2 = ch.array(cpose[3])
    prt3 = ch.array(cpose[4])
    pt3 = ch.array(cpose[5])
    prt4 = ch.array(cpose[6])
    pt4 = ch.array(cpose[7])
    prt5 = ch.array(cpose[8])
    pt5 = ch.array(cpose[9])
    prt6 = ch.array(cpose[10])
    pt6 = ch.array(cpose[11])
    wf = open('camera_matrix.txt', 'w')
    for i in range(len(cpose)):
        if i%2 == 0:
            tmpr = R.from_rotvec(cpose[i])
            r_mat = tmpr.as_dcm()
            wf.write(str(r_mat) + '\n')
        else:
            wf.write(str(cpose[i]) + '\n')
    return prt1, pt1, prt2, pt2, prt3, pt3, prt4, pt4, prt5, pt5, prt6, pt6, pf, pw, ph

parser = argparse.ArgumentParser()
parser.add_argument("-rt1", type=float, nargs=3, help="view1 camera pose rotation pars")
parser.add_argument("-t1", type=float, nargs=3, help="view1 camera pose translation parts")
parser.add_argument("-c")
args = parser.parse_args()

# print args.rt1, args.t1, args.a
# rt1 = 1
# if args.rt1 is not None:
#     rt1 = ch.array(args.rt1)
# print rt1
print parsing_camera_pose(args.c)
