import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('data/Contour_test/3.jpg',0)
edges = cv2.Canny(img,70,150)
cv2.imwrite('data/Contour_test/contour_3.jpg', edges)
# imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# im = cv2.drawContours(image, contours, -1, (0,255,0), 3)
#
#
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),\
plt.imshow(edges, cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.savefig('data/Contour_test/cc_3.png')