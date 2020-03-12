import cv2
import numpy as np

"""
def get_corners(img, t=128, s=3):
    h, w = img.shape

    corners = np.zeros((4, 2), dtype='uint16')

    for y in range(s-1, h /4):
        for x in range(s-1, w / 4):
            if img[y, x] > t:
                if np.mean(img[y:y+s, x:x+s]) > t and \
                np.mean(img[y-s:y+s, x-s:x]) < t and \
                np.mean(img[y-s:y, x:x+s]) < t:
                    corners[0, 0] = y
                    corners[0, 1] = x

    return corners
"""

INPUTS_PATH = '../inputs/'
INPUT_LAB = 'test_lab_1.jpeg'

LAB_W = 8
LAB_H = 8

THRESHOLD = 128
THRESHOLD_F = THRESHOLD / 255

img = cv2.imread(INPUTS_PATH + INPUT_LAB, 0)
h, w = img.shape
img = cv2.resize(img, (int(w/10), int(h/10)))
h, w = img.shape
s_img = cv2.blur(img,(3,3))

#cv2.imshow('LI', img)
#cv2.imshow('resized smoothed LI', s_img)

#b_img = np.zeros(s_img.shape, dtype='uint8')
r, b_img = cv2.threshold(s_img,THRESHOLD, 255, cv2.THRESH_BINARY)
#b_img = (img > THRESHOLD).astype('uint8')

#cv2.imshow('binary LI', b_img)

e_img = cv2.Canny(b_img, 100, 200)

#cv2.imshow('edges LI', e_img)

corners = cv2.goodFeaturesToTrack(e_img, 4, 0.1, 100)
corners = np.flip(corners, axis=0)

for c in corners:
    x,y = c.ravel()
    cv2.circle(img,(x,y),4,128,-1)

max_p = corners.max(0, initial=0)
min_p = corners.min(0, initial=max(h, w))

new_corners = np.zeros((4, 2), dtype='uint16')

new_w = max_p[0][0] - min_p[0][0]
new_h = max_p[0][1] - min_p[0][0]

new_corners[0] = [0, 0]
new_corners[1] = [new_w, 0]
new_corners[2] = [0, new_h]
new_corners[3] = [new_w, new_h]

M, mask = cv2.findHomography(corners, new_corners)
n_img = cv2.warpPerspective(b_img, M, (new_w, new_h))

cv2.imshow('b IL', b_img)
cv2.imshow('n IL', n_img)

lab = np.zeros((LAB_H, LAB_W), dtype=bool)

c_x = new_w/LAB_W
c_y = new_h/LAB_H

for y in range(LAB_H):
    for x in range(LAB_W):
        lab[y, x] = np.mean(n_img[int(y*c_y):int(y*c_y+c_y), int(x*c_x):int(x*c_x+c_x)]) > THRESHOLD

lab_i = cv2.resize(lab.astype('uint8') * 255, (256, 256), cv2.INTER_CUBIC  )
r, lab_i = cv2.threshold(lab_i,THRESHOLD+10, 255, cv2.THRESH_BINARY)
cv2.imshow("Final labyrinth", lab_i)

cv2.waitKey(0)
cv2.destroyAllWindows()