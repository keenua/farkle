from imageutils import *
from recognition import recognize
import cv2
from matplotlib import pyplot as plt
import numpy as np

def fix_rotation(img, cnt):
    rect = cv2.minAreaRect(cnt)

    box = cv2.boxPoints(rect)
    box = np.int0(box)

    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped

def detect_dice(img):
    result = []

    contours = detect_contours(img, GRAY)

    for contour, area, rect in contours:
        if 1800 < area < 5000:
            res = fix_rotation(img, contour)
            result.append((rect, res))

    return result

def detect_hold_markers(img):
    return [center(rect) for _, area, rect in detect_contours(img, ORANGE) if 300 < area < 5000]

def detect_selection_marker(img):
    return [center(rect) for _, area, rect in detect_contours(img, YELLOW) if 300 < area < 5000]

if __name__ == '__main__':
    SCREENSHOT_DIR = 'e:\\Games\\Steam\\Screenshots'
    OY,OX = (200, 600)
    H,W = (600, 900)

    img = cv2.imread(f'{SCREENSHOT_DIR}/379430_20210322222253_1.png')
    img = img[OY:OY+H, OX:OX+W]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    for (x,y,w,h), die in detect_dice(img):
        res = recognize(die)
        print(res)

        plt.subplot(1, 1, 1) 
        plt.imshow(img_rgb[y:y+h,x:x+w]) 
        plt.show()


