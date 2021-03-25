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

def detect(img):
    result = []

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([21, 6, 0], np.uint8)
    upper_gray = np.array([124, 71, 255], np.uint8)
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

    blur = cv2.bitwise_and(img, img, mask = mask_gray)
    blur = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV)[1]

    contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]

    imgcont = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    for c in contours:
        area = cv2.contourArea(c)
        cv2.drawContours(imgcont, [c], 0, (0,255,0), 1)

        if 1800 < area < 5000:
            res = fix_rotation(img, c)
            rect = cv2.boundingRect(c)
            result.append((rect, res))

    return result

if __name__ == '__main__':
    SCREENSHOT_DIR = 'e:\\Games\\Steam\\Screenshots'
    OY,OX = (200, 600)
    H,W = (600, 900)

    img = cv2.imread(f'{SCREENSHOT_DIR}/379430_20210322222253_1.png')
    img = img[OY:OY+H, OX:OX+W]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    for (x,y,w,h), die in detect(img):
        res = recognize(die)
        print(res)

        plt.subplot(1, 1, 1) 
        plt.imshow(img_rgb[y:y+h,x:x+w]) 
        plt.show()


