import cv2
from matplotlib import pyplot as plt
import numpy as np
from os import path, listdir

params = cv2.SimpleBlobDetector_Params()

# Set Area filtering parameters
params.filterByArea = True
params.minArea = 30

# Set Circularity filtering parameters
params.filterByCircularity = True 
params.minCircularity = 0.2

# Set Convexity filtering parameters
params.filterByConvexity = True
params.minConvexity = 0.1
    
# Set inertia filtering parameters
params.filterByInertia = True
params.minInertiaRatio = 0.05

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

lower_gray = np.array([50,0,0], np.uint8)
upper_gray = np.array([255,255,75], np.uint8)

def detect(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

    blur = cv2.bitwise_and(img, img, mask = mask_gray)
    blur = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV)[1]

    keypoints = detector.detect(thresh)

    return [(int(k.pt[0]), int(k.pt[1])) for k in keypoints]

if __name__ == '__main__':
    SCREENSHOT_DIR = 'e:\\Work\\Projects\\farkle\\train\\6\\'

    for f in listdir(SCREENSHOT_DIR):
        fp = path.join(SCREENSHOT_DIR, f)
        print(fp)
        img = cv2.imread(fp)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        for c,r in detect(img):
            img_rgb = cv2.circle(img_rgb, (c,r), 2, color=(0, 255, 0, 0), thickness=1)

        plt.subplot(1, 1, 1) 
        plt.imshow(img_rgb) 
        plt.show()

