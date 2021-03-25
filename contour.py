import cv2
from matplotlib import pyplot as plt
import numpy as np
from os import listdir, path

SCREENSHOT_DIR = 'e:\\Work\\ProjectFiles\\farkle\\screenshots\\'

for f in listdir(SCREENSHOT_DIR):
    OY,OX = (200, 600)
    H,W = (600, 900)

    p = path.join(SCREENSHOT_DIR, f)
    img = cv2.imread(p)
    img = img[OY:OY+H, OX:OX+W]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([21, 6, 0], np.uint8)
    upper_gray = np.array([124, 71, 255], np.uint8)
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    img_res = cv2.bitwise_and(img, img, mask = mask_gray)

    blur = cv2.bitwise_and(img, img, mask = mask_gray)
    blur = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    flag, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV)

    #Find contours
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True) 

    imgcont = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    out = imgcont.copy()

    mask = np.zeros_like(imgcont)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)

    for c in contours:
        area = cv2.contourArea(c)
        cv2.drawContours(imgcont, [c], 0, (0,255,0), 1)

        if area < 5000 and area > 1800:
            rect = cv2.boundingRect(c)
            x,y,w,h = rect
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.drawContours(mask, [c], 0, 255, -1)

    out = cv2.bitwise_and(out, out, mask=mask)
    fig = plt.figure(figsize=(2, 2))

    fig.add_subplot(2, 2, 1) 
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
    fig.add_subplot(2, 2, 2) 
    plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGBA))
    fig.add_subplot(2, 2, 3) 
    plt.imshow(imgcont)
    fig.add_subplot(2, 2, 4) 
    plt.imshow(out)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
