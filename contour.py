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

    colors = []

    for c in range(85, 145, 2):
        colors.append((c, c + 21))

    for c in range(40, 55, 2):
        colors.append((c, c + 12))

    mask = np.zeros((H, W), dtype= 'uint8')

    for l, u in colors:
        lower = np.array([l] * 3, dtype = 'uint8')
        upper = np.array([u] * 3, dtype = 'uint8')

        c_mask = cv2.inRange(img, lower, upper)
        mask = cv2.bitwise_or(mask, c_mask)

    blur = cv2.bitwise_and(img, img, mask = mask)
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
            x,y,w,h = cv2.boundingRect(c)
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
