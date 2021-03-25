from recognition import recognize
import cv2
from matplotlib import pyplot as plt
import numpy as np

colors = []

for c in range(85, 145, 2):
    colors.append((c, c + 21))

for c in range(40, 55, 2):
    colors.append((c, c + 12))

def detect(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype= 'uint8')

    for l, u in colors:
        lower = np.array([l] * 3, dtype = 'uint8')
        upper = np.array([u] * 3, dtype = 'uint8')

        c_mask = cv2.inRange(img, lower, upper)
        mask = cv2.bitwise_or(mask, c_mask)

    blur = cv2.bitwise_and(img, img, mask = mask)
    blur = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV)

    _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    imgcont = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    mask = np.zeros_like(imgcont)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)

    result = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < 5000 and area > 1800:
            x,y,w,h = cv2.boundingRect(c)
            result.append((x,y,w,h))

    return result

if __name__ == '__main__':
    SCREENSHOT_DIR = 'e:\\Games\\Steam\\Screenshots'
    OY,OX = (200, 600)
    H,W = (600, 900)

    img = cv2.imread(f'{SCREENSHOT_DIR}/379430_20210322222253_1.png')
    img = img[OY:OY+H, OX:OX+W]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    for x,y,w,h in detect(img):
        die = img[y:y+h,x:x+w]
        res = recognize(die)
        print(res)

        plt.subplot(1, 1, 1) 
        plt.imshow(img_rgb[y:y+h,x:x+w]) 
        plt.show()


