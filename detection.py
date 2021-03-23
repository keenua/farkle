from recognition import recognize
import cv2
from matplotlib import pyplot as plt
import numpy as np

SCREENSHOT_DIR = 'e:\\Games\\Steam\\Screenshots'
OY,OX = (300, 600)
H,W = (600, 900)

img = cv2.imread(f'{SCREENSHOT_DIR}/379430_20210322222336_1.png')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

main_img = img
  
classifier = cv2.CascadeClassifier('data\\all\\cascade.xml') 
found = classifier.detectMultiScale(main_img[OY:OY+H, OX:OX+W], minSize = (60, 60), maxSize = (100, 100)) 

amount_found = len(found) 
  
if amount_found != 0: 
    for (x, y, width, height) in found: 
        x1,y1,x2,y2=(x + OX, y + OY, x + OX + height, y + OY + width)
        die = img[y1:y2, x1:x2]

        res = recognize(die)
        print(res)

        plt.subplot(1, 1, 1) 
        plt.imshow(img_rgb[y1:y2, x1:x2]) 
        plt.show()


          

