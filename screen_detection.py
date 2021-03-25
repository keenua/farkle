from typing import *
from recognition import recognize
from grabscreen import grab_screen
import numpy as np
import cv2

def text(img: np.ndarray, text: str, pos: Tuple[int, int]):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = pos
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(img, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

classifier = cv2.CascadeClassifier('data\\all\\cascade.xml') 

while True:
    img = grab_screen((600, 200, 1500, 900))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)

    found = classifier.detectMultiScale(img, minSize = (75, 75), maxSize = (95, 95)) 

    dice = []

    for (x, y, width, height) in found:
        cv2.rectangle(img, (x, y), (x + height, y + width), (0, 255, 0), 5) 
        die = img[y:y+width,x:x+height]
        dice.append((recognize(die), x, y))
    
    for (value, x, y) in dice:
        text(img, str(value), (x, y))

    cv2.imshow('cv2', img)
    cv2.waitKey(10)