from detection2 import detect
import time
from typing import *

import cv2
import numpy as np

from getkeys import key_check
from grabscreen import grab_screen
from recognition import recognize
from uuid import uuid4
from os import path

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

paused_status = False

save_to = 'e:\\Work\\Projects\\farkle\\train\\under\\'
save_bg_to = 'e:\\Work\\Projects\\farkle\\train\\under\\bg'

dice = []
bg = None

while True:
    pressed = key_check()

    if 'Y' in pressed: 
        paused_status = not(paused_status)

    if 'E' in pressed or 'Q' in pressed or 'F' in pressed:
        for _, _, _, die in dice:
            file = path.join(save_to, f'{uuid4()}.png')
            cv2.imwrite(file, die)

        print(f'{len(dice)} images saved!')
        continue

    if bg is not None and 'B' in pressed:
        file = path.join(save_bg_to, f'{uuid4()}.png')
        cv2.imwrite(file, bg)
        print(f'background image saved!')

    if paused_status: 
        print("Paused... (y)")
        time.sleep(1)
        continue
    
    img = grab_screen((600, 200, 1499, 799))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)

    dice = []
    bg = img.copy()
    for (x, y, width, height) in detect(img):
        die = img[y:y+height,x:x+width].copy()
        value = recognize(die)

        cv2.rectangle(img, (x, y), (x + height, y + width), (0, 255, 0), 5) 
        if value != 0:
            bg[y:y+width, x:x+height] = np.zeros((width, height, 4))
            dice.append((recognize(die), x, y, die))
    
    for (value, x, y, _) in dice:
        text(img, str(value), (x, y))

    cv2.imshow('cv2', img)
    cv2.waitKey(1)
