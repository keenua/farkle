import time
from os import mkdir, path
from typing import *
from uuid import uuid4

import cv2
import numpy as np
from farkle.utils import grab_screen, new_keys
from farkle.visual.detection import detect_dice
from farkle.visual.recognition import recognize


def text(img: np.ndarray, text: str, pos: Tuple[int, int]):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = pos
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)


def collect_samples(save_to_dir: str = 'samples'):
    paused_status = False

    dice = []
    all = []

    while True:
        pressed, all = new_keys(all)

        if 'Y' in pressed:
            paused_status = not(paused_status)

        if 'E' in pressed or 'Q' in pressed or 'F' in pressed:
            for value, _, _, die in dice:
                v_dir = path.join(save_to_dir, str(value))
                if not path.isdir(v_dir):
                    mkdir(v_dir)
                file = path.join(v_dir, f'{uuid4()}.png')
                cv2.imwrite(file, die)

            print(f'{len(dice)} images saved!')
            continue

        if paused_status:
            print("Paused... (y)")
            time.sleep(1)
            continue

        img = grab_screen((600, 200, 1500, 900))

        dice = []
        for (x, y, width, height), die in detect_dice(img):
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 5)

            value = recognize(die)
            dice.append((recognize(die), x, y, die))

        for (value, x, y, _) in dice:
            if value != 0:
                text(img, str(value), (x, y))

        cv2.imshow('cv2', img)
        cv2.waitKey(1)
