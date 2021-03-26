from time import sleep
from typing import *

import cv2
from pytesseract import image_to_string

from grabscreen import grab_screen
from imageutils import (ColorModel, add_whitespace_around, remove_whitespace,
                        thresh_by_color)

FILE = 'e:\\Work\\ProjectFiles\\farkle\\train_old2\\test.png'

FROM_SCREEN = False
SAVE_SCREENSHOTS = False
SHOW_IMAGES = False

screenshot = None


def __from_screenshot(p):
    global screenshot
    x1, y1, x2, y2 = p
    if screenshot is None:
        screenshot = cv2.imread(FILE)
    return screenshot[y1:y2, x1:x2].copy()


grab = grab_screen if FROM_SCREEN else __from_screenshot

LABELS = {
    'goal': ((80, 40, 150, 90), [100, 0, 0], [255, 50, 50], ColorModel.RGB, True),
    'opp_total': ((85, 72, 190, 125), [0, 0, 0], [255, 255, 200], ColorModel.HSV, False),
    'opp_round': ((100, 134, 195, 187), [100, 0, 0], [255, 30, 255], ColorModel.RGB, True),
    'opp_selected': ((100, 133 + 69, 195, 186 + 69), [100, 0, 0], [255, 30, 255], ColorModel.RGB, True),
    'hero_total': ((85, 867, 190, 920), [0, 0, 0], [255, 255, 200], ColorModel.HSV, False),
    'hero_round': ((100, 1000-69, 195, 1053-69), [0, 0, 60], [255, 255, 255], ColorModel.HSV, False),
    'hero_selected': ((100, 1000, 195, 1053), [0, 0, 60], [255, 255, 255], ColorModel.HSV, False),
}


class Score:
    def __init__(self, **entries):
        self.goal = 0
        self.opp_total = 0
        self.opp_round = 0
        self.opp_selected = 0
        self.hero_total = 0
        self.hero_round = 0
        self.hero_selected = 0
        self.__dict__.update(entries)


def __text_to_int(text: str, default: int = 0) -> int:
    try:
        if text is None:
            return default
        return int(text)
    except ValueError:
        return default


def recognize_score() -> Score:
    scores = dict()
    for n, (p, l, u, cm, inv) in LABELS.items():
        img = grab(p)
        if SAVE_SCREENSHOTS:
            cv2.imwrite(f'{n}.png', img)

        img = thresh_by_color(img, l, u, cm, inv)
        img = remove_whitespace(img)
        img = add_whitespace_around(img, 10, 0)

        text = image_to_string(
            img, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789')
        scores[n] = text

        if SHOW_IMAGES:
            cv2.imshow('window', img)
            cv2.waitKey(5000)

    score = Score(**{k: __text_to_int(v) for k, v in scores.items()})
    return score


if __name__ == '__main__':
    while True:
        score = recognize_score()
        print(score.__dict__)
