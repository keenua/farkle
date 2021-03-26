from os import listdir, path
from typing import *

import cv2
from pytesseract import image_to_string

from grabscreen import grab_screen
from imageutils import *

PATH = 'e:\\Work\\ProjectFiles\\farkle\\train_old2\\test2.png'
# PATH = 'e:\\Work\\ProjectFiles\\farkle\\screenshots\\'

SAVE_SCREENSHOTS = False
SHOW_IMAGES = False

LABELS = {
    'goal': ((80, 40, 150, 90), RED),
    'opp_total': ((85, 76, 190, 115), YELLOW),
    'opp_round': ((85, 134, 195, 187), BRED),
    'opp_selected': ((85, 202, 195, 255), RED),
    'hero_total': ((85, 867, 190, 920), YELLOW),
    'hero_round': ((85, 931, 195, 984), BLACK),
    'hero_selected': ((85, 1000, 195, 1053), BLACK),
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


def recognize_score(screenshot: str = None) -> Score:
    scores = dict()
    for n, (p, range) in LABELS.items():
        img = grab_screen(p, screenshot)
        if SAVE_SCREENSHOTS:
            cv2.imwrite(f'{n}.png', img)

        img = thresh_by_color(img, range)
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
        if path.isdir(PATH):
            for f in listdir(PATH):
                p = path.join(PATH, f)
                score = recognize_score(p)

                print(score.__dict__)

                img = cv2.imread(p)
                cv2.imshow('window', cv2.resize(img, (800, 600)))
                cv2.waitKey(0)
        else:
            score = recognize_score(PATH)
            print(score.__dict__)

        if PATH is not None and path.isfile(PATH):
            break
