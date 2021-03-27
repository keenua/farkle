import numpy as np
from imageutils import center
from os import path, listdir
from recognition import recognize
import cv2
from detection import detect_dice, detect_hold_markers, detect_selection_marker
from grabscreen import grab_screen
from typing import *

from score import Score, recognize_score

DICE_REGION = (600, 200, 1500, 900)
TURN_REGION = (1732, 800, 1733, 801)
DEBUG = False

PATH = None# 'e:\\Work\\ProjectFiles\\farkle\\train_old2\\test2.png'


class Die:
    def __init__(self, value: int, rect: Tuple[int, int, int, int]):
        self.value: int = value
        self.rect = rect
        self.center = center(rect)
        self.selected = False
        self.held = False


class State:
    def __init__(self):
        self.score: Score = None
        self.is_hero_turn = False
        self.dice: List[Die] = []


def __closest_die(dice, point):
    p = np.array(point)
    ordered = sorted(dice, key=lambda die: np.linalg.norm(die.center - p))
    return ordered[0]


def recognize_dice(screenshot: str = None) -> List[Die]:
    dice = []

    img = grab_screen(DICE_REGION, screenshot)

    if DEBUG:
        to_show = img.copy()

    for rect, die in detect_dice(img):
        (x, y, width, height) = rect
        value = recognize(die)

        if value > 0:
            die = Die(value, rect)
            dice.append(die)

        if DEBUG:
            cv2.rectangle(to_show, (x, y), (x + width,
                          y + height), (0, 255, 0), 5)

    if not dice:
        return []

    for (x, y) in detect_hold_markers(img):
        closest_die = __closest_die(dice, (x, y))
        closest_die.held = True

        if DEBUG:
            cv2.rectangle(to_show, (x, y), (x + 5, y + 5), (255, 0, 0), 5)

    for (x, y) in detect_selection_marker(img):
        closest_die = __closest_die(dice, (x, y))
        closest_die.selected = True

        if DEBUG:
            cv2.rectangle(to_show, (x, y), (x + 5, y + 5), (0, 0, 255), 5)

    if DEBUG:
        cv2.imshow('window', to_show)

    return dice


def recognize_turn(screenshot: str = None) -> bool:
    img = grab_screen(TURN_REGION, screenshot)
    return (img == 255).all()


def recognize_state(screenshot: str = None) -> State:
    state = State()

    if not recognize_turn(screenshot):
        return state

    state.is_hero_turn = True
    state.dice = recognize_dice(screenshot)
    state.score = recognize_score(screenshot)

    return state


def print_state(state: State):
    print(f'Hero turn: {state.is_hero_turn}')
    print(state.score.__dict__)
    for d in state.dice:
        print(d.__dict__)


if __name__ == '__main__':
    while True:
        if PATH is not None and path.isdir(PATH):
            for f in listdir(PATH):
                p = path.join(PATH, f)
                state = recognize_state(p)

                print_state(state)

                img = cv2.imread(p)
                cv2.imshow('window', cv2.resize(img, (800, 600)))
        else:
            state = recognize_state(PATH)
            print_state(state)
            cv2.waitKey(10)

        if PATH is not None and path.isfile(PATH):
            break
