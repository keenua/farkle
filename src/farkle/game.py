import sys
from typing import *

import cv2
import numpy as np

from farkle.logic.dm import move
from farkle.utils import grab_screen
from farkle.visual.state import DICE_REGION, Die, recognize_state

MAX_POINTS = 4000


def text(img: np.ndarray, text: str, pos: tuple):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = pos
    fontScale = 0.5
    fontColor = (0, 255, 0)
    lineType = 2

    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)


def print_line(text: str):
    space = ' ' * 20
    print(f'{text} {space}', end='\r')


def show_dice(dice: List[Die]):
    img = grab_screen(DICE_REGION)
    r = 2

    for d in dice:
        [x, y] = d.center
        cv2.rectangle(img, (x - r, y - r), (x + r, y + r), (0, 255, 0), r * 2)
        text(img, str(d.value), (x - 20, y - 20))

    cv2.imshow('window', img)
    cv2.waitKey(10)


def check_selection(state, keep):
    to_hold = keep.copy()
    for d in state.dice:
        if d.held:
            to_hold.remove(d.value)

    if not to_hold:
        return True

    current = [d for d in state.dice if d.selected][0]

    if not current.held and current.value in to_hold:
        print_line(f'select current {current.value}')
        return False

    target = [d for d in state.dice if not d.held and to_hold[0] == d.value][0]

    [x, y] = target.center - current.center
    if abs(x) > abs(y):
        if x > 0:
            print_line(
                f'move right from {current.value} towards {target.value}')
        else:
            print_line(
                f'move left from {current.value} towards {target.value}')
    else:
        if y > 0:
            print_line(
                f'move down from {current.value} towards {target.value}')
        else:
            print_line(f'move up from {current.value} towards {target.value}')

    return False


def play():
    while True:
        try:
            state = recognize_state()

            if not state.is_hero_turn or len(state.dice) == 0 or state.score.goal != MAX_POINTS:
                print_line(f'Waiting for hero\'s turn')
                continue

            dice = [d.value for d in state.dice]
            (keep, should_roll) = move(dice, MAX_POINTS, state.score.hero_round,
                                       state.score.hero_total, state.score.opp_total)

            show_dice(state.dice)

            if not check_selection(state, keep):
                continue

            if should_roll:
                print_line(f'Roll')
            else:
                print_line(f'Bank')
        except Exception:
            e = sys.exc_info()[0]
            print(f'failed with {e}\n')
