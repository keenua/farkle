from os import path, listdir
from recognition import recognize
import cv2
from detection import detect_dice
from grabscreen import grab_screen
from typing import *

from score import Score, recognize_score

DICE_REGION = (600, 200, 1500, 900)
TURN_REGION = (1732, 800, 1733, 801)
DEBUG = True

PATH = 'e:\\Work\\ProjectFiles\\farkle\\train_old2\\test2.png'

class Die:
    def __init__(self, value:int):
        self.value: int = value
        self.selected = False
        self.held = False

class State:
    def __init__(self):
        self.score: Score = None
        self.is_hero_turn = False
        self.dice: List[Die] = []

def recognize_dice(screenshot: str = None) -> List[Die]:
    dice = []

    img = grab_screen(DICE_REGION, screenshot)
    for (x, y, width, height), die in detect_dice(img):
        value = recognize(die)

        if value > 0:
            dice.append(Die(value))

        if DEBUG:
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 5) 

    if DEBUG:
        cv2.imshow('window', img)

    return dice

def recognize_turn(screenshot: str = None) -> bool:
    img = grab_screen(TURN_REGION, screenshot)
    return (img == 255).all()

def recognize_state(screenshot: str = None) -> State:
    state = State()

    state.dice = recognize_dice(screenshot)
    state.score = recognize_score(screenshot)
    state.is_hero_turn = recognize_turn(screenshot)

    return state

def print_state(state: State):
    print(f'Hero turn: {state.is_hero_turn}')
    print(state.score.__dict__)
    for d in state.dice:
        print(d.__dict__)

if __name__ == '__main__':
    while True:
        if path.isdir(PATH):
            for f in listdir(PATH):
                p = path.join(PATH, f)
                state = recognize_state(p)

                print_state(state)

                img = cv2.imread(p)
                cv2.imshow('window', cv2.resize(img, (800, 600)))
                cv2.waitKey(0)
        else: 
            state = recognize_state(PATH)
            print_state(state)
            if DEBUG:
                cv2.waitKey(0)
        
        if PATH is not None and path.isfile(PATH):
            break
