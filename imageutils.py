from typing import *
import cv2 
from random import randint
from uuid import uuid4
from os import path, listdir

SOURCE_DIR = 'e:\\Work\\ProjectFiles\\farkle\\train\\bg\\'
DEST_DIR = 'e:\\Work\\Projects\\farkle\\train\\bg\\'
COUNT_PER_FILE = 100

def random_size() -> Tuple[int, int]:
    min_w, max_w = (60, 90)
    min_h, max_h = (60, 90)

    w = randint(min_w, max_w)
    h = randint(min_h, max_h)

    return (w, h)

def break_into_pieces(imagepath: str, dest_dir: str, count: int):
    img = cv2.imread(imagepath)
    height, width, _ = img.shape

    for _ in range(count):
        (w, h) = random_size()

        x = randint(0, width - w)
        y = randint(0, height - h)

        part = img[y:y+h, x:x+w]
        dest_path = path.join(dest_dir, f'{uuid4()}.png')
        cv2.imwrite(dest_path, part)

if __name__ == '__main__':
    for f in listdir(SOURCE_DIR):
        filepath = path.join(SOURCE_DIR, f)
        break_into_pieces(filepath, DEST_DIR, COUNT_PER_FILE)