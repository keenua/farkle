from typing import *
import cv2
from random import randint
import numpy as np
from enum import Enum


class ColorModel(Enum):
    RGB = 1
    HSV = 2


ColorRange = Tuple[ColorModel, List[int], List[int], bool]

ORANGE: ColorRange = (ColorModel.RGB, [100, 0, 0], [255, 100, 50], True)
GRAY: ColorRange = (ColorModel.HSV, [21, 6, 0], [124, 71, 255], True)
BRED: ColorRange = (ColorModel.RGB, [100, 0, 0], [255, 30, 255], True)
RED: ColorRange = (ColorModel.RGB, [100, 0, 0], [255, 50, 50], True)
BLACK: ColorRange = (ColorModel.HSV, [0, 0, 60], [255, 255, 255], False)
YELLOW: ColorRange = (ColorModel.HSV, [0, 0, 0], [255, 255, 200], False)


def random_size() -> Tuple[int, int]:
    min_w, max_w = (60, 90)
    min_h, max_h = (60, 90)

    w = randint(min_w, max_w)
    h = randint(min_h, max_h)

    return (w, h)


def thresh_by_color(img: np.ndarray, color_range: ColorRange) -> np.ndarray:
    (color_model, lower, upper, invert) = color_range

    lowerb = np.array(lower, np.uint8)
    upperb = np.array(upper, np.uint8)

    color = cv2.COLOR_BGR2HSV if color_model == ColorModel.HSV else cv2.COLOR_BGR2RGB

    mask = cv2.cvtColor(img, color)
    mask = cv2.inRange(mask, lowerb, upperb)

    thresh = cv2.bitwise_and(img, img, mask=mask)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGRA2GRAY)
    thresh = cv2.threshold(
        thresh, 0, 255, cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY)[1]
    return thresh


def remove_whitespace(img):
    gray = img.copy()
    gray = 255*(gray < 128).astype(np.uint8)  # To invert the text to white
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones(
        (2, 2), dtype=np.uint8))  # Perform noise filtering
    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    # Crop the image - note we do this on the original image
    rect = img[y:y+h, x:x+w]
    return rect


def add_whitespace_around(img, add_h=2, add_w=2):
    h, w = img.shape
    result = np.full((h + add_h * 2, w + add_w * 2), 255, dtype='uint8')
    result[add_h:h+add_h, add_w:w+add_w] = img
    return result
