import os
from enum import Enum
from random import randint
from typing import *

import cv2
import numpy as np
from imutils import paths

Rect = Tuple[int, int, int, int]


class ColorModel(Enum):
    RGB = 1
    HSV = 2


class ColorRange():
    def __init__(self, model: ColorModel, lower: List[int], upper: List[int], invert: bool):
        self.model = model
        self.lower = lower
        self.upper = upper
        self.invert = invert
        self.cv2_converter = cv2.COLOR_BGR2HSV if model == ColorModel.HSV else cv2.COLOR_BGR2RGB

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.model, self.lower, self.upper, self.invert)

    def threshold(self, img: np.ndarray) -> np.ndarray:
        lowerb = np.array(self.lower, np.uint8)
        upperb = np.array(self.upper, np.uint8)

        mask = cv2.cvtColor(img, self.cv2_converter)
        mask = cv2.inRange(mask, lowerb, upperb)

        thresh = cv2.bitwise_and(img, img, mask=mask)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGRA2GRAY)
        thresh = cv2.threshold(
            thresh, 0, 255, cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY)[1]
        return thresh


class ColorRanges():
    ORANGE = ColorRange(ColorModel.RGB, [100, 0, 0], [255, 100, 50], True)
    GRAY = ColorRange(ColorModel.HSV, [21, 6, 0], [124, 71, 255], True)
    BRED = ColorRange(ColorModel.RGB, [100, 0, 0], [255, 30, 255], True)
    RED = ColorRange(ColorModel.RGB, [100, 0, 0], [255, 50, 50], True)
    BLACK = ColorRange(ColorModel.HSV, [0, 0, 60], [255, 255, 255], False)
    YELLOW = ColorRange(ColorModel.HSV, [0, 0, 0], [255, 255, 200], False)


class ContourInfo():
    def __init__(self, contour, area: float, bounding_rect: Rect):
        self.contour = contour
        self.area = area
        self.bounding_rect = bounding_rect


def remove_whitespace(img: np.ndarray) -> np.ndarray:
    gray = img.copy()
    gray = 255*(gray < 128).astype(np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones(
        (2, 2), dtype=np.uint8))
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    rect = img[y:y+h, x:x+w]
    return rect


def add_whitespace_around(img: np.ndarray, add_h: int = 2, add_w: int = 2) -> np.ndarray:
    h, w = img.shape
    result = np.full((h + add_h * 2, w + add_w * 2), 255, dtype='uint8')
    result[add_h:h+add_h, add_w:w+add_w] = img
    return result


def detect_contours(img: np.ndarray, color_range: ColorRange) -> List[ContourInfo]:
    result = []

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    thresh = color_range.threshold(img)
    contours = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

    for c in contours:
        area = cv2.contourArea(c)
        bounding_rect = cv2.boundingRect(c)
        info = ContourInfo(c, area, bounding_rect)
        result.append(info)

    return result


def center(rect: Tuple[int, int, int, int]) -> np.ndarray:
    (x, y, width, height) = rect
    return np.array((x + width // 2, y + height // 2))


def dhash(image: np.ndarray, hashSize: int = 8) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def remove_duplicates(dir: str):
    imagePaths = list(paths.list_images(dir))

    print(f'Was: {len(imagePaths)}')

    hashes = {}
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        h = dhash(image)
        p = hashes.get(h, [])
        p.append(imagePath)
        hashes[h] = p

    for v in hashes.values():
        if len(v) > 1:
            images = [(i, cv2.imread(i)) for i in v]

            groups = []

            for path, image in images:
                found = False
                for g in groups:
                    if np.array_equal(g[0][1], image):
                        g.append((path, image))
                        found = True
                        break
                if not found:
                    groups.append([(path, image)])

            for g in groups:
                for p, _ in g[1:]:
                    os.remove(p)

    imagePaths = list(paths.list_images(dir))

    print(f'Now: {len(imagePaths)}')
