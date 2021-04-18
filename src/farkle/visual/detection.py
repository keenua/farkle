import cv2
import numpy as np
from farkle.utils import *
from farkle.visual.recognition import recognize
from matplotlib import pyplot as plt

MIN_AREA = 300
MIN_DIE_AREA = 1800
MAX_AREA = 5000


def fix_rotation(img: np.ndarray, cnt) -> np.ndarray:
    rect = cv2.minAreaRect(cnt)

    box = cv2.boxPoints(rect)
    box = np.int0(box)

    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped


def detect_dice(img: np.ndarray) -> Tuple[Rect, np.ndarray]:
    result = []

    contours = detect_contours(img, ColorRanges.GRAY)

    for c in contours:
        if MIN_DIE_AREA < c.area < MAX_AREA:
            res = fix_rotation(img, c.contour)
            result.append((c.bounding_rect, res))

    return result


def detect_objects_by_color(img: np.ndarray, color_range: ColorRange) -> List[np.ndarray]:
    return [center(c.bounding_rect) for c in detect_contours(img, color_range) if MIN_AREA < c.area < MAX_AREA]


def detect_hold_markers(img: np.ndarray) -> List[np.ndarray]:
    return detect_objects_by_color(img, ColorRanges.ORANGE)


def detect_selection_marker(img) -> List[np.ndarray]:
    return detect_objects_by_color(img, ColorRanges.YELLOW)


def demo(filepath: str):
    OY, OX = (200, 600)
    H, W = (600, 900)

    img = cv2.imread(filepath)
    img = img[OY:OY+H, OX:OX+W]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for (x, y, w, h), die in detect_dice(img):
        res = recognize(die)
        print(res)

        plt.subplot(1, 1, 1)
        plt.imshow(img_rgb[y:y+h, x:x+w])
        plt.show()
