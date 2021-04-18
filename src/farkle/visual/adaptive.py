from typing import List, Tuple
import cv2
import numpy as np

from farkle.utils import ColorModel, ColorRange, ColorRanges

OY, OX = (200, 600)
H, W = (600, 900)

SCALE = 100
CROP = False

STARTING_COLOR = ColorRanges.RED
WINDOW_W, WINDOW_H = (600, 1200)

WINDOW_NAME = 'experiment'


def nothing(_):
    pass


def show_image(img: np.ndarray, color_range: ColorRange):
    thresh = color_range.threshold(img)
    to_show = np.hstack((img, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)))

    if SCALE != 100:
        width = int(to_show.shape[1] * SCALE / 100)
        height = int(to_show.shape[0] * SCALE / 100)
        to_show = cv2.resize(to_show, (width, height))

    cv2.imshow(WINDOW_NAME, to_show)


def create_trackbars(color_range: ColorRange) -> List[str]:
    parts = ['R', 'G', 'B'] if color_range.model == ColorModel.RGB else [
        'H', 'S', 'V']

    captions = [f'{m} {p}' for m in ['min', 'max'] for p in parts]

    values = color_range.lower + color_range.upper

    for (caption, value) in zip(captions, values):
        cv2.createTrackbar(caption, WINDOW_NAME, value, 255, nothing)

    return captions


def get_trackbar_values(names: List[str]) -> List[int]:
    return [cv2.getTrackbarPos(n, WINDOW_NAME) for n in names]


def create_window(file: str, color_range: ColorRange) -> Tuple[np.ndarray, List[str]]:
    img = cv2.imread(file)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

    if CROP:
        img = img[OY:OY+H, OX:OX+W]

    trackbars = create_trackbars(color_range)

    return img, trackbars


def experiment(file: str):
    color_range = STARTING_COLOR
    main, trackbars = create_window(file, color_range)

    while(1):
        show_image(main.copy(), color_range)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        all = get_trackbar_values(trackbars)
        color_range = ColorRange(color_range.model, all[:3], all[3:], color_range.invert)

    cv2.destroyAllWindows()
    
