from farkle.visual.recognition import recognize
import cv2
from os import listdir, mkdir, path
from shutil import move

SCREENSHOT_DIR = 'samples'

def sort_samples(dir: str = 'samples'):
    for i in range(1, 7):
        dir = path.join(SCREENSHOT_DIR, str(i))
        if not path.isdir(dir):
            mkdir(dir)

    for f in listdir(SCREENSHOT_DIR):
        fp = path.join(SCREENSHOT_DIR, f)
        if path.isfile(fp):
            img = cv2.imread(fp)
            res = recognize(img)
            dest = path.join(SCREENSHOT_DIR, str(res), f)
            move(fp, dest)
