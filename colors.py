from typing import Counter
import cv2
from matplotlib import pyplot as plt
import numpy as np
from os import listdir, path
from math import ceil

from numpy.lib.function_base import average

SCREENSHOT_DIR = 'e:\\Work\\ProjectFiles\\farkle\\screenshots\\'

rects = [
    (209, 314, 85, 82),
    (97, 344, 82, 82),
    (111, 235, 86, 82),
    (321, 289, 78, 79),
    (269, 200, 77, 73),
    (174, 136, 83, 80)
]

def get_colors(img):
    colors = [(int(average(color)), int(average(color))) for x in img for color in x]
    counter = Counter(colors)
    del counter[(0,0)]
    return counter

def get_heatmap(colors):
    heat = np.zeros((256, 256), dtype= 'float')
    m = max(colors.values())

    for (min_c, range), count in colors.items():
        heat[min_c, range] = count / m

    return heat


for f in listdir(SCREENSHOT_DIR):
    if f != '379430_20210322013509_1.png':
        continue

    OY,OX = (200, 600)
    H,W = (600, 900)
    
    p = path.join(SCREENSHOT_DIR, f)
    img = cv2.imread(p)
    img = img[OY:OY+H, OX:OX+W]

    mask = np.zeros((H, W), dtype= 'uint8')

    for x,y,w,h in rects:
        mask[y:y+h,x:x+w] = 255
    
    good = cv2.bitwise_and(img, img, mask=mask)
    bad = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

    good_colors = get_colors(good)
    bad_colors = get_colors(bad)
    unique_colors = good_colors.copy()
    all(map(lambda x: unique_colors.pop(x, 0), bad_colors))

    top_colors = Counter({c: v for c, v in unique_colors.most_common(200)})
    print(top_colors)

    fig = plt.figure(figsize=(2, 2))
    fig.add_subplot(2, 2, 1) 
    plt.imshow(get_heatmap(good_colors), cmap='hot', interpolation='nearest')
    fig.add_subplot(2, 2, 2) 
    plt.imshow(get_heatmap(bad_colors), cmap='hot', interpolation='nearest')
    fig.add_subplot(2, 2, 3) 
    plt.imshow(get_heatmap(unique_colors), cmap='hot', interpolation='nearest')
    fig.add_subplot(2, 2, 4) 
    plt.imshow(get_heatmap(top_colors), cmap='hot', interpolation='nearest')
    plt.show()

    # plt.imshow(cv2.cvtColor(good, cv2.COLOR_BGR2RGBA))
    # plt.show()

    # plt.imshow(cv2.cvtColor(bad, cv2.COLOR_BGR2RGBA))
    # plt.show()