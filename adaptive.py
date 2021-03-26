from pytesseract.pytesseract import image_to_string
from imageutils import ColorModel, thresh_by_color
import cv2
import numpy as np

# FILE = 'e:\\Work\\ProjectFiles\\farkle\\screenshots\\379430_20210322013509_1.png'
FILE = 'opp_round.png'
OY, OX = (200, 600)
H, W = (600, 900)


COLOR_MODEL = ColorModel.RGB
INVERT = True
SCALE = 100
CROP = False

[h1, s1, v1]=[100,0,0]
[h2, s2, v2]=[255,50,50]

# [h1, s1, v1]=[0, 5, 50]
# [h2, s2, v2]=[179, 50, 255]

name = 'image'

main = cv2.imread(FILE)
cv2.namedWindow(name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(name, 600, 1200) 

if CROP:
    main = main[OY:OY+H, OX:OX+W]

def nothing(x):
    pass


def show_image(img, lower, upper, t):
    thresh = thresh_by_color(img, lower, upper, COLOR_MODEL, INVERT)
    to_show = np.hstack((img, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)))

    if SCALE != 100:
        width = int(to_show.shape[1] * SCALE / 100)
        height = int(to_show.shape[0] * SCALE / 100)
        to_show = cv2.resize(to_show, (width, height))

    cv2.imshow(name, to_show)

cv2.namedWindow(name)

t1 = 0
t2 = 255

# create trackbars for color change
cv2.createTrackbar('H1', name, h1, 255, nothing)
cv2.createTrackbar('S1', name, s1, 255, nothing)
cv2.createTrackbar('V1', name, v1, 255, nothing)
cv2.createTrackbar('H2', name, h2, 255, nothing)
cv2.createTrackbar('S2', name, s2, 255, nothing)
cv2.createTrackbar('V2', name, v2, 255, nothing)

cv2.createTrackbar('T1', name, t1, 255, nothing)
cv2.createTrackbar('T2', name, t2, 255, nothing)

while(1):
    show_image(main.copy(), [h1, s1, v1], [h2, s2, v2], [t1, t2])
    k=cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    h1=cv2.getTrackbarPos('H1', name)
    s1=cv2.getTrackbarPos('S1', name)
    v1=cv2.getTrackbarPos('V1', name)

    h2=cv2.getTrackbarPos('H2', name)
    s2=cv2.getTrackbarPos('S2', name)
    v2=cv2.getTrackbarPos('V2', name)

    t1=cv2.getTrackbarPos('T1', name)
    t2=cv2.getTrackbarPos('T2', name)

cv2.destroyAllWindows()
