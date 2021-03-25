import cv2
import numpy as np

FILE = 'e:\\Work\\ProjectFiles\\farkle\\screenshots\\379430_20210322013509_1.png'
OY, OX = (200, 600)
H, W = (600, 900)
name = 'image'

main = cv2.imread(FILE)
main = main[OY:OY+H, OX:OX+W]


def nothing(x):
    pass


def show_image(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_gray = np.array(lower, np.uint8)
    upper_gray = np.array(upper, np.uint8)
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    img_res = cv2.bitwise_and(img, img, mask=mask_gray)

    blur = cv2.bitwise_and(img, img, mask=mask_gray)
    blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    flag, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    im2, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    imgcont = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    out = imgcont.copy()

    mask = np.zeros_like(imgcont)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)

    for c in contours:
        area = cv2.contourArea(c)
        cv2.drawContours(imgcont, [c], 0, (0, 255, 0), 1)

        if area < 5000 and area > 1800:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.drawContours(mask, [c], 0, 255, -1)

    out = cv2.bitwise_and(out, out, mask=mask)

    row1 = np.hstack((img, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)))
    row2 = np.hstack((cv2.cvtColor(imgcont, cv2.COLOR_RGBA2BGR), cv2.cvtColor(out, cv2.COLOR_RGBA2BGR)))
    to_show = np.vstack((row1, row2))

    scale_percent = 25
    width = int(to_show.shape[1] * scale_percent / 100)
    height = int(to_show.shape[0] * scale_percent / 100)
    to_show = cv2.resize(to_show, (width, height))
    cv2.imshow(name, to_show)

cv2.namedWindow(name)

[h1, s1, v1]=[0, 5, 50]
[h2, s2, v2]=[179, 50, 255]


# create trackbars for color change
cv2.createTrackbar('H1', name, h1, 255, nothing)
cv2.createTrackbar('S1', name, s1, 255, nothing)
cv2.createTrackbar('V1', name, v1, 255, nothing)
cv2.createTrackbar('H2', name, h2, 255, nothing)
cv2.createTrackbar('S2', name, s2, 255, nothing)
cv2.createTrackbar('V2', name, v2, 255, nothing)

while(1):
    show_image(main.copy(), [h1, s1, v1], [h2, s2, v2])
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



cv2.destroyAllWindows()
