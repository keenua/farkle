import cv2
from os import listdir, path
from uuid import uuid4

#SCREENSHOT_DIR = 'e:/Work/ProjectFiles/farkle/screenshots'
SCREENSHOT_DIR = 'e:\\Games\\Steam\\Screenshots'
DEST_DIR = 'e:/Work/Projects/farkle/train/under'

for f in listdir(SCREENSHOT_DIR):
    img = cv2.imread(path.join(SCREENSHOT_DIR, f))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    stop_data = cv2.CascadeClassifier('data\\all\\cascade.xml') 

    found = stop_data.detectMultiScale(img,  
                                    minSize = (60, 60), maxSize = (100, 100)) 
  
    if len(found) != 0: 
        for (x, y, width, height) in found: 
            potential_die = img[y:y+height, x:x+width]
            cv2.imwrite(path.join(DEST_DIR, f'{uuid4()}.png'), potential_die)