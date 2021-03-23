from grabscreen import grab_screen
import cv2

while True:
    img = grab_screen((600, 300, 1500, 900))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)

    classifier = cv2.CascadeClassifier('data\\all\\cascade.xml') 
    found = classifier.detectMultiScale(img, minSize = (75, 75), maxSize = (95, 95)) 
    
    for (x, y, width, height) in found: 
        cv2.rectangle(img, (x, y), (x + height, y + width), (0, 255, 0), 5) 

    cv2.imshow('cv2', img)
    cv2.waitKey(10)