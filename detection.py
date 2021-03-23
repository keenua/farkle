import cv2
from matplotlib import pyplot as plt

SCREENSHOT_DIR = 'e:\\Games\\Steam\\Screenshots'

img = cv2.imread(f'{SCREENSHOT_DIR}/379430_20210322222336_1.png')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

main_img = img
  
stop_data = cv2.CascadeClassifier('data\\all\\cascade.xml') 

oy, ox = (300, 600)
h,w = (600, 900)

found = stop_data.detectMultiScale(main_img[oy:oy+h, ox:ox+w], minSize = (60, 60), maxSize = (100, 100)) 
  
amount_found = len(found) 
  
if amount_found != 0: 
      
    # There may be more than one 
    # sign in the image 
    for (x, y, width, height) in found: 
          
        # We draw a green rectangle around 
        # every recognized sign 
        cv2.rectangle(img_rgb, (x + ox, y + oy),  
                      (x + ox + height, y + oy + width),  
                      (0, 255, 0), 5) 
          
# Creates the environment of  
# the picture and shows it 
plt.subplot(1, 1, 1) 
plt.imshow(img_rgb) 
plt.show()