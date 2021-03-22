import cv2
from matplotlib import pyplot as plt

SCREENSHOT_DIR = 'e:/Work/ProjectFiles/farkle/screenshots'

img = cv2.imread(f'{SCREENSHOT_DIR}/379430_20210322013509_1.png')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

main_img = img_gray
  
stop_data = cv2.CascadeClassifier('data\\1\\cascade.xml') 

found = stop_data.detectMultiScale(main_img,  
                                   minSize = (60, 60), maxSize = (100, 100)) 
  
# Don't do anything if there's  
# no sign 
amount_found = len(found) 
  
if amount_found != 0: 
      
    # There may be more than one 
    # sign in the image 
    for (x, y, width, height) in found: 
          
        # We draw a green rectangle around 
        # every recognized sign 
        cv2.rectangle(img_rgb, (x, y),  
                      (x + height, y + width),  
                      (0, 255, 0), 5) 
          
# Creates the environment of  
# the picture and shows it 
plt.subplot(1, 1, 1) 
plt.imshow(img_rgb) 
plt.show()