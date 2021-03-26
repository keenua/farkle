# Done by Frannecklp

import cv2
import numpy as np
import win32gui, win32con, win32api, win32ui

last_path:str = None
last_screenshot: np.ndarray = None

def grab_screen(region=None, screenshot_path=None):
    if screenshot_path is not None:
        return __from_screenshot(region, screenshot_path)

    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,x2,y2 = region
            width = x2 - left
            height = y2 - top
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img

def __from_screenshot(region, screenshot_path):
    global last_screenshot
    global last_path

    x1, y1, x2, y2 = region
    
    is_cached = screenshot_path == last_path and last_screenshot is not None
    image = last_screenshot if is_cached else cv2.imread(screenshot_path)

    last_screenshot = image
    last_path = screenshot_path

    if region is None:
        return image.copy()

    return image[y1:y2, x1:x2].copy()