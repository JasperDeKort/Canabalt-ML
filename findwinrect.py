# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:28:05 2017

@author: jaspe
"""

import win32gui
import win32con
#import time
#import cv2
#import numpy as np
#from grabscreen import grab_screen

def find_canabalt():
    # coordinates are hard coded for now. win32gui coordinates seem to have some extra side margin
    # these coordinates were tested to find the right image and return a proper (600,800,3) window using grab_screen
    winid = win32gui.FindWindow(None,"Canabalt")
    win32gui.SetWindowPos(winid,win32con.HWND_TOP, -7,0,816,640,0)
    return (1,32,800,631)

## main only used for testing purposes. commented out so unneeded imports could be removed.
def main():
#    time.sleep(2)
#    winid = win32gui.FindWindow(None,"Canabalt")
#    win32gui.SetWindowPos(winid,win32con.HWND_TOP, -7,0,816,640,0)
#    print(win32gui.GetWindowRect(winid))
#    savable = grab_screen(region=(1,32,800,631))
#    death = savable[:][250:280]
#    np.save("death250to280.npy",death)
#    print(savable.shape)
#    while True:
#        cv2.imshow('processed view', death)
#        if cv2.waitKey(25) & 0xFF == ord('q'):
#            cv2.destroyAllWindows()
#            break
    pass

if __name__ == "__main__":
    main()