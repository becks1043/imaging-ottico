#ROI_selection
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import ndimage
import os 
file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\rumore\Gruppo 1\SNR\Bin1\misura0033.TIF"
# Leggi le immagini
img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
img_copy =img.copy()
roi = []

def mouse_callback(event, x, y, flags, param):
    global roi, img_copy
    if event == cv2.EVENT_LBUTTONDOWN: 
        roi = [(x,y)]
    elif event== cv2.EVENT_LBUTTONUP:
        roi.append((x,y))
        cv2.rectangle(img_copy, roi[0], roi[1], (0, 255, 0), 2)
        cv2.imshow("selezione ROI", img_copy)

#mostra l'immagine e imposta il callback
cv2.imshow("selezione ROI", img)
cv2.setMouseCallback("selezione ROI", mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()
#estraiamo la roi
if len(roi)==2:
    x1, y1 =roi[0]
    x2, y2 =roi[1]
    roi_cropped = img[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
    cv2.imshow("ROI", roi_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
