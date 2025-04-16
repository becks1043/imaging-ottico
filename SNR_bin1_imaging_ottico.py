#task2_imaging_ottico
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import ndimage
import os 

file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\rumore\Gruppo 1\SNR\Bin1"
file_list = sorted(os.listdir(file_path))
images = [] #a cui abbiamo sottratto la baseline
for file1, file2, file3 in zip(file_list[::3], file_list[1::3], file_list[2::3]):
    path1 = os.path.join(file_path, file1)
    path2 = os.path.join(file_path, file2)
    path3 = os.path.join(file_path, file3)
    # Leggi le immagini
    buio = cv2.imread(path1, cv2.IMREAD_UNCHANGED)
    dark_noise = np.mean(buio.flatten()) #calcolo della baseline
    img1 = cv2.imread(path2, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(path3, cv2.IMREAD_UNCHANGED)
    immagini = img2 - dark_noise
    images.append(immagini)
images= np.array(images)

"""
concatenated_img = np.concatenate(images, axis=1) 
plt.imshow(concatenated_img, cmap='gray')
#plt.colorbar()
plt.show()
"""
#segmentiamo le immagini
print(f"la shape del tensore images è {np.shape(images)}")
mean_images = np.mean(images, axis = 0)
mean_images_norm = mean_images/ np.max(mean_images)
mask =  mean_images_norm> 0.1
plt.imshow(mask, cmap= "gray")
plt.colorbar()
plt.show()

labeled_array, num_features = ndimage.label(mask)
mask_label_1 = (labeled_array == 1) #segmentiamo il led
mask_label_0= (labeled_array == 0) #sgmentiamo il background
"""
# Visualizzazione
plt.imshow(images[10]*mask_label_1, cmap="gray")
plt.colorbar()
plt.show()
"""
concatenated_img = np.concatenate(images*mask_label_1, axis=1) 
plt.imshow(concatenated_img, cmap='gray')
plt.show()

#signal to noise ratio
SNR = []
for i in range(np.shape(images)[0]):
    snr = (np.mean((images[i]*mask_label_1).flatten()) - np.mean((images[i]*mask_label_0).flatten()))/np.std((images[i]*mask_label_0).flatten())
    SNR.append(snr)

print(f"SNR={SNR}")
x_data = [0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50.] # tempi di esposizione quella a 100s manca per scelta

plt.scatter(x_data, SNR, color="blue", marker="x",cmap=1)
plt.xlabel("Tempi di esposizione [s]")
plt.ylabel("SNR [adm]")
plt.title("Andamento della SNR vs T_exp")
plt.show()
