#SNR_all_bin_imaging_ottico
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import os 

images = [] #a cui abbiamo sottratto la baseline
SNR = []

file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\rumore\Gruppo 1\SNR\Bin1"
file_list = sorted(os.listdir(file_path))

for file1, file2, file3 in zip(file_list[::3], file_list[1::3], file_list[2::3]):
    path1 = os.path.join(file_path, file1)
    path2 = os.path.join(file_path, file2)
    path3 = os.path.join(file_path, file3)
    # Leggi le immagini
    buio = cv2.imread(path1, cv2.IMREAD_UNCHANGED)
    plt.imshow(buio, cmap="gray")
    plt.show()
    dark_noise = np.mean(buio.flatten()) #calcolo della baseline
    img1 = cv2.imread(path2, cv2.IMREAD_UNCHANGED)
    img1.astype(float)
    img2 = cv2.imread(path3, cv2.IMREAD_UNCHANGED)
    img2.astype(float)
    immagini = img2 - dark_noise
    images.append(immagini)
#segmentiamo le immagini
mean_images = np.mean(images, axis = 0)
mean_images_norm = mean_images/ np.max(mean_images)
mask =  mean_images_norm> 0.1
labeled_array, num_features = ndimage.label(mask)
mask_label_1 = (labeled_array == 1) #segmentiamo il led
mask_label_0= (labeled_array == 0) #sgmentiamo il background

#signal to noise ratio
for i in range(np.shape(images)[0]):
    snr = (np.mean((images[i]*mask_label_1).flatten()) - np.mean((images[i]*mask_label_0).flatten()))/np.std((images[i]*mask_label_0).flatten())
    SNR.append(snr)

x_data = [0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50.] # tempi di esposizione quella a 100s manca per scelta
print("------")
print(f"len(images)={len(images)}\nlen(SNR)={len(SNR)}")
SNR = np.array(SNR)
# Salva gli array su un file .npy
#np.save(r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\rumore\Gruppo 1\SNR\SNR_arr\bin4.npy", [SNR])
cartella_SNR_arr = r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\rumore\Gruppo 1\SNR\SNR_arr"
# Carica tutti i file .npy
file_npy = [f for f in os.listdir(cartella_SNR_arr)] #if f.endswith('.npy') se aggiungiamo altri fle diversi da .npy
arrays = [np.load(os.path.join(cartella_SNR_arr, f)) for f in file_npy]

#interpolazione delle SNR
colors = ["red", "orange", "green"]
for i in range(3):
    plt.scatter(np.array(x_data), arrays[i], color=colors[i], marker="x",cmap=1, label=f"binning {i*2}")
    
plt.xlabel("Tempi di esposizione [s]")
plt.ylabel("SNR [adm]")
plt.title("Andamento della SNR vs T_exp")
plt.grid(which="both", linestyle="--",alpha=0.5)
plt.xscale("log")
plt.legend()
plt.show()

#Soft binning pictures

images_bin2 = images[-1].reshape(256, 2, 256, 2).sum(axis=(1, 3))
images_bin4 = images[-1].reshape(128, 4, 128, 4).sum(axis=(1, 3))
images_bin8 = images[-1].reshape(64, 8, 64, 8).sum(axis=(1, 3))

fig, axes = plt.subplots(2,2, figsize=(6,5))
axes[0][0].imshow(images[-1], cmap="gray", label=f"dimensione {np.shape(images[-1])}")
axes[0][0].legend(title=f"dim {np.shape(images[-1])}")
axes[0][1].imshow(images_bin2, cmap="gray",label= f"dimensione {np.shape(images_bin2)}")
axes[0][1].legend(title = f"dim {np.shape(images_bin2)}")
axes[1][0].imshow(images_bin4, cmap="gray", label= f"dimensione {np.shape(images_bin4)}")
axes[1][0].legend(title =f"dim {np.shape(images_bin4)}")
axes[1][1].imshow(images_bin8, cmap="gray", label= f"dimensione {np.shape(images_bin8)}")
axes[1][1].legend(title=f"dim {np.shape(images_bin8)}")
plt.tight_layout()
plt.show()
#soft binning SNR
images_bin2 = []
images_bin4 = []
images_bin8 = []
for i in range(np.shape(images)[0]):
    bin2 = images[i].reshape(256, 2, 256, 2).sum(axis=(1, 3))
    images_bin2.append(bin2)
    bin4 = images[i].reshape(128, 4, 128, 4).sum(axis=(1, 3))
    images_bin4.append(bin4)
    bin8 = images[i].reshape(64, 8, 64, 8).sum(axis=(1, 3))
    images_bin8.append(bin8)

#segmentiamo le immagini
mean_images_bin2 = np.mean(images_bin2, axis = 0)
mean_images_norm2 = mean_images_bin2/ np.max(mean_images_bin2)
mask_bin2 =  mean_images_norm2> 0.1
labeled_array_bin2, num_features_bin2 = ndimage.label(mask_bin2)
mask_label_1_bin2 = (labeled_array_bin2 == 1) #segmentiamo il led
mask_label_0_bin2 = (labeled_array_bin2 == 0) #sgmentiamo il background

SNR_bin2 = []
#signal to noise ratio
for i in range(np.shape(images_bin2)[0]):
    snr = (np.mean((images_bin2[i]*mask_label_1_bin2).flatten()) - np.mean((images_bin2[i]*mask_label_0_bin2).flatten()))/np.std((images_bin2[i]*mask_label_0_bin2).flatten())
    SNR_bin2.append(snr)

print(SNR_bin2)
#np.save(r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\rumore\Gruppo 1\SNR\SNR_arr\soft_bin8.npy", [SNR_bin2])
#Plot delle SNR
colors1 = [ "blue", "purple", "gray"]
for i in range(3):
    plt.scatter(np.array(x_data), arrays[i], color=colors[i], marker="x",cmap=1, label=f"Hard binning {i*2}")
    plt.scatter(np.array(x_data), arrays[3+i], color=colors1[i], marker="x",cmap=1, label=f"Soft binning {2**(i+1)}")

plt.xlabel("Tempi di esposizione [s]")
plt.ylabel("SNR [adm]")
plt.title("Andamento della SNR vs T_exp")
plt.grid(which="both", linestyle="--",alpha=0.5)
plt.xscale("log")
plt.legend()
plt.show()