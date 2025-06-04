#cold_white_spectrum
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import os
import cv2

file_path_spectrum = r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\coldWhiteLEDspectrum.txt"
cartella = r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\rumore\Gruppo 1\SpettroLuceBianca"

file_list = os.listdir(cartella)
images = []
for file1, file2 in zip(file_list[::2], file_list[1::2]):
    path1 = os.path.join(cartella, file1)
    path2 = os.path.join(cartella, file2)
    try:
        if path1.lower().endswith(".tif") and path2.lower().endswith(".tif"):
            buio = cv2.imread(path1, cv2.IMREAD_UNCHANGED).astype(float)
            img = cv2.imread(path2, cv2.IMREAD_UNCHANGED).astype(float)
            img -= buio
            images.append(img)
    except Exception as e:
        print(f"Errore con i file {file1} e {file2}: {e}")

landa, fotoni = np.loadtxt(file_path_spectrum, unpack = True)

plt.plot(landa, fotoni, color="darkblue")
plt.xlabel(fr"$\lambda$[nm]")
plt.ylabel(fr"Intensità [A.U]")
plt.title("Spettro di riferimento")
plt.grid(linestyle="--")
plt.show()

#plt.imshow(np.hstack(images))
#segmentiamo le immagini
mean_img = np.mean(images, axis=0)
norm_img = mean_img/np.max(mean_img)
mask = norm_img > 0.1
labeled_array, num_features = ndimage.label(mask)
label1 = (labeled_array == 1) #segmentiamo il led
label2 = (labeled_array == 0) #sgmentiamo il background
images = np.array(images)
plt.imshow(np.hstack(images*label1), cmap="gray")
plt.show()

led = images*label1
led = led.reshape(5, -1)
adc_counts = np.mean(led, axis=1)
#adc_counts /= np.max(adc_counts)
print(adc_counts)

#numero di fotoni per lo spettro
filters = np.array([635., 532., 488., 457., 405.]) #filtri passa banda
CF = np.array([3.6, 3.6, 3.6, 3.6, 3.6]) # dovrebbe essere costante per tutti
QE = np.array([0.95, 0.85, 0.8, 0.75, 0.6 ]) #quantum efficiency
t_exp = np.array([5., 2., 5., 1.5, 100]) #s
Tr = np.array([0.9,0.9,0.9,0.9,0.8]) #trasmittanza del filtro
N_f = (adc_counts/t_exp)*CF/QE
N_f /= Tr

plt.scatter(filters, N_f/np.max(N_f), color="darkred")
plt.plot(landa, fotoni, color="darkblue")
plt.show()