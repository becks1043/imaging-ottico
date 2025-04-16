#luce_vs_fnumber_imaging_ottico
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import ndimage
import os 

#deve variale come l'inverso del quadrato
cartella = r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\rumore\Gruppo 1\FNum"
images = []

file_list = sorted(os.listdir(cartella))
for file in file_list:
    path = os.path.join(cartella, file)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    images.append(img)

images= np.array(images, dtype = float)
#la prima è di buio
buio = np.mean(images[0].flatten())
images = images[1:] - images[0] #immagine pulite in questo caso ho sottatto il fondo 1 pixel a 1 pixel

#calcolo della sigma termica
sigma_termico = []
for i in range(8):
    fluttuazioni = images[2*i] - images[2*i + 1]
    arr = fluttuazioni.flatten()
    sigma_result = np.std(arr)
    sigmas = sigma_result/np.sqrt(2)
    sigma_termico.append(sigmas)

#segmentiamo le immagini
mean_images = np.mean(images, axis = 0)
mean_images= mean_images/ np.max(mean_images)
mask =  mean_images> 0.15
labeled_array, num_features = ndimage.label(mask)
mask_label_1 = (labeled_array == 1) #segmentiamo il led

#visualizziamo la segmentazione
plt.imshow(images[-1]*mask_label_1, cmap="gray")
plt.show()

#calcolo della sigma shot nella ROI
sigma_shot = np.sqrt(np.mean((images*mask_label_1).flatten())) #livello medio di grigio in una ROI uniforme

#sigma tot
sigma = np.sqrt(np.array(sigma_termico)**2 + np.array(sigma_shot)**2)
#prendiamo solo una foto per f numebr
f_n = [0.95, 1.4, 2., 2.8, 4., 5.6, 8., 11.]
intensity = []
for i in range(8):
    img = images[2*i]*mask_label_1
    roi_intensity = np.mean(img.flatten())
    intensity.append(roi_intensity)

intensity, f_n = np.array(intensity), np.array(f_n)

def function(x, a, b):
    return b/(a + x**2) 

popt, pcov = curve_fit(function, f_n, intensity, sigma = sigma, p0 = [ 1., 0.], absolute_sigma = False)
chi_square = np.sum((((intensity - function(f_n, *popt)) / sigma)**2))
dof = len(f_n) - len(popt)

print("-----")
print(f"chi quadro {chi_square}\ {dof}")
print(f"popt = {popt}")
print("-----")
x = np.linspace(min(f_n), max(f_n), 1000)

plt.errorbar(f_n, intensity, sigma, color="blue", fmt= ".",capsize=3, label="dati")
plt.plot(x, function(x, *popt), color="red", label="fit")
plt.xlabel("f/#")
plt.ylabel("intensità media del led [pixel/elettroni]")
plt.grid(which="both", linestyle="--",alpha=0.5)
plt.legend(title = f"$\chi^2$ = {np.round(chi_square, 1)} /{dof}")
plt.show()
