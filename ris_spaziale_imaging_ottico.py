#ris_spaziale_imaging_ottico
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq

import os 

cartella= r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\rumore\Gruppo 1\RisSpaziale"

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

fig, axes = plt.subplots(2,2)
for i in range(len(images)):
    r,c = i//2, i%2
    axes[r][c].imshow(images[i], cmap ="gray")
    #axes[r][c].colorbar()
    axes[r][c].set_title(f"Dim {int(512/(2**(i)))}x{int(512/(2**(i)))}")
plt.tight_layout()
plt.show()
    
#CTF contrast transfer function 

plt.imshow(images[0], cmap= "gray")
plt.show()
#scegliamo un profilo dall'immagine e plottiamolo
xa, xb = 135 , 150 #gruppo -2
ya, yb = 140 , 320
xa1, xb1 = 260, 300
ya1, yb1 = 275, 317
#media su più profili per stabilizzare la risposta
ROI = np.concatenate([np.mean(images[0][ya1:yb1+1, xa1:xb1+1],axis=1), np.mean(images[0][ya:yb+1, xa:xb+1], axis=1)])
ROI /= np.max(ROI)

#conversione pixel lunghezza
pixel_mm = 0.341 #mm

ROI_profile = ROI 
y_pixel = np.linspace(0, np.shape(ROI)[0], np.shape(ROI)[0])

plt.plot(y_pixel*pixel_mm, ROI_profile, color="b")
plt.xlabel("Distanza [mm]")
plt.ylabel("Intensità [adm]")
plt.title("Profilo d'intensità della ROI")
plt.grid()
plt.show()
#DFT
# Calcolo della trasformata di Fourier (FFT)
N = np.shape(y_pixel)[0]
Fs = 1/pixel_mm
fft_result = np.fft.fft(ROI_profile)
freq = np.fft.fftfreq(N, d=1/Fs)  # Frequenze corrispondenti

# Prendo solo la metà positiva dello spettro
mask = freq >= 0
freq = freq[mask]
amplitude = np.abs(fft_result[mask]) / N  # Normalizzazione

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(freq, amplitude)
plt.title("Trasformata di Fourier del segnale")
plt.xlabel("Frequenza (Hz)")
plt.ylabel("Ampiezza")
plt.grid(True)
plt.tight_layout()
plt.show()

# Frequenze spaziali (es: cicli/mm)
M = 1.1935 #fattore di magnificazione
frequenze = np.array([0.25 ,0.280, 0.315, 0.353, 0.397, 0.445, 
                      0.500, 0.561, 0.630, 0.707, 0.793, 0.891,
                      1., 1.12, 1.26, 1.41, 1.59, 1.78,
                      2.0, 2.24, 2.52, 2.83, 3.17, 3.56])  #f standard fantoccio UTAF
frequenze /= M

# Intensità max/min misurate su ogni gruppo di barre (ipotetiche)
ffile = r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\intensita_USAF.cvs"
df = pd.read_csv(ffile, skiprows=1,delimiter="\t", names=["column","area", "media", "min", "max"])

I_max = np.array(df["max"]).astype(float)/df["max"][0]
I_min = np.array(df["min"]).astype(float)/df["max"][0]
# CTFS
CTF = (I_max - I_min) / (I_max + I_min)
def exp(x, a, b):
    return a*np.exp(-b*x)
popt, pcov = curve_fit(exp, frequenze, CTF)
x = np.linspace(0, max(frequenze), 1000)
# Plot
plt.scatter(frequenze, CTF, color='darkred', label="CTF")
plt.plot(x, exp(x, *popt), color="blue")
plt.xlabel("Frequenza spaziale (cicli/mm)")
plt.ylabel("CTF")
plt.title("Contrast Transfer Function")
plt.ylim(0, 1.1)
plt.grid(True)
plt.legend()
plt.show()

