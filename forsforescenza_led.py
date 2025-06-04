#forsforescenza_led
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import ndimage
import tifffile as tiff


cartella = r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\rumore\Gruppo 1\Fosforescenza"
dark = r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\rumore\Gruppo 1\Fosforescenza\buio.TIF"
file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\rumore\Gruppo 1\Fosforescenza\MyStreamFile00001.TIF"

led_fosf = tiff.imread(file_path).astype(float)
buio = tiff.imread(dark).astype(float)
buio_mean = np.mean(buio, axis=0)
led_fosf -= buio_mean

#segmentiamo le immagini
mean_img = np.mean(led_fosf, axis=0)
norm_img = mean_img/np.max(mean_img)
mask = norm_img > 0.1
labeled_array, num_features = ndimage.label(mask)
label1 = (labeled_array == 1) #segmentiamo il led
label2 = (labeled_array == 0) #sgmentiamo il background

led = led_fosf*label1
plt.imshow(np.hstack(led[:10,:,:]))
plt.show()

#intensità di fluorescenza e tempi
int_roi = np.mean(led, axis=(1,2))  #intensità media del led 
t = np.arange(0, 100, 1)
print(int_roi.shape)
#errori
sigma_termico = np.std(buio, axis=0)
sigma_termico = np.mean(sigma_termico)
sigma_shot = np.sqrt(np.mean(led, axis= (1,2))) #livello medio di grigio in una ROI uniforme
sigma_shot *= np.sqrt(3.6) #Conversion Factor per passare in elettroni
sigma = np.sqrt(np.array(sigma_termico)**2 + np.array(sigma_shot)**2)

#fit 
def exp(x, a, b ,c, d, e):
    return (a*np.exp(-x/b)+d*np.exp(-x/c)) + e #*np.heaviside(x, 0)

guess = None #[4000, 38, 4]
popt, pcov = curve_fit(exp, t[1:], int_roi[1:], sigma = sigma[1:], p0= guess, absolute_sigma = False, maxfev = 10000)
chi_square = np.sum((((int_roi[1:] - exp(t[1:], *popt)) / sigma[1:])**2))
dof = len(t) - len(popt)

print("-----")
print(f"chi quadro {chi_square}\ {dof}")
print(f"popt = {popt}")
print("-----")

x = np.linspace(0, max(t), 1000)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,6))
ax1.errorbar(t, int_roi, sigma, color="darkblue", fmt=".")
ax1.plot(x, exp(x, *popt), color="darkred")
ax1.set_xlabel("t [s]")
#ax1.set_ylabel("intensità [A.U.]")
ax1.grid(True, linestyle="--")
# Grafico residui
ax2.plot(t, (int_roi - exp(t, *popt))/sigma, 'o', color='darkred')
ax2.axhline(0, color='black', linestyle='--')
ax2.set_xlabel("Tempo [ms]")
ax2.set_ylabel("Residui")
ax2.grid(True, linestyle="--")
plt.show()