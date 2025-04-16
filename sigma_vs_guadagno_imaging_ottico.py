#sigma_vs_temperatura_imaging_ottico
import numpy as np
from scipy.optimize import curve_fit
import cv2
import matplotlib.pyplot as plt
import os 

cartella = r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\rumore\Gruppo 1\rumore\lettura" #cartella dei rumori di lettura

y_data = [] #sigma delle immagine
x_data = [0., 1., 2.] #guadagno

# Ottieni tutti i file nella cartella e ordina (se serve)
file_list = sorted(os.listdir(cartella))
# Leggi i file a due a due
for file1, file2 in zip(file_list[::2], file_list[1::2]):
    path1 = os.path.join(cartella, file1)
    path2 = os.path.join(cartella, file2)
    #print(f"Lettura di {path1} e {path2}")
    
    # Leggi le immagini
    img1 = cv2.imread(path1, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(path2, cv2.IMREAD_UNCHANGED)

    # Ottieni la matrice dei livelli di grigio
    gray_matrix1 = np.array(img1)
    gray_matrix1 = gray_matrix1.astype(float)

    gray_matrix2 = np.array(img2)
    gray_matrix2 = gray_matrix2.astype(float)

    #rumore termico 
    gray_matrix_result = gray_matrix1 - gray_matrix2
    arr = gray_matrix_result.flatten()
    sigma_result = np.std(arr)
    sigma = sigma_result/np.sqrt(2)
    y_data.append(sigma)
    

print(f"sigma {y_data}\n{len(y_data)}")
def linear(x, a, b):
    return x*a + b 

popt, _ = curve_fit(linear, x_data, y_data)

x = np.linspace(0, 2.5, 1000)
plt.scatter(x_data, y_data, color="blue", label=f"data")
plt.plot(x, linear(x, *popt), color = "r", label="fit lineare")
plt.legend()
plt.xlabel("Guadagno CCD")
plt.ylabel("sigma [adm]")
plt.grid(which="both", linestyle="--",alpha=0.5)
plt.show()