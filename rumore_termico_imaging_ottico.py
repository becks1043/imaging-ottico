#rumore_termico_imaging_ottico
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os 

# Percorso della cartella
cartella = r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\rumore\termico\0C"

# Ottieni tutti i file nella cartella e ordina (se serve)
file_list = sorted(os.listdir(cartella))

y_data = [] #sigma delle immagini
x_data = [0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100.] # tempi di esposizione

#istogrammi
count = [] #conteggi dei livelli di grigio
count_max = [] #massimo conteggio del livello di grigio
matrice_grigi = [] #matrici non sottratte
# Leggi i file a due a due
for file1, file2 in zip(file_list[::2], file_list[1::2]):
    path1 = os.path.join(cartella, file1)
    path2 = os.path.join(cartella, file2)
    print(f"Lettura di {path1} e {path2}")
    
    # Leggi le immagini
    img1 = cv2.imread(path1, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(path2, cv2.IMREAD_UNCHANGED)

    # Ottieni la matrice dei livelli di grigio
    gray_matrix1 = np.array(img1)
    gray_matrix1 = gray_matrix1.astype(float)

    gray_matrix2 = np.array(img2)
    gray_matrix2 = gray_matrix2.astype(float)
    matrice_grigi.append(gray_matrix1) 
    #matrice_grigi.append(gray_matrix2)

    print(f"Dimensione: {gray_matrix1.shape}")  # Altezza x Larghezza
    print(f"type dell'array {type(gray_matrix1)}")
    print(f"Min: {gray_matrix1.min()}, Max: {gray_matrix1.max()}")  # Min e Max livelli di grigio

    #rumore termico 
    gray_matrix_result = gray_matrix1 - gray_matrix2
    arr = gray_matrix_result.flatten()
    count.append(arr)
    #count_max.append(max(arr))
    sigma_result = np.std(arr)
    sigma = sigma_result/np.sqrt(2)
    y_data.append(sigma)
    #print(f"la deviazione standard è {sigma}")
   
print(len(matrice_grigi))
matrice_grigi = np.array(matrice_grigi)
#print(matrice_grigi, np.shape(matrice_grigi))
media = []
for i in range(np.shape(matrice_grigi)[0]):
    mat = matrice_grigi[i].flatten()
    mean = np.mean(mat)
    media.append(mean)

def linear(x, a, b):
    return a*x + b
popt,_ = curve_fit(linear, x_data, media)
x = np.linspace(min(x_data), max(x_data), 1000)
plt.plot(x, linear(x, *popt), color="r", label="fit lineare")
plt.errorbar(x_data, media, color= "blue",fmt=".", capsize=2)
plt.xlabel("tempi esposizione [s]")
plt.ylabel("media [adm]")
plt.title("Andamento della corrente di buio")
plt.grid(which="both", linestyle="--",alpha=0.5)
plt.show()

count= np.array(count)
print(f"lunghezza count {np.shape(count)}")
"""
f, axes = plt.subplots(2, 6, figsize=(12,6))
for i in range(12):
    r,c = i//6, i%6
    min_val, max_val = -np.max(count[i,:]), np.max(count[i,:])
    bins = np.arange(min_val, max_val + 1, 1)
    axes[r][c].hist(count[i, :], bins=bins, histtype='step', edgecolor='blue',alpha=0.7)
    axes[r][c].set_xticks(np.linspace(min_val, max_val, num=5, dtype=int))
    axes[r][c].set_yticks(np.linspace(0, np.max(np.histogram(count[i, :], bins=bins)[0]), num=5, dtype=int))
    #axes[r][c].legend(title=f't{x_data[i]} [s]')
    # Etichette degli assi solo una volta
    if r == 1:  # Solo nella riga inferiore
        axes[r, c].set_xlabel("Livelli di grigio")
    if c == 0:  # Solo nella colonna sinistra
        axes[r, c].set_ylabel("Conteggi")
    axes[r,c].grid(which="both", linestyle="--",alpha=0.5)
    plt.tight_layout()

plt.show()

print(f"sigma {y_data}")
plt.scatter(x_data, y_data, color="blue")
plt.xlabel("tempi di esposizione [s]")
plt.ylabel("sigma [adm]")
plt.grid(which="both", linestyle="--",alpha=0.5)
plt.show()
"""
