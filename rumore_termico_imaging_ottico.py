#rumore_termico_imaging_ottico
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 

# Percorso della cartella
cartella = r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\0C"

# Ottieni tutti i file nella cartella e ordina (se serve)
file_list = sorted(os.listdir(cartella))

y_data = [] #sigma delle immagini
x_data = [0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100.] # tempi di esposizione
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

    print(f"Dimensione: {gray_matrix1.shape}")  # Altezza x Larghezza
    print(f"type dell'array {type(gray_matrix1)}")
    print(f"Min: {gray_matrix1.min()}, Max: {gray_matrix1.max()}")  # Min e Max livelli di grigio

    #rumore termico 
    gray_matrix_result = gray_matrix1 - gray_matrix2
    arr = gray_matrix_result.flatten()
    sigma_result = np.std(arr)
    sigma = sigma_result/np.sqrt(2)
    y_data.append(sigma)
    print(f"la deviazione standard è {sigma}")
    #istogramma
    min_val, max_val = -max(arr), max(arr)
    bins = np.arange(min_val, max_val + 1, 1)
    plt.hist(arr, bins=bins, histtype='step', edgecolor='blue',alpha=0.7)
    plt.xlabel("valori dei pixel")
    plt.ylabel("conteggi")
    plt.show()
    plt.grid()
    #plot della mappa 
    #plt.imshow(img1, cmap="gray")
    #plt.axis("off")
    #plt.show()

f, axes = plt.subplots(2, 6, figsize=(16,9))
for i in range(12):
    r,c = i//6, i%6
    axes[r][c].hist(arr, bins=bins, histtype='step', edgecolor='blue',alpha=0.7)
    axes[r][c].set_xticks([])
    axes[r][c].set_yticks([])
    
plt.tight_layout()

print(f"sigma {y_data}")
plt.scatter(x_data, y_data, color="blue")
plt.xlabel("tempi di esposizione [s]")
plt.ylabel("sigma [adm]")
#plt.grid()
plt.show()
