# rumore_di_lettura_imaging_ottico

# 1) voglio unire le prime due immagini della cartella 0C, 10C, 20C, 30C, 40C e salvare salvare 5 sigma, 
#     punti che servono per fare un grafico a tempo fissato =0.02s
# 2) faccio poi la stessa cosa per terza e quarta immagine, poi quinta e sesta, fino alla 24esima
# 3) alla fine mi troverò con 12 array da 5 punti, e possiamo plottare il grafico sigma-temperatura, vedremo 12 funzioni, una per tempo 

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 

times = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100.]) # seconds
Temp = np.array([0, -10, -20, -30, -40]) # degrees Celsius

# Percorso della cartella
cartella = "C:/Users/user/Desktop/Programming/es7_Imaging_Ottico/rumore/"
T = ["0C", "10C", "20C", "30C", "40C"]
#name = "/misura00"
#name0 = "/misura000"

y_data = np.zeros((5, 12))

for i in range(5):
    # cicle for the (i+1)-th temperature 
    print(f"{i}-esimo ciclo a T -{T[i]} gradi")
    path = os.path.join(cartella, T[i])
    
    for j in range(1, 24, 2):
        print(f"calcolo della sigma a -{T[i]} e tempo {times[int((j-1)/2)]}")
        if (j<10):
            path1 = os.path.join(path, f'misura000{j}.TIF')
        else:
            path1 = os.path.join(path, f'misura00{j}.TIF')
        if (j<9):
            path2 = os.path.join(path, f'misura000{j+1}.TIF')
        else: 
            path2 = os.path.join(path, f'misura00{j+1}.TIF')
    
        # Leggi le immagini
        img1 = cv2.imread(path1, cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(path2, cv2.IMREAD_UNCHANGED)
        # Ottieni la matrice dei livelli di grigio
        gray_matrix1 = np.array(img1, dtype=float)
        gray_matrix2 = np.array(img2, dtype=float)
    
        #print(f"Dimensione: {gray_matrix1.shape}")  # Altezza x Larghezza
        #print(f"type dell'array {type(gray_matrix1)}")
        #print(f"Min: {gray_matrix1.min()}, Max: {gray_matrix1.max()}")  # Min e Max livelli di grigio
    
        #rumore termico 
        gray_matrix_result = gray_matrix1 - gray_matrix2
        arr = gray_matrix_result.flatten()
        sigma_result = np.std(arr)
        sigma = sigma_result/np.sqrt(2)
        
        y_data[i][int((j-1)/2)] = sigma

    print(f"Sigma a temperatura -{T[i]}: ", y_data[i][:], "\n per i t=[.02, .05, .1, .2, .5, 1, 2, 5, 10, 20, 50, 100] secondi")

print("-----------------")
print( y_data.shape )
print(y_data)
print("-----------------")

plt.figure()
for k in range(12):
    plt.scatter(Temp, y_data[:, k], label=f't={times[k]} s')
plt.xlabel(r"Temperatura camera [$^o$ C]", size=15)
plt.ylabel("sigma [adim.]", size=15)
plt.grid()
plt.legend(loc='best', fontsize=15)
plt.show()

