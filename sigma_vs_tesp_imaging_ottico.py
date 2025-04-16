#sigma_vs_temperatura_imaging_ottico
#FORSE aggiungi una interpolazione ai grafici, l'andamento non ci aspettiamo essere esponenziale
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 

y_data = [] # sigma delle immagini
x_data = [0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100.] # tempi di esposizione
temperature = [ -10., -20., -30., -40 , 0] # temperature della cella peltier

# Percorso della cartella
cartella = r"C:\Users\emili\Desktop\magistrale\LabMed\imaging_ottico\rumore\termico"

for dir in sorted(os.listdir(cartella)):
    temp = os.path.join(cartella,dir)
    print(f"sono nella cartella {temp}")

    # Ottieni tutti i file nella cartella e ordina (se serve)
    file_list = sorted(os.listdir(temp))

    # Leggi i file a due a due
    for file1, file2 in zip(file_list[::2], file_list[1::2]):
        path1 = os.path.join(temp, file1)
        path2 = os.path.join(temp, file2)
        print(f"Lettura di {path1} e {path2}")
        
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
        #count_max.append(max(arr))
        sigma_result = np.std(arr)
        sigma = sigma_result/np.sqrt(2)
        y_data.append(sigma)
        #print(f"la deviazione standard è {sigma}")

print(f"sigma {y_data}\n{len(y_data)}")

colors = plt.cm.viridis(np.linspace(0, 1, len(temperature))) 

for i in range(len(temperature)):
    plt.scatter(x_data, y_data[i*12:(i+1)*12], color=colors[i], label=f"T {i*(-10)} $C^0$")
    
plt.legend()
plt.xlabel("tempi di esposizione [s]")
plt.ylabel("sigma [adm]")
plt.xscale("log")
plt.grid(which="both", linestyle="--",alpha=0.5)
plt.show()