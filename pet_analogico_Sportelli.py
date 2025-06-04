#pet_analogico_Sportelli
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

file_path = r"C:\Users\emili\Desktop\magistrale\LabMed\PETanalogico\histogram_data.txt"
bins, count = np.loadtxt(file_path, unpack = True)

def MGF(x, *params):
    result = np.zeros_like(x)
    for i in range(0, len(params), 3):
        a = params[i]
        mu = params[i +1]
        sigma = params[i+2]
        result += a * np.exp(-((x-mu)**2)/(2*(sigma**2)))
    return result
guess = [2.62e4, 171., 90., 2.2e4, 325., 200.,6.6e4, 512., 20.] #2.62e4, 160., 120,
popt, pcov = curve_fit( MGF, bins, count, p0= guess)
x = np.linspace(0, max(bins), 1000)
print(f"parametri del fit mu = {popt[7]}+-{np.sqrt(pcov[7,7])}\nsigma = {popt[8]}")
#plot
plt.plot(bins, count, color ="blue")
plt.plot(x, MGF(x, *popt), color = "red")
plt.legend(title = f"R = {np.round((2.35*popt[8]/popt[7])*100,2)}%")
plt.show()