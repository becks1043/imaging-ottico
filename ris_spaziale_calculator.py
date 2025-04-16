#ris_spaziale_calculator

 
#programma per calcolare la risoluzione [line/mm]
def resolution(group, element):
    return 2**(group + ((element-1)/6))
print("Calcoliamo la risoluzione grazie alla tabella USAF 1951 Target")
while True:
    comando = input("Vuoi calcolare la risoluzione? Digita 'no' per uscire, digita 'si' per farlo\n" )
    if comando.lower() == "no":
        print("Bye-Bye :)")
        break
    if comando.lower() == "si":
        print("ok :)")
        gruppo = int(input("inesrisci il gruppo\n"))
        elemento = int(input("inesrisci l'elemento\n"))
        print(f"La risoluzione è {resolution(gruppo, elemento)}")