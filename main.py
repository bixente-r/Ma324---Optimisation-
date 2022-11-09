from lib_opti import *
import numpy as np

#%% définition de la fonction (x :  vecteur et x,y :  points)
fct = "g(x) = (x^2_1 + x_2 - 11)^2 + (x + x^2_2 - 7)^2"

def g(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def g1(x,y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

#%% définition du gradient et de la Hessienne

def grad_g(x):
    return np.array([4*x[0]*(x[0]**2+x[1]-11)+2*(x[0]+x[1]**2-7),2*(x[0]**2+x[1]-11)+4*x[1]*(x[0] + x[1]**2-7)])


def H_g(x):
    H = np.array([[4*(x[0]**2 + x[1] - 11)+8*x[0]**2 + 2, 4*x[0]+4*x[1]], [4*x[0] + 4*x[1], 4*(x[0] + x[1]**2 - 7) + 8*x[1]**2 + 2]])
    return H


#%% définition des points critiques (Chercher sur Wolframalpha)
d_pt_crit = {"P1":(-3.77931,-3.28319),"P2":(-3.07303,-0.081353),"P3":(-2.80512,3.13131),"P4":(-0.270845,-0.923039),
             "P5":(-0.127961,-1.95371),"P6":(0.0866775,2.88425),"P7":(3,2),"P8":(3.38515,0.0738519),
             "P9":(3.58443,-1.84813)}


#%% définition des paramètres

# paramètres : main affichage fct
x = 5         # x_grid 
y = 5         # y_grid
pas = 0.01    # pas_grid
surf = False
carte = False

# paramètres : main etude pt critique
pas_fixe = 10**-5
Nitermax = 10e6
epsilon = 10**-10
pas2 = 0.08    # ajustement de l'affichage du gradient de descente
dx = 0.1
dy = 0.1
pselect = d_pt_crit["P1"]
d_metho = {"Méthode du gradient à pas fixe ": pasfixe,"Méthode du gradient à pas optimal ": pasoptimal,
           "Méthode du gradient préconditionné à pas optimal ": gradpre,
           "Méthode de Newton ": newton,"Méthode quasi-Newton BFGS ": BFGS}
metho_on = [True, True, True, True, True]

#%% main affichage fct

X,Y = grid(x,y,pas)
if surf == True:
    surface(X,Y,pas,g1,fct)
if carte == True:
    carte(X,Y,pas,g1,x,y,d_pt_crit,fct)

#%% main etude pt critique


x0 = pselect - np.array([dx,dy])
xmin, xmax = (x0[0] + pselect[0])/2 - pas, (x0[0] + pselect[0])/2 + pas
ymin, ymax = (x0[1] + pselect[1])/2 - pas, (x0[1] + pselect[1])/2 + pas



print(f"\nx0 = {x0}\n")



carte_1 = plt.figure(2)
plt.title(f'Recherche du point critique {pselect}')
plt.xlabel("X")
plt.ylabel("Y")
i = 0
for k,f in d_metho.items():
   
    if metho_on[i] == True:
        sol,nbiter,tempscalcul,list_x,list_y = f(grad_g,H_g,x0,epsilon,Nitermax,pas_fixe)
        print(f"Méthode : {k}\n - Solution : {sol}\n - Nombre d'itération : {nbiter}\n - Temps de calcul : {tempscalcul} sec\n")
    i += 1
    plt.plot(list_x, list_y, marker=",", label=f"{k}", linewidth=0.5)
plt.contour(X, Y, g1(X,Y), 50)
plt.scatter(pselect[0],pselect[1], color='red', label='point critique')
plt.scatter(x0[0],x0[1], color='green', label='point de départ')
plt.ylim(ymin,ymax)
plt.xlim(xmin,xmax)
plt.colorbar()
plt.legend()
plt.grid()
plt.show()
"""
carte_1.savefig("C:/Users/Public/Documents/IPSA/carte_gp4.svg")"""