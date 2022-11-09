"""
TITRE : Librairie de fonction pour l'optimisation différentiable
AUTEUR : Maxime GOSSELIN
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import time



#%% mesh grid

def grid(x,y,pas):
    X = np.arange(-x, x+pas, pas)
    Y = np.arange(-y, y+pas, pas)
    X, Y = np.meshgrid(X, Y)
    return X, Y

#%% Surface représentative

def surface(X,Y,pas,g,name):
    """
    Surface représentative de la fonction de R2 -> R
    X, Y : meshgrid (utiliser fct grid)
    pas : pas pour le grid (fct grid)
    g : fonction représenté (g(x,y))
    name : str g(x)=... format latex
    """
    surface_1 = plt.figure(1)
    ax = plt.axes(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, g(X,Y), cmap=cm.inferno, linewidth=0, antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(-4, 5)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    surface_1.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(f"Surface de la fonction ${name}$")

    plt.show()

#%% Carte de niveau

def carte(X,Y,pas,g,x_lim,y_lim, l_point,name, pt=True):
    """
    Carte des niveaux de la fonction de R2 -> R
    X, Y : meshgrid (utiliser fct grid)
    pas : pas pour le grid (fct grid)
    g : fonction représenté (g(x,y))
    x_lim,y_lim : param x,y de fct grid
    l_point : dict des points critiques {'nom_pt' : (xp,yp)}
    name : str g(x)=... format latex
    pt : activer l'affichage des points critiques sur le graphe
    """    
    carte_1 = plt.figure(2)
    plt.contour(X, Y, g(X,Y), 50)
    if pt == True:
        for i,j in l_point.items():
            plt.scatter(j[0],j[1], label=i+" "+str(j))
    plt.xlim(-x_lim,x_lim)
    plt.ylim(-y_lim,y_lim)
    plt.colorbar()
    plt.legend(fontsize=8)
    plt.title(f"Carte des niveaux de la fonction ${name}$")
    plt.grid()
    plt.show()


#%% méthodes de descente

def rechercheDuPas(x,d,p1, gf, Hf, tolR, Nitermax):
    j = 1
    p = 10*p1
    while(abs(p1 - p) > tolR and j < Nitermax):
        phi1 = d.T@gf(x+p*d)
        #print("phi1 = ",phi1)
        phi2 = d.T@Hf(x+p*d)@d
        #print("phi2 = ",phi2)
        p1 = p
        p = p1 - (phi1/phi2)
        j += 1
    return p

def pasfixe(gf, Hf, x0, eps, Nitermax, pas):
    t = time.time()
    Niter = 0
    d = -gf(x0)
    x=x0
    list_x = [x[0]]
    list_y = [x[1]]
    while(np.linalg.norm(d) > eps) and (Niter<Nitermax):
        x = x + pas*d
        list_x.append(x[0])
        list_y.append(x[1])
        d = -gf(x)
        Niter += 1
    t = time.time() - t
    t = round(t,4)
    return x, Niter, t, list_x, list_y




def pasoptimal(gf, Hf, x0, eps, Nitermax, pas):
    t = time.time()
    Niter = 0
    d = -gf(x0)
    x=x0
    list_x = [x[0]]
    list_y = [x[1]]
    while((np.linalg.norm(d) > eps) and (Niter<Nitermax)):
        rho = rechercheDuPas(x,d,10**-4, gf, Hf, 10**-6, 10**5)
        x = x + rho*d
        list_x.append(x[0])
        list_y.append(x[1])
        d = -gf(x)
        Niter += 1
    t = time.time() - t
    t = round(t,4)
    return x, Niter, t, list_x, list_y



def gradpre(gf, Hf, x0, tol, Nitermax, pas):
    t = time.time()
    Niter = 0
    D = np.diag(np.diag(Hf(x0)))
    x=x0
    list_x = [x[0]]
    list_y = [x[1]]
    while(np.linalg.norm(gf(x)) > tol and Niter < Nitermax):
        d = np.linalg.solve(D,-gf(x))
        rho = rechercheDuPas(x,d,10**-4, gf, Hf, 10**-6, 10**5)
        x = x + rho*d
        list_x.append(x[0])
        list_y.append(x[1])
        D = np.diag(np.diag(Hf(x)))
        Niter += 1
    t = time.time() - t
    t = round(t,4)
    return x, Niter, t, list_x, list_y    


def newton(gf, Hf, x0, tol, Nitermax, pas):
    t = time.time()
    Niter = 0
    D = Hf(x0)
    x=x0
    list_x = [x[0]]
    list_y = [x[1]]
    while(np.linalg.norm(gf(x)) > tol and Niter < Nitermax):
        d = np.linalg.solve(D,-gf(x))
        rho = rechercheDuPas(x,d,10**-4, gf, Hf, 10**-6, 10**5)
        x = x + rho*d
        list_x.append(x[0])
        list_y.append(x[1])
        D = Hf(x)
        Niter += 1
    t = time.time() - t
    t = round(t,4)
    return x, Niter, t, list_x, list_y


def BFGS(gf, Hf, x0, tol, Nitermax, pas):
    t = time.time()
    Niter = 0
    D = np.eye(2)
    x=x0
    list_x = [x[0]]
    list_y = [x[1]]
    while(np.linalg.norm(gf(x)) > tol and Niter < Nitermax):
        d = np.linalg.solve(D,-gf(x))
        rho = rechercheDuPas(x,d,10**-4, gf, Hf, 10**-6, 10**5)
        x1 = x
        x = x + rho*d
        list_x.append(x[0])
        list_y.append(x[1])
        y = np.mat(gf(x) - gf(x1))
        s = np.mat(rho*d)
        D = D + ((y.T@y)/(y@s.T))-((D@s.T@s@D)/(s@D@s.T))
        Niter += 1
    t = time.time() - t
    t = round(t,4)
    return x, Niter, t, list_x, list_y
