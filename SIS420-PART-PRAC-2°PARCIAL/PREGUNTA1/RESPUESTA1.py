import os

# Calculo cientifico y vectorial para python
import numpy as np

# Librerias para graficar
from matplotlib import pyplot
# Modulo de optimizacion en scipy
from scipy import optimize




data = np.loadtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA1','DATASET2.txt'), delimiter=',')
#print(data)
X=data[:,0:7]
y=data[:,7]
m=y.size
#print(X)
#print(y)

def featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu) / sigma
    
    return X_norm, mu, sigma

X_norm,mu,sigma=featureNormalize(X)
#print(' dfdsfsd')
#print(X_norm)
X = np.concatenate([np.ones((m, 1)),X_norm], axis=1)
#X = np.concatenate([np.ones((m, 1)),X], axis=1)
def computeCost(X, y, theta):
    # inicializa algunos valores importantes
    m = y.size  # numero de ejemplos de entrenamiento
    
    J = 0
    #h = np.dot(X, theta)
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
        # Inicializa algunos valores importantes
    m = y.shape[0]  # numero de ejemplos de entrenamiento
   
    # hace una copia de theta, para evitar cambiar la matriz original, 
    # ya que las matrices numpy se pasan por referencia a las funciones

    theta = theta.copy()
    
    J_history = [] # Lista que se utiliza para almacenar el costo en cada iteración
    
    for i in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        # save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))
    
    return theta, J_history


theta = np.zeros(8)
# configuraciones para el descenso por el gradiente
iterations = 15000
alpha = 0.003

theta, J_history = gradientDescent(X ,y, theta, alpha, iterations)
print('Theta encontrada por descenso gradiente: {:.4f}, {:.4f}, {:.4f}'.format(*theta))


print('costo calculado con theta',theta,' = ', computeCost(X,y,theta))
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Numero de iteraciones')
pyplot.ylabel('Costo J')
#pyplot.show()

print('PREDECIR')
#link de la informacion del auto que se va a predecir
#https://www.facebook.com/marketplace/item/1398795753856600/?ref=search&referral_code=marketplace_search&referral_story_type=post&tracking=browse_serp%3A5155f871-9801-48d0-8a5b-27c3468defbe
datos=[1,12,2019,1,5,2,1,2]
#link de la informacion del auto que se va a predecir
#https://www.facebook.com/marketplace/item/327495799197189/?ref=search&referral_code=marketplace_search&referral_story_type=post&tracking=browse_serp%3A5155f871-9801-48d0-8a5b-27c3468defbe
datos2=[1,12,2022,1,5,2,1,1]
#https://www.facebook.com/marketplace/item/661533374823140/?ref=search&referral_code=marketplace_search&referral_story_type=post&tracking=browse_serp%3A5155f871-9801-48d0-8a5b-27c3468defbe
datos3=[1,12,1991,1,4,2,1,2]
d=datos3.copy()
da=datos2.copy()
dat=datos.copy()
datos[1:8]=(datos[1:8]-mu)/sigma
datos2[1:8]=(datos2[1:8]-mu)/sigma
datos3[1:8]=(datos3[1:8]-mu)/sigma
print('el precio para los datos de este auto 1 ',dat,' es = ',np.dot(datos,theta),'$')
print('el precio para los datos de este auto 2 ',da,' es = ',np.dot(datos2,theta),'$')
print('el precio para los datos de este auto 3 ',d,' es = ',np.dot(datos3,theta),'$')
print('ECUACION DE LA NORMAL')
thetaa = np.zeros(8)
def normalEqn(X, y):
  
    thetaa = np.zeros(X.shape[1])
    
    thetaa = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    
    return thetaa

thetaa=normalEqn(X,y)
print('Theta encontrada por descenso gradiente: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(*thetaa))
J=computeCost(X,y,thetaa)
print('Con theta = : {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(*thetaa),'\n Costo calculado = %.2f' % J)
print('PREDECIR')
print('el precio para los datos de este auto ',dat,' es = ',np.dot(datos,thetaa),'$')
print('el precio para los datos de este auto 2 ',da,' es = ',np.dot(datos2,thetaa),'$')
print('el precio para los datos de este auto 3 ',d,' es = ',np.dot(datos3,thetaa),'$')