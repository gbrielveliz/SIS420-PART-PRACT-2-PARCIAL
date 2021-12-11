import os

# Calculo cientifico y vectorial para python
import numpy as np

# Librerias para graficar
from matplotlib import pyplot
# Modulo de optimizacion en scipy
from scipy import optimize


#data = np.loadtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA2','DATASET.txt'), delimiter=',')
#data=np.genfromtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA2','DATASET.txt'), delimiter=',',dtype=None)
edad = np.loadtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA2','DATASETT.txt'), delimiter=',', usecols=[0])
clas_trabajo = np.loadtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA2','DATASETT.txt'), delimiter=',',dtype=str, usecols=[1])              
fnlwgt=np.loadtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA2','DATASETT.txt'), delimiter=',', usecols=[2])
educacion=np.loadtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA2','DATASETT.txt'), delimiter=',',dtype=str, usecols=[3])
num_educacion=np.loadtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA2','DATASETT.txt'), delimiter=',', usecols=[4])
estado_civil=np.loadtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA2','DATASETT.txt'), delimiter=',',dtype=str, usecols=[5])
ocupacion =np.loadtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA2','DATASETT.txt'), delimiter=',',dtype=str, usecols=[6])
parentesco=np.loadtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA2','DATASETT.txt'), delimiter=',',dtype=str, usecols=[7])
raza=np.loadtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA2','DATASETT.txt'), delimiter=',',dtype=str, usecols=[8])
sexo=np.loadtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA2','DATASETT.txt'), delimiter=',',dtype=str, usecols=[9])
plusvalia=np.loadtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA2','DATASETT.txt'), delimiter=',', usecols=[10])
perdida_capital=np.loadtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA2','DATASETT.txt'), delimiter=',', usecols=[11])
horas_semana=np.loadtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA2','DATASETT.txt'), delimiter=',', usecols=[12])
pais_origen=np.loadtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA2','DATASETT.txt'), delimiter=',',dtype=str, usecols=[13]) 
ganancias=np.loadtxt(os.path.join( r'C:\Users\Gabo\Desktop\Nueva carpeta\PREGUNTA2','DATASETT.txt'), delimiter=',',dtype=str, usecols=[14]) 
#print(pais_origen)

#t=['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov','State-gov', 'Without-pay', 'Never-worked','?']
trabajos=['Never-worked','Without-pay','State-gov','Local-gov','?','Federal-gov','Self-emp-inc','Self-emp-not-inc','Private']
#e=['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
educaciones=['Preschool', '1st-4th','5th-6th','7th-8th','9th','10th','11th','12th','Assoc-voc','Assoc-acdm','HS-grad','Prof-school','Some-college','Bachelors','Masters','Doctorate']
#ec=['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
estados_civiles=['Married-civ-spouse','Married-spouse-absent', 'Married-AF-spouse','Divorced', 'Widowed','Separated','Never-married']
#oc=['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
# 'Priv-house-serv', 'Protective-serv', 'Armed-Forces','?']
ocupaciones=['Tech-support','Craft-repair','Handlers-cleaners','Farming-fishing','?','Other-service', 'Transport-moving','Sales','Machine-op-inspct','Adm-clerical','Priv-house-serv','Protective-serv','Armed-Forces','Exec-managerial','Prof-specialty',]
#p=['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
parentescos=['Own-child','Husband','Wife','Not-in-family','Other-relative','Unmarried']
#r=['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
razas=['Amer-Indian-Eskimo','Other','Black','Asian-Pac-Islander','White']
#s=['Female', 'Male']
sexos=['Female', 'Male']
#pa=['United-States', 'Cambodia','England','Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines',
# 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 
# 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands','?']
paises=['Vietnam','Cuba','Haiti','Cambodia','Puerto-Rico', 'Laos','Jamaica','India','Honduras','Trinadad&Tobago','Iran','El-Salvador','Guatemala','Nicaragua','Dominican-Republic','Yugoslavia','Thailand','Peru','Columbia','Ecuador','Philippines','Taiwan','Hong','Hungary','?','Mexico','South','Scotland','China','Greece','Holand-Netherlands','Portugal','Ireland','Outlying-US(Guam-USVI-etc)','France','Poland','Italy', 'Japan','England','Canada','Germany','United-States']
Gananciass=['<=50K','>50K']
for i in range(0, clas_trabajo.size):
    for j in range(0,len(trabajos)):
        if clas_trabajo[i]==trabajos[j]:
            clas_trabajo[i]=j+1           
clas_trabajo= [int(x) for x in clas_trabajo]

for i in range(0, educacion.size):
    for j in range(0,len(educaciones)):
        if educacion[i]==educaciones[j]:
            educacion[i]=j+1
educacion= [int(x) for x in educacion]

for i in range(0, estado_civil.size):
    for j in range(0,len(estados_civiles)):
        if estado_civil[i]==estados_civiles[j]:
            estado_civil[i]=j+1
estado_civil= [int(x) for x in estado_civil]

for i in range(0, ocupacion.size):
    for j in range(0,len(ocupaciones)):
        if ocupacion[i]==ocupaciones[j]:
            ocupacion[i]=j+1
ocupacion= [int(x) for x in ocupacion]

for i in range(0, parentesco.size):
    for j in range(0,len(parentescos)):
        if parentesco[i]==parentescos[j]:
            parentesco[i]=j+1
parentesco= [int(x) for x in parentesco]

for i in range(0, raza.size):
    for j in range(0,len(razas)):
        if raza[i]==razas[j]:
            raza[i]=j+1
raza= [int(x) for x in raza]

for i in range(0, sexo.size):
    for j in range(0,len(sexos)):
        if sexo[i]==sexos[j]:
            sexo[i]=j+1           
sexo= [int(x) for x in sexo]

for i in range(0, pais_origen.size):
    for j in range(0,len(paises)):
        if pais_origen[i]==paises[j]:
            pais_origen[i]=j+1           
pais_origen= [int(x) for x in pais_origen]

for i in range(0, ganancias.size):
    for j in range(0,len(Gananciass)):
        if ganancias[i]==Gananciass[j]:
            ganancias[i]=j
ganancias=[int(x) for x in ganancias]            

y=np.array(ganancias)
m=y.size
edad =edad.reshape(len(edad),1)
clas_trabajo=np.array(clas_trabajo)
clas_trabajo=clas_trabajo.reshape(len(clas_trabajo),1)
fnlwgt =fnlwgt.reshape(len(fnlwgt),1)
educacion=np.array(educacion)
educacion =educacion.reshape(len(educacion),1)
num_educacion =num_educacion.reshape(len(num_educacion),1)
estado_civil=np.array(estado_civil)
estado_civil =estado_civil.reshape(len(estado_civil),1)
ocupacion=np.array(ocupacion)
ocupacion =ocupacion.reshape(len(ocupacion),1)
parentesco=np.array(parentesco)
parentesco= parentesco.reshape(len(parentesco),1)
raza=np.array(raza)
raza =raza.reshape(len(raza),1)
sexo=np.array(sexo)
sexo =sexo.reshape(len(sexo),1)
plusvalia =plusvalia.reshape(len(plusvalia),1)
perdida_capital =perdida_capital.reshape(len(perdida_capital),1)
horas_semana =horas_semana.reshape(len(horas_semana),1)
pais_origen=np.array(pais_origen)
pais_origen=pais_origen.reshape(len(pais_origen),1)

X = np.concatenate([edad,clas_trabajo,fnlwgt,educacion,num_educacion,estado_civil,ocupacion,parentesco,raza,sexo,plusvalia,perdida_capital,horas_semana,pais_origen],axis=1)

#print(X)
#print(y)
print('m = ',m)
print('n = ',X.shape[1])

def  featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu) / sigma
    
    return X_norm, mu, sigma

X_nor,mu,sigma=featureNormalize(X)
X=np.concatenate([np.ones((m, 1)),X_nor],axis=1)

def calcularSigmoide(z):
    # Calcula la sigmoide de una entrada z
    # convierte la intrada a un arreglo numpy
    z = np.array(z)
  
    g = np.zeros(z.shape)

    g = 1.0 / (1.0 + np.exp(-z))

    return g

def calcularCosto(theta, X, y):
    # Inicializar algunos valores utiles
    m = y.size  # numero de ejemplos de entrenamiento

    J = 0
    h = calcularSigmoide(X.dot(theta.T))
    J = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h)))
    
    return J

def descensoGradiente(theta, X, y, alpha, num_iters):
    # Inicializa algunos valores
    m = y.shape[0] # numero de ejemplos de entrenamiento
    
    # realiza una copia de theta, el cual será acutalizada por el descenso por el gradiente
    theta = theta.copy()
    J_history = []
    
    for i in range(num_iters):
        h = calcularSigmoide(X.dot(theta.T))
        theta = theta - (alpha / m) * (h - y).dot(X)
       
        J_history.append(calcularCosto(theta, X, y))
    return theta, J_history


alpha = 0.003
num_iters =30000

# inicializa theta y ejecuta el descenso por el gradiente
theta = np.zeros(15)
theta, J_history = descensoGradiente(theta, X, y, alpha, num_iters)
print('Theta encontrada por descenso gradiente: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f} , {:.4f}'.format(*theta))

print("costo =", calcularCosto(theta,X,y))

pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Numero de iteraciones')
pyplot.ylabel('Costo J')
pyplot.show()

def predicion(theta, X):
    
    p = calcularSigmoide(X.dot(theta.T))
    for x in range(0,len(p)):
        if p[x]>0.5:
            p[x]=1
        else:
            p[x]=0
    return p

y_prec=predicion(theta,X)
print(y[0:10])
print(y_prec[0:10])
print('Precision del conjuto de entrenamiento: {:.2f}%'.format(np.mean(y_prec == y) * 100))




