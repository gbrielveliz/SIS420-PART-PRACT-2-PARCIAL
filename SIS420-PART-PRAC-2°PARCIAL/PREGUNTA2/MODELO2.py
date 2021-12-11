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




#FUNCION DE ACTIVACION PARA LA CAPA OCULTA
def relu(x):
  return np.maximum(0, x)

def reluPrime(x):
  return x > 0


#FUNCIONES DE ACTIVACION 
def linear(x):
    return x

def sigmoid(x):
  return 1.0/ (1.0 + np.exp(-x))
  

def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=-1,keepdims=True)


#FUNCIONES DE COSTO O PERDIDA

# Mean Square Error -> usada para regresión (con activación lineal)
def mse(y, y_hat):
    return np.mean((y_hat - y.reshape(y_hat.shape))**2)

# Binary Cross Entropy -> usada para clasificación binaria (con sigmoid)
def bce(y, y_hat):
    #return - np.mean(y.reshape(y_hat.shape)*np.log(y_hat) - (1 - y.reshape(y_hat.shape))*np.log(1 - y_hat))
    return (1 / m) * np.sum(-y.dot(np.log(y_hat)) - (1 - y).dot(np.log(1 - y_hat)))

# Cross Entropy (aplica softmax + cross entropy de manera estable) -> usada para clasificación multiclase
def crossentropy(y, y_hat):
    logits = y_hat[np.arange(len(y_hat)),y]
    entropy = - logits + np.log(np.sum(np.exp(y_hat),axis=-1))
    return entropy.mean()



#DERIVADAS DE LAS FUNCIONES
def grad_mse(y, y_hat):
    return y_hat - y.reshape(y_hat.shape)

def grad_bce(y, y_hat):
    return y_hat - y.reshape(y_hat.shape)

def grad_crossentropy(y, y_hat):
    answers = np.zeros_like(y_hat)
    answers[np.arange(len(y_hat)),y] = 1    
    return (- answers + softmax(y_hat)) / y_hat.shape[0]



# clase base MLP 
class MLP():
  def __init__(self, D_in, H, D_out, loss, grad_loss, activation):
    # pesos de la capa 1
    self.w1, self.b1 = np.random.normal(loc=0.0,
                                  scale=np.sqrt(2/(D_in+H)),
                                  size=(D_in, H)), np.zeros(H)
    # pesos de la capa 2
    self.w2, self.b2 = np.random.normal(loc=0.0,
                                  scale=np.sqrt(2/(H+D_out)),
                                  size=(H, D_out)), np.zeros(D_out)
    self.ws = []
    # función de pérdida y derivada
    self.loss = loss
    self.grad_loss = grad_loss
    # función de activación
    self.activation = activation

  def __call__(self, x):
    # salida de la capa 1
    self.h_pre = np.dot(x, self.w1) + self.b1
    self.h = relu(self.h_pre)
    # salida del MLP
    y_hat = np.dot(self.h, self.w2) + self.b2 
    return self.activation(y_hat)
    
  def fit(self, X, Y, epochs = 100, lr = 0.001, batch_size=None, verbose=True, log_each=10):
    batch_size = len(X) if batch_size == None else batch_size
    batches = len(X) // batch_size
    l = []
    for e in range(1,epochs+1):     
        # Mini-Batch Gradient Descent
        _l = []
        for b in range(batches):
            # batch de datos
            x = X[b*batch_size:(b+1)*batch_size]
            y = Y[b*batch_size:(b+1)*batch_size] 
            # salida del perceptrón
            y_pred = self(x) 
            # función de pérdida
            loss = self.loss(y, y_pred)
            _l.append(loss)        
            # Backprop 
            dldy = self.grad_loss(y, y_pred) 
            grad_w2 = np.dot(self.h.T, dldy)
            grad_b2 = dldy.mean(axis=0)
            dldh = np.dot(dldy, self.w2.T)*reluPrime(self.h_pre)      
            grad_w1 = np.dot(x.T, dldh)
            grad_b1 = dldh.mean(axis=0)
            # Update (GD)
            self.w1 = self.w1 - lr * grad_w1
            self.b1 = self.b1 - lr * grad_b1
            self.w2 = self.w2 - lr * grad_w2
            self.b2 = self.b2 - lr * grad_b2
        l.append(np.mean(_l))
        # guardamos pesos intermedios para visualización
        self.ws.append((
            self.w1.copy(),
            self.b1.copy(),
            self.w2.copy(),
            self.b2.copy()
        ))
        if verbose and not e % log_each:
            print(f'Epoch: {e}/{epochs}, Loss: {np.mean(l):.5f}')

  def predict(self, ws, x):
    w1, b1, w2, b2 = ws
    h = relu(np.dot(x, w1) + b1)
    y_hat = np.dot(h, w2) + b2
    return self.activation(y_hat)

# MLP para regresión
class MLPRegression(MLP):
    def __init__(self, D_in, H, D_out):
        super().__init__(D_in, H, D_out, mse, grad_mse, linear)

# MLP para clasificación binaria
class MLPBinaryClassification(MLP):
    def __init__(self, D_in, H, D_out):
        super().__init__(D_in, H, D_out, bce, grad_bce, sigmoid)

# MLP para clasificación multiclase
class MLPClassification(MLP):
    def __init__(self, D_in, H, D_out):
        super().__init__(D_in, H, D_out, crossentropy, grad_crossentropy, linear)


def  featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu) / sigma
    
    return X_norm, mu, sigma

X_norm,mu,sigma=featureNormalize(X)
X=np.array(X_norm)
mlp=MLPBinaryClassification(D_in=14, H=4, D_out=1)
epochs, lr = 100, 0.003
mlp.fit(X,y,epochs,lr, batch_size=10, log_each=10)

def predicion(X):

    h = relu(np.dot(X, mlp.w1) + mlp.b1)
    y_hat = np.dot(h, mlp.w2) + mlp.b2
    p= mlp.activation(y_hat)
    for x in range(0,len(p)):
        if p[x]>0.5:
            p[x]=1
        else:
            p[x]=0
    return p

y_prec=predicion(X)
print(y[0:10])
print(y_prec[0:10])
print('Precision del conjuto de entrenamiento: {:.2f}%'.format(np.mean(y_prec == y) * 100))