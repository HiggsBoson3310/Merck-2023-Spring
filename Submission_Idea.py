import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from tqdm.notebook import tqdm
from sklearn.base import clone

X_train, y_train = make_regression(n_samples=5000,n_features=10,n_informative=5,n_targets=1,noise=0.0)
fstar = np.percentile(y_train,90)
NEP = 10
Bsize = 100

import numdifftools as dif
class SingleLayer():

    def __init__(self,dim_in: int, dim_l: int, dim_o: int,th: float,alpha:float=1e-4,gamma:float=0.2):
        if(len(th)==dim_in*dim_l+dim_l+dim_l*dim_o+dim_o):
            self.alpha = alpha
            self.gamma = gamma
            self.dim_in = dim_in
            self.dim_l = dim_l
            self.dim_o = dim_o
            self.th = th
            self.Mil = th[:dim_in*dim_l].reshape(dim_l,dim_in)
            self.Mlo = th[dim_l*dim_in+dim_l:dim_l*dim_in+dim_l+dim_l*dim_o].reshape(dim_o,dim_l)
            self.bl = np.reshape(th[dim_in*dim_l:dim_in*dim_l+dim_l],(-1,1))
            self.bo = np.reshape(th[dim_l*dim_in+dim_l+dim_l*dim_o:],(-1,1))
            self.loss = 1
            self.loss_curve = []
        else:
            raise ValueError("Incorrect dimensions of single layer")
    
    def update_params(self,th_new):
        if(len(th_new)==self.dim_in*self.dim_l+self.dim_l+self.dim_l*self.dim_o+self.dim_o):
            self.th = th_new
            self.Mil = th_new[:self.dim_in*self.dim_l].reshape(self.dim_l,self.dim_in)
            self.Mlo = th_new[self.dim_l*self.dim_in+self.dim_l:
                self.dim_l*self.dim_in+self.dim_l+self.dim_l*self.dim_o].reshape(self.dim_o,self.dim_l)
            self.bl = np.reshape(th_new[self.dim_in*self.dim_l:self.dim_in*self.dim_l+self.dim_l],(-1,1))
            self.bo = np.reshape(th_new[self.dim_l*self.dim_in+self.dim_l+self.dim_l*self.dim_o:],(-1,1))
        else:
            raise ValueError("Incorrect dimensions when updating params")
        
    def predict(self,x):
        if(len(x)==self.dim_in):
            l = np.maximum(self.Mil@x.reshape(-1,1)+self.bl,np.zeros(self.dim_l).reshape(-1,1))
            return np.maximum(self.Mlo@l+self.bo,np.zeros(self.dim_o).reshape(-1,1))
        else:
            raise ValueError("Incorrect input dimension")
    
    def predict_th(self,th_new,x):
        Mil = th_new[:self.dim_in*self.dim_l].reshape(self.dim_l,self.dim_in)
        Mlo = th_new[self.dim_l*self.dim_in+self.dim_l:
            self.dim_l*self.dim_in+self.dim_l+self.dim_l*self.dim_o].reshape(self.dim_o,self.dim_l)
        bl = np.reshape(th_new[self.dim_in*self.dim_l:self.dim_in*self.dim_l+self.dim_l],(-1,1))
        bo = np.reshape(th_new[self.dim_l*self.dim_in+self.dim_l+self.dim_l*self.dim_o:],(-1,1))
        l = np.maximum(Mil@x.reshape(-1,1)+bl,np.zeros(self.dim_l).reshape(-1,1))
        #return np.maximum(Mlo@l+bo,np.zeros(self.dim_o).reshape(-1,1))
        return Mlo@l+bo
        
    def y(self,r,x):
        return r+self.gamma*max(self.predict(x))
        
    def Loss(self,th,b):
        l = 0
        s,a,r,sp = np.array(b[:self.dim_in]),b[self.dim_in],\
                    b[self.dim_in+1],np.array(b[self.dim_in+2:])
        l += (self.y(r,sp)-self.predict_th(th,s)[int(a)])**2
        return l[0]
    
    def SGD_step(self, batch):
        for b in batch:
            f = lambda yy: self.Loss(yy,b)
            grad = dif.Gradient(f)
            th = self.th - self.alpha * grad(self.th)
            self.update_params(th)
        loss = 0 
        for b in batch:
            loss += self.Loss(self.th,b)
        self.loss = loss/len(batch)
        self.loss_curve.append(self.loss)
        return None

# Greedy sampling function on feature space (just distance to nearest neighboor)
def GS_x(x,x_list):
    distances = np.zeros_like(x_list)

    for ii,xx in enumerate(x_list):
        distances[ii] = np.sqrt(np.sum((x-xx)**2))
    
    return np.min(distances)

def GS_y(x,y_init,model):
    Ypreds = model.predict([x])
    distances =  np.zeros_like(y_init)
    for ii, xx in enumerate(y_init):
        distances[ii] = np.abs(Ypreds-xx)
    return np.min(distances)

def state(model,cand,pool_xs,pool_vals):
    gsx = GS_x(cand,pool_xs)
    gsy = GS_y(cand,pool_vals,model)
    return np.array([*cand,gsx,gsy])

learner = SingleLayer(12,128,2,np.random.random(size=(12*128+128+128*2+2)),alpha=1e-2)

# Let us define a try with 10 episodes of Q-learning training.
# A budget of 100 points per episode
# A size of 10 for the optimization minibatch
B = 15
EPISODES = 10
MINI_BATCH_SIZE = 10
init_size = 100
# Define a transition memory
Mm = []
# A reward memory
Rm = []
# Curve losses
CL = []

#Initial model
rndf_model =  RandomForestRegressor(100, n_jobs=-2,random_state=158)

for _ in tqdm(range(EPISODES),desc='Learner Episodes'):
    l_cont = 0
    
    M = []
    R = []
    L = []
    indices = list(range(5000))
    np.random.shuffle(indices)
    X_init = X_train[indices[:init_size]]
    y_init = y_train[indices[:init_size]]

    rnd_forest = clone(rndf_model)
    rnd_forest.fit(X_init,y_init)
       
    X_cands = X_train[indices[init_size:]]
    y_cands = y_train[indices[init_size:]]
    val_test = rnd_forest.predict(X_init)
    for kk in tqdm(range(len(X_cands)-1),desc='Candiate Evaluation'):
        c, v = X_cands[kk], y_cands[kk]
        cp1, vp1 = X_cands[kk+1], y_cands[kk+1]
        si = state(rnd_forest,c,X_init,y_init)
        ai = np.argmax(learner.predict(si))
        y_init_o = y_init.copy()
        if(ai==1):
            l_cont+=1
            X_init = np.append(X_init,[c],0)
            y_init = np.append(y_init,[v],0)
            rnd_forest.fit(X_init,y_init)
        y_pred = rnd_forest.predict(X_init)
        R.append(mean_squared_error(y_init,y_pred)-mean_squared_error(y_init_o,val_test))
        val_test = y_pred.copy()
        if(l_cont==B):
            M.append([*si,ai,R[-1],None])
            break
        else:
            sip1 = state(rnd_forest,cp1,X_init,y_init)
            M.append([*si,ai,R[-1],*sip1])
        if(len(M)>=MINI_BATCH_SIZE):
            indices = np.random.choice(len(M),MINI_BATCH_SIZE,replace=False)
            mini_batch = [M[ii] for ii in indices]
        else:
            mini_batch = M.copy()

        #print(gpr_x.kernel_)
        L.append(mean_squared_error(y_init,y_pred))
        np.random.shuffle(mini_batch)
        learner.SGD_step(mini_batch)
    
    Mm.append(M)
    Rm.append(R)
    CL.append(L)