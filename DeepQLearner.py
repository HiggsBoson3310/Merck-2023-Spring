import numpy as np
import numdifftools as dif
import tensorflow as tf

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
            self.bl = np.reshape(th[dim_in*dim_l:dim_in*dim_l+dim_l],(-1,1))
            self.Mlo = th[dim_l*dim_in+dim_l:dim_l*dim_in+dim_l+dim_l*dim_o].reshape(dim_o,dim_l)
            self.bo = np.reshape(th[dim_l*dim_in+dim_l+dim_l*dim_o:],(-1,1))
            self.loss = 1
            self.loss_curve = []
        else:
            raise ValueError("Incorrect dimensions of single layer")
    
    def update_params(self,th_new):
        if(len(th_new)==self.dim_in*self.dim_l+self.dim_l+self.dim_l*self.dim_o+self.dim_o):
            self.th = th_new
            self.Mil = th_new[:self.dim_in*self.dim_l].reshape(self.dim_l,self.dim_in)
            self.bl = np.reshape(th_new[self.dim_in*self.dim_l:self.dim_in*self.dim_l+self.dim_l],(-1,1))

            self.Mlo = th_new[self.dim_in*self.dim_l+self.dim_l:
                              self.dim_in*self.dim_l+self.dim_l+self.dim_l*self.dim_o].reshape(self.dim_o,self.dim_l)
            self.bo = np.reshape(th_new[self.dim_in*self.dim_l+self.dim_l+self.dim_l*self.dim_o:],(-1,1))
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
        return np.maximum(Mlo@l+bo,np.zeros(self.dim_o).reshape(-1,1))
        
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