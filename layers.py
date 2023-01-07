import numpy as np
import activ.NoActivation
class Dropout:
    def __name__(self):
        return 'Dropout'
    def __init__(self,p):
        self.W = None
        self.b = None
        self.p = p
        self.drop = None
        #activation is always No Activation
    def forward_calc(self,X,training=True):
        if training:
            l = len(X)      
            a = np.arange(0,l)
            a = np.random.permutation(a)
            d = int((1 - self.p) * (l)) 
            self.drop = a[:d]
            X[(self.drop)] = X[(self.drop)] * 0 
        return X
    def backward_calc(self,z):
        z[(self.drop)] = z[self.drop] * 0
        return z 
    def update_w(self,lr=None,batch_size=None):
        pass
class Layer:
    def __name__(self):
        return "Layer"
    def __init__(self,inpf,outpf,activation=NoActivation):
        self.inpf = inpf
        self.outpf = outpf
        self.W = (np.random.randn(outpf,inpf)*1/(0.5*(inpf+outpf))).astype(np.longdouble)
#         self.W = (2*((self.W-self.W.min())/(self.W.max()-self.W.min()))-1)
#         lower, upper = -(1.0 /sqrt(outpf)),(1.0/sqrt(outpf))
#         self.W = (lower + np.random.randn(outpf,inpf)*(upper-lower)).astype(np.longdouble)
        self.b = np.ones((1,outpf))
        self.activation = activation()
        self.optim = None
        self.z = None
        self.X = None
        self.dCdW = np.zeros(self.W.shape,dtype=np.longdouble)
        self.dCdb = np.zeros((1,outpf),dtype=np.longdouble)
        
    def forward_calc(self,X):
        
        self.X = X
        self.z = sum((X * self.W).T) + self.b

        if self.activation.__name__() != "NoActivation":
            a = self.activation.forward_calc(self.z).astype(np.longdouble)
            return a.astype(np.longdouble)

        return self.z.astype(np.longdouble)
    
    def backward_calc(self,d):

        #if the layer has an activation function, first find the derivative of the activation function
        if self.activation.__name__() != "NoActivation" and self.activation.__name__() != 'SoftMax':
            d = (d * self.activation.backward_calc(self.z)).astype(np.longdouble)
        if self.activation.__name__() == "SoftMax":
            d = (self.activation.backward_calc(d)).astype(np.longdouble)
        #calculate the derivative of the weights and biases
        self.dCdW += (d.T * self.X)
        self.dCdb += (d * self.b)
        #calculate the derivative of X to send back to the previous layer
        #it isn't necessary to save dCdX because we don't change that parameter
        dCdX = sum(d.T * self.W)
        return dCdX.astype(np.longdouble)
            
    def update_w(self,lr=0.0001,batch_size=10,optimizer=Adam):
        if optimizer != None:
            self.optim = optimizer(lr=lr,beta1=0.9,beta2=.999)   
        #update the weights and the biases
        if self.optim.__name__() == "Adam":

            change = self.optim.adam((self.dCdW/batch_size))
            self.W -= change
        else:
            self.W -= (self.dCdW/batch_size * lr)

        self.b -= (self.dCdb/batch_size * lr)

        #reset the derivatives of the weights and the biases to zero
        self.dCdW = np.zeros(self.W.shape,dtype=np.longdouble)
        self.dCdb = np.zeros(self.b.shape,dtype=np.longdouble)
