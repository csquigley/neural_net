#import all necessary libraries, tools
import numpy as np
from math import e,log, sqrt
#sigmoid activation function
class Sigmoid:
    def __name__(self):
        return "Sigmoid"
    
    def forward_calc(self,z):
        return (1/(1+e**-z))
    
    def backward_calc(self,z):
        dzda = ((1-1/(1+e**-z))*(1/(1+e**-z)))
        return dzda

#ReLU activation function    
class ReLU:
    def __name__(self):
        return "ReLU"
    def forward_calc(self,z):  
        z = np.array([max(i,0) for i in z[0]]).reshape(1,-1).astype(np.longdouble)
        return z
    def backward_calc(self,z):
        z = np.array([1 if x > 0 else 0 for x in z[0]]).reshape(z.shape).astype(np.longdouble)
        return z
    
#LeakyReLU activation function
class LeakyReLU:
    def __name__(self):
        return "LeakyReLU"
    def forward_calc(self,z):
        z = np.array([x if x > 0 else (x * 0.1) for x in z[0]]).reshape(z.shape).astype(np.longdouble)
        return z
    def backward_calc(self,z):
        z = np.array([1 if x > 0 else 0.1 for x in z[0]]).reshape(z.shape).astype(np.longdouble)
        return z 
#No Activation Function         
class NoActivation:
    def __name__(self):
        return "NoActivation"
#SoftMax activation function
class SoftMax:

    def __init__(self):
        self.output = None
        self.z = None
    def __name__(self):
        return "SoftMax"
    def forward_calc(self,z):
        self.z = z
        z = z.reshape(-1,)
        z = z - z.max()
        self.output = (e**z/sum(e**z)).astype(np.longdouble)

        return self.output.astype(np.longdouble)
    def backward_calc(self,s):
# Alternatively...
#         d = np.zeros((len(self.output),(len(self.output))))
#         for j in range(0,len(self.output)):
#             for i in range(0,len(self.output)):
#                 if j == i:
#                     d[j][i] = self.output[j]*(1-self.output[i]) 
#                 else:
#                     d[j][i] = -self.output[j]*self.output[i]
#         d = (z@d).reshape(z.shape).astype(np.longdouble)
        d = self.output - s
        return d
 