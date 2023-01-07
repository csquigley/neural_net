import numpy as np

#Squared Error cost function
class SquaredError: 
    def __init__(self):
        self.y_pred = 0
        self.y_true = 0
    def __name__(self):
        return "SquaredError" 
    def cost(self,y_pred,y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return (self.y_true-self.y_pred)**2
    def backward_calc(self):
        return np.array([2*(self.y_pred-self.y_true)]).astype(np.longdouble)

#Cross Entropy Cost Function    
class CrossEntropy:
    def __init__(self):
        self.y_preds = None
        self.y_trues = None
    def __name__(self):
        return "CrossEntropy"
    def cost(self,y_preds,y_trues):
#         self.y_preds = np.clip(y_preds.reshape(-1,),1.0e-1599,1.0)
        self.y_preds = y_preds.reshape(-1,)
        self.y_trues = y_trues.reshape(-1,)
        return -np.log(self.y_preds[self.y_trues.argmax()])

    def backward_calc(self):
#         r = -(1/self.y_preds[self.y_trues.argmax()])
#         r = self.y_trues * r
                             
#         return r
        #it is easier to just return the true values and use the easier derivative
        return self.y_trues
class BinaryCrossEntropy:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    def __name__(self):
        return 'BinaryCrossEntropy'
    def cost(self,y_pred,y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return -(self.y_true*log(self.y_pred) + (1-self.y_true)*log(1-self.y_pred))
    def backward_calc(self):
        c = -(self.y_true/self.y_pred) + ((1-self.y_true)/(1-self.y_pred))
        return c