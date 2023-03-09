import numpy as np
from math import e, log, sqrt
import json
from costs import SquaredError

class Network:
    
    def __init__(self,*layers):
        #initialize network object, the most important thing here is the layers
        self.layers = layers
        self.lr = 0.001
        self.cost = None
        self.X = None
        self.y = None
        self.batch_size = 10
        self.tested = 0
        
    def settings(self,lr=0.001,cost=SquaredError,batch_size=10):
        #this method exists to update parameters (such as the learning rate) while training
        self.lr = lr
        self.cost = cost()
        self.batch_size = batch_size
        
    def connect_data(self,X,y):
        #connect the data separately because it allows us to use different data to train the network later
        self.X = X
        self.y = y
        
    def train(self):
        costs = []
        #there is an error in here with your update weight function.
        #you aren't doing anything with the remainder. This is being carried over to
        #the next epoch. Because the derivative just sits idle until the next epoch starts. 
        #but each epoch the derivative needs to be reset. 
        for n,i in enumerate(self.X):
            #predict y for input x
            for l,layer in enumerate(self.layers):
                i = layer.forward_calc(i)

            #calculate cost
            cost = self.cost.cost(i,self.y[n])
            #append cost for statistics, measuring performance
            costs.append(cost)
            #calculate the derivative of the cost function for the given result
            d = self.cost.backward_calc().reshape(1,-1)

            #propagate the derivatives backward through the layers
            for i in range(len(self.layers)-1,-1,-1):
                d = self.layers[i].backward_calc(d)

            #update the weights when n = batch_size
            if (n + 1) % self.batch_size == 0:
                for layer in self.layers:
                    layer.update_w(lr=self.lr,batch_size=self.batch_size) 
        #calculate the average cost for the epoch
        costs = sum(costs)/len(costs)
        return costs
    
    def predict(self,X):
        #predict y without running backpropagation 
        preds = []
        for i in X:
            for layer in self.layers:
                if layer.__name__() == 'Dropout':
                    i = layer.forward_calc(i,training=False)
                else:
                    i = layer.forward_calc(i)

#             i = i[0]
            preds.append(i)
        preds = np.array(preds)
        return preds
    
    def save(self,file_path):
        #save the model and the weights as .json  
        #some precision is lost (not noticeable though)
        with open(file_path,'w') as f:
            data = {
                'architecture' : [{'type':layer.__name__(),'inpf': layer.inpf,'outpf':layer.outpf,'AF':layer.activation.__name__()} if layer.__name__() == 'Layer'
                                  else {'type':layer.__name__(),'p':layer.p} for layer in self.layers],
                'weights': [layer.W.astype(np.float32).tolist() if type(layer.W) != type(None) else None for layer in self.layers],
                'bias':[layer.b.astype(np.float32).tolist() if type(layer.b) != type(None) else None for layer in self.layers],
                'cost':self.cost.__name__()
            }
            data=json.dumps(data)
            f.write(data)
            f.close()
               
    def load(self,file_path):
        #load a model and its weights from .json
        with open(file_path) as f:
            data = json.loads(f.read())
            self.layers = [Layer(layer['inpf'],layer['outpf'],eval(layer['AF'])) if layer['type'] == 'Layer' else Dropout(p=layer['p']) for layer in data['architecture']]
            self.cost = eval(data['cost'])()
            for n,layer in enumerate(self.layers):
                if type(layer.W) == type(np.array([0])):
                    layer.W = np.array(data['weights'][n])
                    layer.b = np.array(data['bias'][n])
                else:
                    pass

    def info(self):
    #return information about the parameters, number of nodes, etc. 
        print("Network Architecture")
        for layer in self.layers:
            if layer.__name__() == "Layer":
                print(f"{(layer.__name__()+'  ')[:7]} - Inputs: {layer.inpf}, Outputs: {layer.outpf}, Activation: {layer.activation.__name__()}")
            else:
                print(f"{(layer.__name__()+'  ')[:7]} - p= ({layer.p})")
        if self.cost != None:
            print(f"Cost Function: {self.cost.__name__()}\n")
        else:
            print(f"No Cost Function Connected\n")
        print("Settings")
        print(f"lr={self.lr} batch_size={self.batch_size} optimizer=ADAM")