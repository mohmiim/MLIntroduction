'''
Created on May 30, 2019

@author: mohammedmostaqfa
'''

from tensorflow.keras.callbacks import Callback

class myCallBacks(Callback):
    
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss')<=self.loss) :
            print("\n Reached {1} loss on epoch {0}, stopping training".format(epoch+1,self.loss))
            self.model.stop_training = True
    
    def __init__(self, loss=1E-4):
        self.loss = loss
