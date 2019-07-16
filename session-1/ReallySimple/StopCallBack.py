'''
Created on May 30, 2019

@author: mohammedmostaqfa
'''

from tensorflow.keras.callbacks import Callback

class myCallBacks(Callback):
    
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_acc')>=self.val_acc) :
            print("\n Reached 95% accuracy on epoch {0}, stopping training".format(epoch+1))
            self.model.stop_training = True
    
    def __init__(self, val_acc=0.9):
        self.val_acc = val_acc
