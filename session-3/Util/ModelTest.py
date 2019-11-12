'''
Created on Jun. 2, 2019

@author: mohammedmostaqfa
'''
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import plot_model

class ModelTest:
    def __init__(self,generator=None,model=None):
        self.generator = generator
        self.model = model
    
    def plotModel(self,model, imagePath="model.png", show_shapes=True,show_layer_names=False,expand_nested=False,dpi=96):
        plot_model(model,
                   to_file=imagePath,
                   show_shapes=show_shapes,
                   show_layer_names=show_layer_names,
                   rankdir=rankdir,
                   expand_nested=expand_nested,
                   dpi=dpi)
    
    def test(self):
        if (self.generator==None):
            print("No Generator is set")
            return
        if (self.model==None):
            print("No model is set")
            return
        
        filenames = self.generator.filenames
        nb_samples = len(filenames)
        predictions = self.model.predict_generator(self.generator,steps = nb_samples)
        row_index = predictions.argmax(axis=1)
        y_true = self.generator.classes
        target_names = self.generator.class_indices.keys()
        print(target_names)
        print(confusion_matrix(y_true, row_index))
        print('Classification Report')
        target_names = self.generator.class_indices.keys()
        print(classification_report(self.generator.classes, row_index, target_names=target_names))
        
