'''
Created on Jun. 2, 2019

@author: mohammedmostafa
'''
from sklearn.metrics import confusion_matrix, classification_report

class ModelTest:
    def __init__(self,generator=None,model=None):
        self.generator = generator
        self.model = model
    
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
        