'''
Created on Jun. 2, 2019

@author: mohammedmostafa
'''
import matplotlib.pyplot as plt

class showImages :
    
    def show(self):
        x,y = self.generator.next()
        print("Showing {0} for {1}".format(len(x),y))
        for i in range(0,10):
            image = x[i]
            plt.imshow(image)
            plt.show()
       
    def __init__(self, generator):
        self.generator = generator

