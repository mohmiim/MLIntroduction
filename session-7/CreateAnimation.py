'''
Created on Jan 18, 2020

@author: miim
'''
import imageio
import os

folder = "generatedFlowers/"
            
def createImage(folder):
    images = []
    for file_name in os.listdir(folder):
        if file_name.endswith('.png'):
            file_path = os.path.join(folder, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave('generatedFlowers/movie-all.gif', images, fps=3)
    
createImage(folder)