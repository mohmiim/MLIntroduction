'''
Created on Jan 18, 2020

@author: miim
'''
import imageio
import os

folder = "output/animation/all"

def creatImage(folder):
    filenames = [f for f in os.listdir(folder) if  f.endswith('.png')]
    
    with imageio.get_writer('flowers80-2/anim/movie-all-9.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(folder + filename)
            writer.append_data(image)
            
def createImage2(folder):
    images = []
    for file_name in os.listdir(folder):
        if file_name.endswith('.png'):
            file_path = os.path.join(folder, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave('flowers80-2/anim/movie-9-all.gif', images, fps=3)
    
createImage2(folder)