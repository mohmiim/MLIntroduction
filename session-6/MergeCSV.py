'''
Created on Feb 2, 2020

@author: miim
'''

import os
import glob
import pandas as pd
os.chdir("D:/ml/weather")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
print(all_filenames)
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f,encoding= 'unicode_escape') for f in all_filenames ])
#export to csv
combined_csv.to_csv( "combined_csv.csv", index=False)