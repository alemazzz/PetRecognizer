import os
import numpy as np
import tensorflow as tf
import xtarfile as tarfile
import cv2
import re
import sys

IMG_WIDTH = 30
IMG_HEIGHT = 30

def load_data(data_dir):
    images, labels, i, last = list(), list(), 0, 'Abyssinian_'
    
    # Iterate for each file in the given directory and skip if the file is corrupted
    for file_name in os.listdir(os.path.join(data_dir)):
        try:
            image = cv2.resize(cv2.imread(os.path.join(data_dir, file_name)), (IMG_WIDTH, IMG_HEIGHT))
        except:
            continue
        
        # Check with r.expression if the race changes or not
        if matches := re.search(r'^(\D+).*$', file_name):
            if matches.groups()[0] != last:
                i += 1
            last = matches.groups()[0]
        
        # Append image and relative labels
        images.append(image)
        labels.append(i)
    
    return images, labels
        
print(load_data('./images'))