import os
import numpy as np
import pandas as pd 
import pydicom


def window_processing(dcm):
    '''
    A function to generate brain, subdural and soft-tissue windows
    from a DICOM file & concatenate them into a single three-channel
    image
    '''
    #Windowing
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)

    #Standardising
    brain_img = (brain_img - 0)/80
    subdural_img = (subdural_img - (-20))/200
    soft_img = (soft_img - (-150))/380
    #Concatenating
    input_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)

    return input_img
