import numpy as np


def dcm_correction(dcm_img):
    '''
    A function to correct DICOM files.
    '''
    x = dcm_img.pixel_array + 1000
    px_mode = 4096
    x[x >= px_mode] = x[x >= px_mode] - px_mode
    dcm_img.PixelData = x.tobytes()
    dcm_img.RescaleIntercept = -1000


def window_image(dcm, window_center, window_width):
    '''
    A function to perform systemic windowing on DICOM files.
    '''
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0):
        dcm_correction(dcm)

    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)

    return img