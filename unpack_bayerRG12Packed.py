# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:20:22 2024

@author: skovsen
"""
#reference: https://www.1stvision.com/cameras/IDS/IDS-manuals/en/basics-raw-bayer-pixel-formats.html

# the raw file appears to have 2 bytes per pixel, indicating that it's not packed..

# remember to install the following libraries to speed up tiff image load: tifffile and imagecodecs
import numpy as np
import imageio
import time
from pidng.defs import Orientation, PhotometricInterpretation, CFAPattern, CalibrationIlluminant, DNGVersion, PreviewColorSpace
from pidng.core import RAW2DNG, DNGTags, Tag


from colour_demosaicing import (
    ROOT_RESOURCES_EXAMPLES,
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)

import cv2
#raw_array = np.load('07-15-2024 Baseline Image 2.RAW', allow_pickle=True)
#raw = rawpy.imread('07-15-2024 Baseline Image 2.RAW')
import glob
from pathlib import Path
import ntpath

im_height = 9528
im_width = 13376

#image_paths = glob.glob('aperture_test\\2024-08-21\\*.RAW')
# image_paths = glob.glob('D:\\post.doc\\unpack\\svcam_sample\\NC_2025-02-03\\raw\\*.RAW')
image_paths = Path("data/longterm_images/semifield-upload/NC_2025-02-03_test/raw").glob("*.RAW")
image_paths = [path for path in image_paths]
temp_dir = Path("data/longterm_images/semifield-upload/NC_2025-02-03_test/demosaiced_png")
temp_dir.mkdir(parents=True, exist_ok=True)

#from pidng.core import RAW2DNG, DNGTags, Tag
#from pidng.defs import *
import numpy as np
import struct


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 65535] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = ((np.arange(0, 65536.0) / 65535.0) ** inv_gamma) * 65535.0
    # Ensure table is 16-bit
    table = table.astype(np.uint16)

    # Now just index into this with the intensities to get the output
    return table[image]


def get_ccm_from_image(img_with_colorchecker):
    gamma = 1.0
    #img_with_colorchecker = imageio.imread(im_path)
    img_with_colorchecker = cv2.cvtColor(img_with_colorchecker, cv2.COLOR_BayerBG2BGR)
    img_with_colorchecker = (img_with_colorchecker / 256).astype(np.uint8)
    color_correction_ccm_model = cv2.ccm.COLORCHECKER_Macbeth
    color_correction_color_space = cv2.ccm.COLOR_SPACE_sRGB
    color_correction_matrix_type = cv2.ccm.CCM_3x3

    ccm_detector = cv2.mcc.CCheckerDetector.create()

    result = ccm_detector.process(img_with_colorchecker, cv2.ccm.COLORCHECKER_Macbeth, 1)
    
    if result == True:
        color_checkers = ccm_detector.getListColorChecker()
            
        print(f"Use first checker found")
        
        assert len(color_checkers) == 1, f'Unexpected checker count, detected ({len(color_checkers)}), expected (1)'
        checker = color_checkers[0]
                
        chart_sRGB = checker.getChartsRGB()
        src = chart_sRGB[:, 1].copy().reshape(24, 1, 3)
        src /= 255.0

        print(f"Calculating CCM for gamma {gamma}")
        
        ccm_model = cv2.ccm.ColorCorrectionModel(src, cv2.ccm.COLORCHECKER_Macbeth)
        ccm_model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
        ccm_model.setCCM_TYPE(cv2.ccm.CCM_3x3)
        #ccm_model.setInitialMethod(cv2.ccm.INITIAL_METHOD_WHITE_BALANCE)
        ccm_model.setInitialMethod(cv2.ccm.INITIAL_METHOD_LEAST_SQUARE)
        #ccm_model.setDistance(cv2.ccm.DISTANCE_CIE2000)
        ccm_model.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
        #ccm_model.setLinear(cv2.ccm.LINEARIZATION_IDENTITY)
        ccm_model.setLinearGamma(gamma)
        ccm_model.setLinearDegree(3)
        ccm_model.setSaturatedThreshold(0, 0.98)
        ccm_model.run()
        
        ccm_model.getCCM()
        del img_with_colorchecker
        return ccm_model
    else:
        del img_with_colorchecker
        return None        
        



if True:
    for im_path in image_paths:
        print(im_path)
        # image specs
        width = im_width
        height = im_height
        bpp= 16
        numPixels = width*height
        #im_path = 'aperture_test\\2024-08-21\\img_f5.6_20.RAW'
        rawImage = np.fromfile(im_path, dtype=np.uint16).astype(np.uint16)
        
        rawImage = rawImage.reshape((height, width))#/2).astype(np.uint16)
        
        #cv2.imwrite(im_filename[:-4] + '_raw.png', rawImage,  [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        current_ccm = get_ccm_from_image(rawImage)
        if(current_ccm is not None):
            color_matrix = current_ccm.getCCM()
            print(color_matrix)
            rawImage_rgb = cv2.cvtColor(rawImage, cv2.COLOR_BayerBG2RGB)
            rawImageRGBFloat = rawImage_rgb.astype(np.float64)/65535.0
            #img_float = current_ccm.infer(rawImageRGBFloat) * 65535.0
            #img_float = np.dot(rawImageRGBFloat, color_matrix)*65535.0
            img_float = np.dot(rawImageRGBFloat, color_matrix)*20000.0
            #img_float = np.dot(rawImageRGBFloat, color_matrix)*40000.0
            print(np.max(np.max(img_float)))
            img_float[img_float < 0] = 0
            img_float[img_float > 65535] = 65535.0
            colour_image_gamma_adjusted = img_float.astype(np.uint16)
            colour_image_gamma_adjusted = (adjust_gamma(colour_image_gamma_adjusted, gamma=2.0))
            save_path = f"{temp_dir}/{im_path.stem}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(colour_image_gamma_adjusted, cv2.COLOR_RGB2BGR),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        
        # uncalibrated color matrix, just for demo. 
        #ccm1 = [[19549, 10000], [-7877, 10000], [-2582, 10000],	
        #    [-5724, 10000], [10121, 10000], [1917, 10000],
        #    [-1267, 10000], [ -110, 10000], [ 6621, 10000]]
        #ccm1 = [[10000, 10000], [0, 10000], [0, 10000],	
        #    [0, 10000], [10000, 10000], [0, 10000],
        #    [0, 10000], [ 0, 10000], [ 10000, 10000]]
        
        
            # set DNG tags.
            t = DNGTags()
            t.set(Tag.ImageWidth, width)
            t.set(Tag.ImageLength, height)
            t.set(Tag.TileWidth, width)
            t.set(Tag.TileLength, height)
            t.set(Tag.Orientation, Orientation.Horizontal)
            t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
            t.set(Tag.SamplesPerPixel, 1)
            t.set(Tag.BitsPerSample, bpp)
            t.set(Tag.CFARepeatPatternDim, [2,2])
            t.set(Tag.CFAPattern, CFAPattern.RGGB)
            t.set(Tag.BlackLevel, 0)#(4096 >> (16 - bpp)))
            t.set(Tag.WhiteLevel, 65535)#((1 << bpp) -1) )
            ccm = current_ccm.getCCM()
            ccm = ccm.astype(np.float32)
            t.set(Tag.ColorMatrix1, ccm)
            t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.D65)
            t.set(Tag.AsShotNeutral, [[1,1],[1,1],[1,1]])
            t.set(Tag.BaselineExposure, [[-150,100]])
            t.set(Tag.Make, "SVS")
            t.set(Tag.Model, "Camera Model")
            t.set(Tag.DNGVersion, DNGVersion.V1_4)
            t.set(Tag.DNGBackwardVersion, DNGVersion.V1_2)
            t.set(Tag.PreviewColorSpace, PreviewColorSpace.sRGB)
    
            # save to dng file.
            r = RAW2DNG()
            r.options(t, path="")
            r.convert(rawImage, filename=im_path.stem + ".dng")




if False:
    for im_path in image_paths:
        im_filename = ntpath.basename(im_path)
        nparray = np.fromfile(im_path, dtype=np.uint16).astype(np.uint16)
        #print(np.max(np.max(nparray)))
        org_reshaped = nparray.reshape((im_height, im_width))
        #org_reshaped = (org_reshaped & 0b1111011111111111)
        np.min(np.min(org_reshaped))
        np.max(np.max(org_reshaped))
        #cv2.imwrite('aperture_test\\2024-08-21\\output\\' + im_filename[:-3] + 'png', org_reshaped,  [cv2.IMWRITE_PNG_COMPRESSION, 0])
        image_data = (org_reshaped).astype(np.float32)/65535.
        
        colour_image_gamma_adjusted = demosaicing_CFA_Bayer_Malvar2004(image_data, "RGGB")
        colour_image_gamma_adjusted[colour_image_gamma_adjusted<0] = 0
        colour_image_gamma_adjusted[colour_image_gamma_adjusted>1] = 1
            
        test = cv2.cvtColor(colour_image_gamma_adjusted.astype(np.float32), cv2.COLOR_RGB2BGR)
        cv2.imwrite('aperture_test\\2024-08-21\\output\\' + im_filename[:-3] + 'png', (test*65535).astype(np.uint16),  [cv2.IMWRITE_PNG_COMPRESSION, 0])



if False:
    for im_path in image_paths:
        im_filename = ntpath.basename(im_path)
        nparray = np.fromfile(im_path, dtype=np.uint16).astype(np.uint16)
        #print(np.max(np.max(nparray)))
        org_reshaped = nparray.reshape((im_height, im_width))
        #org_reshaped = (org_reshaped & 0b1111011111111111)
        np.min(np.min(org_reshaped))
        np.max(np.max(org_reshaped))
        #cv2.imwrite('aperture_test\\2024-08-21\\output\\' + im_filename[:-3] + 'png', org_reshaped,  [cv2.IMWRITE_PNG_COMPRESSION, 0])
        image_data = (org_reshaped).astype(np.float32)/65535.
        
        colour_image_gamma_adjusted = demosaicing_CFA_Bayer_Malvar2004(image_data, "RGGB")
        colour_image_gamma_adjusted[colour_image_gamma_adjusted<0] = 0
        colour_image_gamma_adjusted[colour_image_gamma_adjusted>1] = 1
            
        test = cv2.cvtColor(colour_image_gamma_adjusted.astype(np.float32), cv2.COLOR_RGB2BGR)
        cv2.imwrite('aperture_test\\2024-08-21\\output\\' + im_filename[:-3] + 'png', (test*65535).astype(np.uint16),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

if False:
    nparray = imageio.imread('packed_855495028830.tiff').flatten().astype(np.uint16)
    
    nparray_12bit = np.empty(im_width*im_height, dtype=np.uint16)
    
    
    
    #nparray_12bit = np.empty(im_width*im_height, dtype=np.uint16)
    #nparray_12bit[::2] = (nparray[::3] << 4) | (nparray[1::3] & 0b00001111)
    #nparray_12bit[1::2] = nparray[1::3] + ((nparray[2::3] & 0b11110000) << 4)
    #nparray_unpacked = nparray_12bit.reshape((im_height, im_width)) << 4
    
    
    index = (9528-4900)*13376 + 8050
    print((nparray[index+5] & 0b11110000)/16)
    
    
    start = time.time()
    nparray_12bit = np.empty(im_width*im_height, dtype=np.uint16)
    nparray_12bit[::2] = (nparray[::3] << 4) + (nparray[1::3] & 0b00001111)
    nparray_12bit[1::2] = nparray[1::3] + ((nparray[2::3] & 0b11110000) << 4)
    nparray_unpacked = nparray_12bit.reshape((im_height, im_width)) << 4
    end = time.time()
    print(end - start)
    
    np.min(np.min((nparray[::3] << 4)))
    
    print(np.mean(np.mean(nparray_unpacked[::2] & 0b1000000000000000)/32768))
    print(np.mean(np.mean(nparray_unpacked[1::2] & 0b1000000000000000)/32768))
    print(np.mean(np.mean(nparray_unpacked[0::2] & 0b0100000000000000)/16384))
    print(np.mean(np.mean(nparray_unpacked[1::2] & 0b0100000000000000)/16384))
    print(np.mean(np.mean(nparray_unpacked[0::2] & 0b0010000000000000)/8192))
    print(np.mean(np.mean(nparray_unpacked[1::2] & 0b0010000000000000)/8192))
    print(np.mean(np.mean(nparray_unpacked[0::2] & 0b0001000000000000)/4096))
    print(np.mean(np.mean(nparray_unpacked[1::2] & 0b0001000000000000)/4096))
    print(np.mean(np.mean(nparray_unpacked[0::2] & 0b0000100000000000)/2048))
    print(np.mean(np.mean(nparray_unpacked[1::2] & 0b0000100000000000)/2048))
    print(np.mean(np.mean(nparray_unpacked[0::2] & 0b0000010000000000)/1024))
    print(np.mean(np.mean(nparray_unpacked[1::2] & 0b0000010000000000)/1024))
    print(np.mean(np.mean(nparray_unpacked[0::2] & 0b0000001000000000)/512))
    print(np.mean(np.mean(nparray_unpacked[1::2] & 0b0000001000000000)/512))
    print(np.mean(np.mean(nparray_unpacked[0::2] & 0b0000000100000000)/256))
    print(np.mean(np.mean(nparray_unpacked[1::2] & 0b0000000100000000)/256))
    print(np.mean(np.mean(nparray_unpacked[0::2] & 0b0000000010000000)/128))
    print(np.mean(np.mean(nparray_unpacked[1::2] & 0b0000000010000000)/128))
    print(np.mean(np.mean(nparray_unpacked[0::2] & 0b0000000001000000)/64))
    print(np.mean(np.mean(nparray_unpacked[1::2] & 0b0000000001000000)/64))
    
    nparray_unpacked = nparray_unpacked & 0b1111000000000000
    encoding = 'mono16'
    
    cv2.imwrite('test_out.png', nparray_unpacked,  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    image_data = (nparray_unpacked).astype(np.float32)/65535.
    
    colour_image_gamma_adjusted = demosaicing_CFA_Bayer_Malvar2004(image_data, "RGGB")
    colour_image_gamma_adjusted[colour_image_gamma_adjusted<0] = 0
    colour_image_gamma_adjusted[colour_image_gamma_adjusted>1] = 1
        
    test = cv2.cvtColor(colour_image_gamma_adjusted.astype(np.float32), cv2.COLOR_RGB2BGR)
    cv2.imwrite('test_out_demosaic.png', (test*65535).astype(np.uint16),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

