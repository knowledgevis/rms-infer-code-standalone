import json
import large_image
import large_image_source_dicom
import large_image_source_tiff
from pydicom import config

# this file is able to run the RMS segmentation model but incorrectly uses Max's 
# older code and doesn't create a correct DICOM outputfile.  This file is kept for 
# legacy. look at rms-seg-with-dicom-output.py for continued work. 
# 6/12/23


# define global variable that is set according to whether GPUs are discovered
USE_GPU = False

#-------------------------------------------


def infer_rhabdo(image_file,**kwargs):

    print(" input image filename = {}".format(image_file))

    # setup the GPU environment for pytorch
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    DEVICE = 'cuda'

    print('perform forward inferencing')

    subprocess = False
    if (subprocess):
        # declare a subprocess that does the GPU allocation to keep the GPU memory from leaking
        msg_queue = Queue()
        gpu_process = Process(target=start_inference, args=(msg_queue,image_file))
        gpu_process.start()
        predict_image = msg_queue.get()
        gpu_process.join()  
    else:
        predict_image = start_inference_mainthread(image_file)


    predict_bgr = cv2.cvtColor(predict_image,cv2.COLOR_RGB2BGR)
    print('output conversion and inferencing complete')

    

    # generate unique names for multiple runs.  Add extension so it is easier to use
    outname = image_file.split('.')[0]+'_predict.png'

    # write the output object using openCV  
    print('writing output')
    cv2.imwrite(outname,predict_bgr)
    print('writing completed')

    # new output of segmentation statistics in a string
    statistics = generateStatsString(predict_image)
    # generate unique names for multiple runs.  Add extension so it is easier to use
    statoutname = image_file.split(',')[0]+'_stats.json'
    open(statoutname,"w").write(statistics)

    # return the name of the output file
    return outname, statoutname


import sys
import random
#import argparse
import torch
import torch.nn as nn
import cv2

import os, glob
import numpy as np
from skimage.io import imread, imsave
from skimage import filters
from skimage.color import rgb2gray
import gc

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import albumentations as albu
import segmentation_models_pytorch as smp

ml = nn.Softmax(dim=1)


NE = 50
ST = 100
ER = 150
AR = 200
PRINT_FREQ = 20
BATCH_SIZE = 80

ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = None
DEVICE = 'cuda'

# the weights file is in the same directory, so make this path reflect that.  If this is 
# running in a docker container, then we should assume the weights are at the toplevel 
# directory instead

WEIGHT_PATH = './'

# these aren't used in the girder version, no files are directly written out 
# by the routines written by FNLCR (Hyun Jung)
WSI_PATH = '.'
PREDICTION_PATH = '.'

IMAGE_SIZE = 384
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
CHANNELS = 3
NUM_CLASSES = 5
CLASS_VALUES = [0, 50, 100, 150, 200]

BLUE = [0, 0, 255] # ARMS: 200
RED = [255, 0, 0] # ERMS: 150
GREEN = [0, 255, 0] # STROMA: 100
YELLOW = [255, 255, 0] # NECROSIS: 50
EPSILON = 1e-6

# what magnification should this pipeline run at
ANALYSIS_MAGNIFICATION = 10
THRESHOLD_MAGNIFICATION = 2.5
ASSUMED_SOURCE_MAGNIFICATION = 20.0

# what % interval we should print out progress so it can be snooped by the web interface
REPORTING_INTERVAL = 5

rot90 = albu.Rotate(limit=(90, 90), always_apply=True)
rotn90 = albu.Rotate(limit=(-90, -90), always_apply=True)

rot180 = albu.Rotate(limit=(180, 180), always_apply=True)
rotn180 = albu.Rotate(limit=(-180, -180), always_apply=True)

rot270 = albu.Rotate(limit=(270, 270), always_apply=True)
rotn270 = albu.Rotate(limit=(-270, -270), always_apply=True)

hflip = albu.HorizontalFlip(always_apply=True)
vflip = albu.VerticalFlip(always_apply=True)
tpose = albu.Transpose(always_apply=True)

pad = albu.PadIfNeeded(p=1.0, min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=0, value=(255, 255, 255), mask_value=0)

# supporting subroutines
#-----------------------------------------------------------------------------

def _generate_th(image_org):
    image = np.copy(image_org)

    org_height = image.shape[0]
    org_width = image.shape[1]

    # start with an all dark image
    otsu_seg = np.zeros((org_height//4, org_width//4), np.uint8)

    # resize the source image to 1/4 size
    aug = albu.Resize(p=1.0, height=org_height // 4, width=org_width // 4)
    augmented = aug(image=image)
    thumbnail = augmented['image']
    # make a gray 1/4 size image and pass to OTSU
    thumbnail_gray = rgb2gray(thumbnail)
    val = filters.threshold_otsu(thumbnail_gray)

    # now val is the threshold from OTSU, so we use this set everything in otsu_seg that is 
    # under the threshold (outside the ROI0 to be pure white background color  
    otsu_seg[thumbnail_gray <= val] = 255

    # multiply back to original size and return the upsampled segmentation mask
    aug = albu.Resize(p=1.0, height=org_height, width=org_width)
    augmented = aug(image=otsu_seg, mask=otsu_seg)
    otsu_seg = augmented['mask']

    print('Otsu segmentation finished')
    return otsu_seg

# redo the generation of the OTSU threshold but do it in a scalable way using large_image to 
# go tile by tile.  Keep track of the lowest value returned from OTSU on any tile and use this for the 
# threshold to generate an output image.

def _new_generate_otsu(image_file):
    # open image source on image
    # iterate through tiles
        # calculate OTSU for this tile and calculate the average threshold value (this will be from part
        # tissue and part background).
    # iterate through tiles to generate OTSU seg image
        # write out thresholded tiles (compare with average threshold value) to otsu_seg file
    # return otsu_seg filename
    pass


def _infer_batch(model, test_patch):
    # print('Test Patch Shape: ', test_patch.shape)
    with torch.no_grad():
        logits_all = model(test_patch[:, :, :, :])
        logits = logits_all[:, 0:NUM_CLASSES, :, :]
    prob_classes_int = ml(logits)
    prob_classes_all = prob_classes_int.cpu().numpy().transpose(0, 2, 3, 1)

    return prob_classes_all

def _augment(index, image):

    if index == 0:
        image= image

    if index == 1:
        augmented = rot90(image=image)
        image = augmented['image']

    if index ==2:
        augmented = rot180(image=image)
        image= augmented['image']

    if index == 3:
        augmented = rot270(image=image)
        image = augmented['image']

    if index == 4:
        augmented = vflip(image=image)
        image = augmented['image']

    if index == 5:
        augmented = hflip(image=image)
        image = augmented['image']

    if index == 6:
        augmented = tpose(image=image)
        image = augmented['image']

    return image
    
def _unaugment(index, image):

    if index == 0:
        image= image

    if index == 1:
        augmented = rotn90(image=image)
        image = augmented['image']

    if index ==2:
        augmented = rotn180(image=image)
        image= augmented['image']

    if index == 3:
        augmented = rotn270(image=image)
        image = augmented['image']

    if index == 4:
        augmented = vflip(image=image)
        image = augmented['image']

    if index == 5:
        augmented = hflip(image=image)
        image = augmented['image']

    if index == 6:
        augmented = tpose(image=image)
        image = augmented['image']

    return image

def _gray_to_color(input_probs):

    index_map = (np.argmax(input_probs, axis=-1)*50).astype('uint8')
    height = index_map.shape[0]
    width = index_map.shape[1]

    heatmap = np.zeros((height, width, 3), np.float32)

    # Background
    heatmap[index_map == 0, 0] = input_probs[:, :, 0][index_map == 0]
    heatmap[index_map == 0, 1] = input_probs[:, :, 0][index_map == 0]
    heatmap[index_map == 0, 2] = input_probs[:, :, 0][index_map == 0]

    # Necrosis
    heatmap[index_map==50, 0] = input_probs[:, :, 1][index_map==50]
    heatmap[index_map==50, 1] = input_probs[:, :, 1][index_map==50]
    heatmap[index_map==50, 2] = 0.

    # Stroma
    heatmap[index_map==100, 0] = 0.
    heatmap[index_map==100, 1] = input_probs[:, :, 2][index_map==100]
    heatmap[index_map==100, 2] = 0.

    # ERMS
    heatmap[index_map==150, 0] = input_probs[:, :, 3][index_map==150]
    heatmap[index_map==150, 1] = 0.
    heatmap[index_map==150, 2] = 0.

    # ARMS
    heatmap[index_map==200, 0] = 0.
    heatmap[index_map==200, 1] = 0.
    heatmap[index_map==200, 2] = input_probs[:, :, 4][index_map==200]

    heatmap[np.average(heatmap, axis=-1)==0, :] = 1.

    return heatmap


# return a string identifier of the basename of the current image file
def returnIdentifierFromImagePath(impath):
    # get the full name of the image
    file = os.path.basename(impath)
    # strip off the extension
    base = file.split('.')[0]
    return(base)


def isNotANumber(variable):
    # this try clause will work for integers and float values, since floats can be cast.  If the
    # variable is any other type (include None), the clause will cause an exception and we will return False
    try:
        tmp = int(variable)
        return False
    except:
        return True

#---------------- main inferencing routine ------------------
def _inference(model, image_path, BATCH_SIZE, num_classes, kernel, num_tta=1):

    model.eval()

    # open an access handler on the large image
    #source = large_image.getTileSource(image_path)
    config.enforce_valid_values = False
    #source = large_image_source_tiff.open(image_path)
    source = large_image_source_dicom.open(image_path)

    # print image metadata
    metadata = source.getMetadata()
    print(metadata)
    print('sizeX:', metadata['sizeX'], 'sizeY:', metadata['sizeY'], 'levels:', metadata['levels'])

    # figure out the size of the actual image and the size that this analysis
    # processing will run at.  The size calculations are made in two steps to make sure the
    # rescaled threshold image size and the analysis image size match without rounding error

    height_org = metadata['sizeY']
    width_org = metadata['sizeX']

    # if we are processing using a reconstructed TIF from VIPS, there will not be a magnification value.
    # So we will assume 20x as the native magnification, which matches the source data the
    # IVG  has provided.

    if isNotANumber(metadata['magnification']):
        print('warning: No magnfication value in source image. Assuming the source image is at ',
            ASSUMED_SOURCE_MAGNIFICATION,' magnification')
        metadata['magnification'] = ASSUMED_SOURCE_MAGNIFICATION
        assumedMagnification = True
    else:
        assumedMagnification = False

    # the theoretical adjustment for the magnification would be as below:
    # height_proc = int(height_org * (ANALYSIS_MAGNIFICATION/metadata['magnification']))
    # width_proc = int(width_org * (ANALYSIS_MAGNIFICATION/metadata['magnification']))

    height_proc = int(height_org * THRESHOLD_MAGNIFICATION/metadata['magnification'])*int(ANALYSIS_MAGNIFICATION/THRESHOLD_MAGNIFICATION)
    width_proc = int(width_org * THRESHOLD_MAGNIFICATION/metadata['magnification'])*int(ANALYSIS_MAGNIFICATION/THRESHOLD_MAGNIFICATION)
    print('analysis image size :',height_proc, width_proc)

    basename_string = os.path.splitext(os.path.basename(image_path))[0]
    print('Basename String: ', basename_string)

    # generate a binary mask for the image
    height_otsu = int(height_proc * THRESHOLD_MAGNIFICATION/ANALYSIS_MAGNIFICATION)
    width_otsu = int(width_proc * THRESHOLD_MAGNIFICATION / ANALYSIS_MAGNIFICATION)
    print('size of threshold mask:',height_otsu,width_otsu)
    # this will always generate a 10x region size, even if the source image has lower resolution
    myRegion = {'top': 0, 'left': 0, 'width': width_org, 'height': height_org}


    if assumedMagnification:
        # we have to manage the downsizing to the threshold magnification.
        threshold_source_image, mimetype = source.getRegion(format=large_image.tilesource.TILE_FORMAT_NUMPY,
                                                            region=myRegion,output={'maxWidth':width_otsu,'maxHeight':height_otsu})
        print('used maxOutput for threshold size')
    else:

        threshold_source_image, mimetype = source.getRegion(format=large_image.tilesource.TILE_FORMAT_NUMPY,
                                                        region=myRegion,
                                                        scale={'magnification': THRESHOLD_MAGNIFICATION})

    print('OTSU image')
    print(threshold_source_image.shape)

    # strip off any extra alpha channel
    threshold_source_image = threshold_source_image[:,:,0:3]
    print(threshold_source_image.shape)
    thumbnail_gray = rgb2gray(threshold_source_image)
    val = filters.threshold_otsu(thumbnail_gray)
    # create empty output for threshold
    otsu_seg = np.zeros((threshold_source_image.shape[0], threshold_source_image.shape[1]), np.uint8)
    # generate a mask=true image where the source pixels were darker than the
    # # threshold value (indicating tissue instead of bright background)
    otsu_seg[thumbnail_gray <= val] = 255
    # OTSU algo. was applied at reduced scale, so scale image back up
    aug = albu.Resize(p=1.0, height=height_proc, width=width_proc)
    augmented = aug(image=otsu_seg, mask=otsu_seg)
    otsu_org = augmented['mask'] // 255
    print('rescaled threshold shape is:', otsu_org.shape)
    #imsave('otsu.png', (augmented['mask'] .astype('uint8')))
    print('Otsu segmentation finished')

    #otsu_org = _generate_th(source,height_org,width_org) // 255


    # initialize the output probability map
    prob_map_seg_stack = np.zeros((height_proc, width_proc, num_classes), dtype=np.float32)

    for b in range(num_tta):

        height = height_proc
        width = width_proc

        PATCH_OFFSET = IMAGE_SIZE // 2
        SLIDE_OFFSET = IMAGE_SIZE // 2
        print('using', (PATCH_OFFSET//IMAGE_SIZE*100),'% patch overlap')

        # these are the counts in the x and y direction.  i.e. how many samples across the image.
        # the divident is slide_offset because this is how much the window is moved each time
        heights = (height + PATCH_OFFSET * 2 - IMAGE_SIZE) // SLIDE_OFFSET +1
        widths = (width + PATCH_OFFSET * 2 - IMAGE_SIZE) // SLIDE_OFFSET +1
        print('heights,widths:',heights,widths)

        heights_v2 = (height + PATCH_OFFSET * 2) // (SLIDE_OFFSET)
        widths_v2 = (width + PATCH_OFFSET * 2) // (SLIDE_OFFSET)
        print('heights_v2,widths_v2',heights_v2,widths_v2)

        # extend the size to allow for the whole actual image to be processed without actual
        # pixels being at a tile boundary.

        # doubled to *4 and *8 when changed 408 #409 to //4
        height_ext = SLIDE_OFFSET * heights + PATCH_OFFSET * 2
        width_ext = SLIDE_OFFSET * widths + PATCH_OFFSET * 4
        print('height_ext,width_ext:',height_ext,width_ext)

        org_slide_ext = np.ones((height_ext, width_ext, 3), np.uint8) * 255
        otsu_ext = np.zeros((height_ext, width_ext), np.uint8)
        prob_map_seg = np.zeros((height_ext, width_ext, num_classes), dtype=np.float32)
        weight_sum = np.zeros((height_ext, width_ext, num_classes), dtype=np.float32)

        #org_slide_ext[PATCH_OFFSET: PATCH_OFFSET + height, PATCH_OFFSET:PATCH_OFFSET + width, 0:3] = image_working[:, :,
        #                                                                                             0:3]

        # load the otsu results
        otsu_ext[PATCH_OFFSET: PATCH_OFFSET + height, PATCH_OFFSET:PATCH_OFFSET + width] = otsu_org[:, :]

        linedup_predictions = np.zeros((heights * widths, IMAGE_SIZE, IMAGE_SIZE, num_classes), dtype=np.float32)
        linedup_predictions[:, :, :, 0] = 1.0

        test_patch_tensor = torch.zeros([BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE], dtype=torch.float)
        if USE_GPU:
            test_path_tensor = test_patch_tensor.cuda(non_blocking=True)
        
        # get an identifier for the patch files to be written out as debugging
        unique_identifier = returnIdentifierFromImagePath(image_path)

        # decide how long this will take and prepare to give status updates in the log file
        iteration_count = heights*widths
        report_interval = iteration_count / (100 / REPORTING_INTERVAL)
        report_count = 0
        # report current state 
        percent_complete = 0

        patch_iter = 0
        inference_index = []
        position = 0
        stopcounter = 0

        for i in range(heights-1):
            for j in range(widths-1):
                #test_patch = org_slide_ext[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE,
                #             j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE, 0:3]

                # specify the region to extract and pull it at the proper magnification.  If a region is outside
                # of the image boundary, the returned tile will be padded with white pixels (background).  The region
                # coordinates are in the coordinate frame of the original, full-resolution image, so we need to calculate
                # them from the analytical coordinates
                top_in_orig = int(i * SLIDE_OFFSET * metadata['magnification']/ANALYSIS_MAGNIFICATION)
                left_in_orig = int(j * SLIDE_OFFSET * metadata['magnification'] / ANALYSIS_MAGNIFICATION)
                image_size_in_orig = int(IMAGE_SIZE* metadata['magnification'] / ANALYSIS_MAGNIFICATION)
                myRegion = {'top': top_in_orig, 'left': left_in_orig, 'width': image_size_in_orig, 'height': image_size_in_orig}
                rawtile, mimetype = source.getRegion(format=large_image.tilesource.TILE_FORMAT_NUMPY,
                                                        region=myRegion, scale={'magnification': ANALYSIS_MAGNIFICATION},
                                                        fill="white",output={'maxWidth':IMAGE_SIZE,'maxHeight':IMAGE_SIZE})
                # strip off any extra channels, RGB only
                test_patch = rawtile[:,:,0:3]
                # print out funny shaped patches... 
                if (test_patch.shape[0] != IMAGE_SIZE) or (test_patch.shape[1] != IMAGE_SIZE):
                    displayTileMetadata(test_patch,myRegion,i,j)
                    print(test_patch.shape)
        
                otsu_patch = otsu_ext[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE,
                                j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE]
                if np.sum(otsu_patch) > int(0.05 * IMAGE_SIZE * IMAGE_SIZE):
                    inference_index.append(patch_iter)
                    test_patch_tensor[position, :, :, :] = torch.from_numpy(test_patch.transpose(2, 0, 1)
                                                                            .astype('float32') / 255.0)
                    position += 1
                patch_iter += 1

                if position == BATCH_SIZE:
                    batch_predictions = _infer_batch(model, test_patch_tensor)

                    for k in range(BATCH_SIZE):
                        linedup_predictions[inference_index[k], :, :, :] = batch_predictions[k, :, :, :]

                    position = 0
                    inference_index = []

                # save data to look at
                #if (temp_i>100) and (temp_i<400):
                if (False):
                    np.save('hyun-patch-'+unique_identifier+'-'+str(temp_i)+'_'+str(temp_j)+'.npy', test_patch)
                    print('test_patch shape:', test_patch.shape, 'i:',temp_i,' j:',temp_j)
                    np.save('hyun-tensor-' + unique_identifier + '-' + str(temp_i) + '_' + str(temp_j) + '.npy', test_patch_tensor.cpu())
                    print('test_tensor shape:', test_patch.shape, 'i:', temp_i, ' j:', temp_j)
                    from PIL import Image
                    im = Image.fromarray(test_patch)
                    im.save('hyun-patch-'+str(temp_i)+'_'+str(temp_j)+'.jpeg')

                # check that it is time to report progress.  If so, print it and flush I/O to make sure it comes 
                # out right after it is printed 
                report_count += 1
                if (report_count > report_interval):
                    percent_complete += REPORTING_INTERVAL
                    print(f'progress: {percent_complete}')
                    sys.stdout.flush()
                    report_count = 0


        # Very last part of the region.  This is if there is a partial batch of tiles left at the
        # end of the image.
        batch_predictions = _infer_batch(model, test_patch_tensor)
        for k in range(position):
            linedup_predictions[inference_index[k], :, :, :] = batch_predictions[k, :, :, :]

        # finished with the model, clear the memory and GPU
        del test_patch_tensor
        del model
        if USE_GPU:
            torch.cuda.empty_cache()

        print('Inferencing complete. Constructing out image from patches')

        patch_iter = 0
        for i in range(heights-1 ):
            for j in range(widths-1):
                prob_map_seg[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE,
                j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE,:] \
                    += np.multiply(linedup_predictions[patch_iter, :, :, :], kernel)
                weight_sum[i * SLIDE_OFFSET: i * SLIDE_OFFSET + IMAGE_SIZE,
                j * SLIDE_OFFSET: j * SLIDE_OFFSET + IMAGE_SIZE,:] \
                    += kernel
                patch_iter += 1
        #np.save("prob_map_seg.npy",prob_map_seg)
        #np.save('weight_sum.npy',weight_sum)
        prob_map_seg = np.true_divide(prob_map_seg, weight_sum)
        prob_map_valid = prob_map_seg[PATCH_OFFSET:PATCH_OFFSET + height, PATCH_OFFSET:PATCH_OFFSET + width, :]

        # free main system memory since the images are big
        del prob_map_seg
        gc.collect()

        prob_map_valid = _unaugment(b, prob_map_valid)
        prob_map_seg_stack += prob_map_valid / num_tta

        # free main system memory since the images are big
        del prob_map_valid
        gc.collect()

    pred_map_final = np.argmax(prob_map_seg_stack, axis=-1)
    #np.save('prob_map_seg_stack.npy', prob_map_seg_stack)
    
    pred_map_final_gray = pred_map_final.astype('uint8') * 50
    del pred_map_final
    gc.collect()
    pred_map_final_ones = [(pred_map_final_gray == v) for v in CLASS_VALUES]
    del pred_map_final_gray
    gc.collect()
    pred_map_final_stack = np.stack(pred_map_final_ones, axis=-1).astype('uint8')
    del pred_map_final_ones
    gc.collect()

    pred_colormap = _gray_to_color(pred_map_final_stack)
    del pred_map_final_stack
    gc.collect()

    #np.save('pred_map_final_stack.npy', pred_map_final_stack)
    #prob_colormap = _gray_to_color(prob_map_seg_stack)
    #np.save('prob_colormap.npy', prob_colormap)

    # gather image data for writing out as DICOM
    dcm_file=pydicom.dcmread(image_path)
    #SizeInX=dcm_file.TotalPixelMatrixColumns
    #SizeInY=dcm_file.TotalPixelMatrixRows
    FrameSize=dcm_file.Rows
    FramesInX=int(SizeInX/FrameSize)
    FramesInY=int(SizeInY/FrameSize)
    # print image metadata
    metadata = {}
    metadata['sizeX'] = dcm_file.TotalPixelMatrixColumns
    metadata['sizeY'] = dcm_file.TotalPixelMatrixRows
    metadata['levels'] = 6   # arbitrary
    metadata['magnification'] = ANALYSIS_MAGNIFICATION
    print(metadata)
    print('sizeX:', metadata['sizeX'], 'sizeY:', metadata['sizeY'], 'levels:', metadata['levels'], metadata['magnification'])


    # write out the result as dicom-wsi
    imageInfo = {}
    imageInfo['ImageWidthInMm'] = 30  #(in mm) ((image_dims[1])*mpp)/1000)
    imageInfo['ImageHeightInMn'] =  20  #(in mm?) (image_dims[0])*mpp)/1000)
    imageInfo['StudyInstanceUID'] = '1.2.3.4.5'
    imageInfo['FrameOfReferenceUID'] = '1.3.4.5.6' 
    imageInfo['StudyID'] = 'prob_map'
    imageInfo['PixelSizeInMm'] = 0.0003  # (in mm?) (mpp/1000)
    imageInfo['ImageRowCount'] = width_org
    imageInfo['ImageColumnCount'] = height_org
    outpath = '/home/clisle/proj/slicer/PW39/images/out-P0005006/probability.dcm'
    out_image = (pred_colormap*255.0).astype('uint8')
    writeProbabilityMapToDICOM(out_image,outpath,imageInfo)

    # return image instead of saving directly
    return out_image




def _gaussian_2d(num_classes, sigma, mu):
    x, y = np.meshgrid(np.linspace(-1, 1, IMAGE_SIZE), np.linspace(-1, 1, IMAGE_SIZE))
    d = np.sqrt(x * x + y * y)
    # sigma, mu = 1.0, 0.0
    k = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

    k_min = np.amin(k)
    k_max = np.amax(k)

    k_normalized = (k - k_min) / (k_max - k_min)
    k_normalized[k_normalized<=EPSILON] = EPSILON

    kernels = [(k_normalized) for i in range(num_classes)]
    kernel = np.stack(kernels, axis=-1)

    print('Kernel shape: ', kernel.shape)
    print('Kernel Min value: ', np.amin(kernel))
    print('Kernel Max value: ', np.amax(kernel))

    return kernel


def reset_seed(seed):
    """
    ref: https://forums.fast.ai/t/accumulating-gradients/33219/28
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if USE_GPU:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def load_best_model(model, path_to_model, best_prec1=0.0):
    if os.path.isfile(path_to_model):
        print("=> loading checkpoint '{}'".format(path_to_model))
        checkpoint = torch.load(path_to_model, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}), best_precision {}"
              .format(path_to_model, checkpoint['epoch'], best_prec1))
        return model
    else:
        print("=> no checkpoint found at '{}'".format(path_to_model))


def inference_image(model, image_path, BATCH_SIZE, num_classes):
    kernel = _gaussian_2d(num_classes, 0.5, 0.0)
    predict_image = _inference(model, image_path, BATCH_SIZE, num_classes, kernel, 1)
    return predict_image

def start_inference(msg_queue, image_file):
    reset_seed(1)

    best_prec1_valid = 0.
    torch.backends.cudnn.benchmark = True

    #saved_weights_list = sorted(glob.glob(WEIGHT_PATH + '*.tar'))
    saved_weights_list = [WEIGHT_PATH+'model_iou_0.4996_0.5897_epoch_45.pth.tar'] 
    print(saved_weights_list)

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASS_VALUES),
        activation=ACTIVATION,
        aux_params=None,
    )

    model = nn.DataParallel(model)
    if USE_GPU:
        model = model.cuda()
    print('load pretrained weights')
    model = load_best_model(model, saved_weights_list[-1], best_prec1_valid)
    print('Loading model is finished!!!!!!!')

    # return image data so girder toplevel task can write it out
    predict_image = inference_image(model,image_file, BATCH_SIZE, len(CLASS_VALUES))

    # put the filename of the image in the message queue and return it to the main process
    msg_queue.put(predict_image)
    
    # not needed anymore, returning value through message queue
    #return predict_image

def start_inference_mainthread(image_file):
    reset_seed(1)

    best_prec1_valid = 0.
    torch.backends.cudnn.benchmark = True

    #saved_weights_list = sorted(glob.glob(WEIGHT_PATH + '*.tar'))
    saved_weights_list = [WEIGHT_PATH+'model_iou_0.7343_0.7175_epoch_60.pth.tar'] 
    print(saved_weights_list)

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASS_VALUES),
        activation=ACTIVATION,
        aux_params=None,
    )

    model = nn.DataParallel(model)
    if USE_GPU:
        model = model.cuda()
    print('load pretrained weights')
    model = load_best_model(model, saved_weights_list[-1], best_prec1_valid)
    print('Loading model is finished!!!!!!!')

    # return image data so girder toplevel task can write it out
    predict_image = inference_image(model,image_file, BATCH_SIZE, len(CLASS_VALUES))
    
    # not needed anymore, returning value through message queue
    return predict_image


# calculate the statistics for the image by converting to numpy and comparing masks against
# the tissue classes. create masks for each class and count the number of pixels

def generateStatsString(predict_image):
    # ERMS=red, ARMS=blue. Stroma=green, Necrosis = RG (yellow)
    img_arr = np.array(predict_image)
    # calculate total pixels = height*width
    total_pixels = img_arr.shape[0]*img_arr.shape[1]
    # count the pixels in the non-zero masks
    background_count = np.count_nonzero((img_arr == [255, 255, 255]).all(axis = 2))
    erms_count = np.count_nonzero((img_arr == [255, 0, 0]).all(axis = 2))
    stroma_count = np.count_nonzero((img_arr == [0, 255, 0]).all(axis = 2)) 
    arms_count = np.count_nonzero((img_arr == [0, 0, 255]).all(axis = 2)) 
    necrosis_count = np.count_nonzero((img_arr == [255, 255, 0]).all(axis = 2)) 
    print(f'erms {erms_count}, stroma {stroma_count}, arms {arms_count}, necrosis {necrosis_count}')
    # how much non-background is present.  This is tissue. Calculate percentages of tissue
    tissue_count = total_pixels - background_count
    erms_percent = erms_count / tissue_count * 100.0
    arms_percent = arms_count / tissue_count * 100.0
    necrosis_percent = necrosis_count / tissue_count * 100.0
    stroma_percent = stroma_count / tissue_count * 100.0
    # pack output values into a string returned as a file
    #statsString = 'ERMS:',erms_percent+'\n'+
    #              'ARMS:',arms_percent+'\n'+
    #              'stroma:',stroma_percent+'\n'+
    #              'necrosis:',necrosis_percent+'\n'
    statsDict = {'ERMS':erms_percent,
                 'ARMS':arms_percent, 
                 'stroma':stroma_percent, 
                 'necrosis':necrosis_percent }
    # convert dict to json string
    print('statsdict:',statsDict)
    statsString = json.dumps(statsDict)
    return statsString


#---------------- 
#----------------
# this dicom write routine is based on Max's early work
#----------------

import pydicom
import io
from io import BytesIO

import skimage.measure
from tqdm import tqdm
from time import  sleep
import dask.array as da
from PIL import Image
from PIL import ImageCms
#import matplotlib.pyplot as plt
from pydicom.encaps import encapsulate
from pydicom.uid import generate_uid
from pydicom.dataset import Dataset
from pydicom.encaps import generate_pixel_data_frame
from pydicom.dataset import Dataset, FileDataset, DataElement
from pydicom.sequence import Sequence
from random import randint
from datetime import datetime


'''
 imageInfoRecord dictionary contains things that were originally extracted from the reference dicom:
 ImageWidthInMm (in mm) ((image_dims[1])*mpp)/1000)
 ImageHeightInMn (in mm?) (image_dims[0])*mpp)/1000)
 StudyInstanceUID
 StudyID
 FrameOfReferenceUID
 PixelSizeInMm (in mm?) (mpp/1000)
 ImageRowCount
 ImageColumnCount

'''

def writeProbabilityMapToDICOM(map,out_path,imageInfoRecord):

    # ImageCms is a color LUT transform
    #created_profile=ImageCms.createProfile('sRGB')
    #prf=ImageCms.ImageCmsProfile(created_profile)
    #ICC_Profil=prf.tobytes()
    pat_name='probabilityMap'
    pat_name=str(pat_name)
    #image_dims=map.shape

    #mpp=float(MPP_value)
    #volume_width=reference_DCM.ImagedVolumeWidth#((image_dims[1])*mpp)/1000
    #volume_height = reference_DCM.ImagedVolumeHeight#((image_dims[0]) * mpp) / 1000
    volume_width=imageInfoRecord['ImageWidthInMm'] #((image_dims[1])*mpp)/1000
    volume_height = imageInfoRecord['ImageHeightInMn'] #((image_dims[0]) * mpp) / 1000
    #OriginalPixelSize=volume_width/image_dims[0]
    OriginalPixelSize=imageInfoRecord['PixelSizeInMm']
    date_time=str(datetime.now())
    date=date_time[0:10].replace('-','')
    time=date_time[10:].replace(':','')
    rows=imageInfoRecord['ImageRowCount']
    columns=imageInfoRecord['ImageColumnCount']
    numbr_of_frames=1
    SOPinstanceUID = generate_uid()
    file_name = 'new_probability_map.dcm'
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.77.1.6'  # VL Whole Slide Microscopy Image Storage
    file_meta.MediaStorageSOPInstanceUID = SOPinstanceUID
    file_meta.ImplementationClassUID = '1.3.6.1.4.1.5962.99.2'
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.ImplementationVersionName = 'KVisTest'
    file_meta.SourceApplicationEntityTitle = 'KVisTitle'
    # see transfer Syntax here: https://www.dicomlibrary.com/dicom/transfer-syntax/
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.4.50' # default for JPEG encoding, which we will use below
    file_meta.FileMetaInformationGroupLength = len(file_meta)
    # using values from Max's original example
    # this is creating the Dicom Dataset; below we will it
    dcm_file = FileDataset(file_name, {}, preamble=b"\0" * 128, file_meta=file_meta, is_implicit_VR=False,is_little_endian=True)
    dcm_file.ImageType = ['DERIVED', 'PRIMARY', 'VOLUME', 'RESAMPLED']
    dcm_file.SOPClassUID = '1.2.840.10008.5.1.4.1.1.77.1.6'  # VL Whole Slide Microscopy Image Storage
    dcm_file.SOPInstanceUID = SOPinstanceUID
    dcm_file.ContentDate = date
    tag = pydicom.tag.Tag('AcquisitionDateTime')
    pd_ele = DataElement(tag, 'TM', time)
    dcm_file.add(pd_ele)

    tag = pydicom.tag.Tag('StudyTime')
    pd_ele = DataElement(tag, 'TM', time)
    dcm_file.add(pd_ele)

    tag = pydicom.tag.Tag('ContentTime')
    pd_ele = DataElement(tag, 'TM', time)
    dcm_file.add(pd_ele)
    dcm_file.AccessionNumber = 'A20210527083404'
    dcm_file.Modality = 'SM'
    dcm_file.Manufacturer = 'FNLCR'
    dcm_file.ReferringPhysicianName = 'SOME^PHYSICIAN'
    ##########################################
    dcm_file_Coding_Scheme_Identific_1 = Dataset()
    dcm_file_Coding_Scheme_Identific_1.CodingSchemeDesignator = "DCM"
    dcm_file_Coding_Scheme_Identific_1.CodingSchemeUID = "DICOM Controlled Terminology"
    dcm_file_Coding_Scheme_Identific_1.CodingSchemeRegistry = "HL7"
    dcm_file_Coding_Scheme_Identific_1.CodingSchemeName = "DICOM Controlled Terminology"
    dcm_file_Coding_Scheme_Identific_2 = Dataset()
    dcm_file_Coding_Scheme_Identific_2.CodingSchemeDesignator = "SCT"
    dcm_file_Coding_Scheme_Identific_2.CodingSchemeUID = "2.16.840.1.113883.6.96"
    dcm_file_Coding_Scheme_Identific_2.CodingSchemeRegistry = "HL7"
    dcm_file_Coding_Scheme_Identific_2.CodingSchemeName = "SNOMED-CT using SNOMED-CT style values"
    dcm_file.CodingSchemeIdentificationSequence = Sequence([dcm_file_Coding_Scheme_Identific_1, dcm_file_Coding_Scheme_Identific_2])
    dcm_file.TimezoneOffsetFromUTC = '+0200'
    dcm_file.StudyDescription = ''
    dcm_file.ManufacturerModelName = 'MyModel'
    dcm_file.VolumetricProperties = 'VOLUME'
    dcm_file.PatientName = pat_name
    dcm_file.PatientID = pat_name
    dcm_file.PatientBirthDate = '2021-10-23'  # '19700101'
    dcm_file.PatientSex = 'M'
    dcm_file.DeviceSerialNumber = 'MySerialNumber'
    dcm_file.SoftwareVersions = 'MyVersion'
    dcm_file.AcquisitionDuration = 80
    ContributingEquipment = Dataset()
    ContributingEquipment.Manufacturer = 'Manu'
    ContributingEquipment.InstitutionName = 'Instui'
    ContributingEquipment.InstitutionAddress = 'Add'
    ContributingEquipment.InstitutionalDepartmentName = 'Develop'
    ContributingEquipment.ManufacturerModelName = 'Decription'
    ContributingEquipment.SoftwareVersions = 'wsi2dcm'
    ContributingEquipment.ContributionDateTime = '20210103165006.573-000'
    ContributingEquipment.ContributionDescription = 'Description'
    PurposeOfReferenceCodeSequence = Dataset()
    PurposeOfReferenceCodeSequence.CodeValue = "109103"
    PurposeOfReferenceCodeSequence.CodingSchemeDesignator = "DCM"
    PurposeOfReferenceCodeSequence.CodeMeaning = "Modifying Equipment"
    ContributingEquipment.PurposeOfReferenceCodeSequence = Sequence([PurposeOfReferenceCodeSequence])
    dcm_file.ContributingEquipmentSequence = Sequence([ContributingEquipment])
    #dcm_file.StudyInstanceUID = StudyInstanceUID#study_instance_uid#---------------------------adaptions
    dcm_file.StudyInstanceUID = imageInfoRecord['StudyInstanceUID']
    dcm_file.SeriesInstanceUID = generate_uid()#series_instance_uid#----------------------------adaptions
    #dcm_file.StudyID = reference_DCM.StudyID#study_id#----------------------------------------------------------------------------------------------------adaptions
    dcm_file.StudyID = imageInfoRecord['StudyID']
    dcm_file.SeriesNumber = ''
    dcm_file.InstanceNumber = '2'#'10'
    #dcm_file.FrameOfReferenceUID = reference_DCM.FrameOfReferenceUID#frame_of_reference#-----------------------------------adaptions
    dcm_file.FrameOfReferenceUID = imageInfoRecord['FrameOfReferenceUID']
    dcm_file.PositionReferenceIndicator = 'SLIDE_CORNER'
    dcm_file.ImageComments = 'http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/'
    dcm_file_DimensionOrganization = Dataset()
    dcm_file_DimensionOrganization.DimensionOrganizationUID = generate_uid()
    dcm_file.DimensionOrganizationSequence = Sequence([dcm_file_DimensionOrganization])
    dcm_file.DimensionOrganizationType = 'TILED_FULL'
    dcm_file.SamplesPerPixel = 3
    dcm_file.PhotometricInterpretation = 'YBR_FULL_422'
    dcm_file.PlanarConfiguration = 0
    dcm_file.NumberOfFrames = int(numbr_of_frames)
    dcm_file.Rows = rows
    dcm_file.Columns = columns
    dcm_file.BitsAllocated = 8
    dcm_file.BitsStored = 8
    dcm_file.HighBit = 7
    dcm_file.PixelRepresentation = 0
    dcm_file.BurnedInAnnotation = 'NO'
    dcm_file.LossyImageCompression = '01'
    dcm_file.LossyImageCompressionRatio = [25.0, 25.0]  # [24.91,24.91]
    dcm_file.LossyImageCompressionMethod = ['ISO_10918_1', 'ISO_10918_1']
    dcm_file.ContainerIdentifier = '888'  # '1000'+str(path[42:])
    dcm_file.IssuerOfTheContainerIdentifierSequence = []
    dcm_file_ContainerTypeCodeSequence = Dataset()
    dcm_file_ContainerTypeCodeSequence.CodeValue = '433466003'
    dcm_file_ContainerTypeCodeSequence.CodingSchemeDesignator = 'SCT'
    dcm_file_ContainerTypeCodeSequence.CodeMeaning = 'Microscope slide'
    dcm_file.ContainerTypeCodeSequence = Sequence([dcm_file_ContainerTypeCodeSequence])
    dcm_file.AcquisitionContextSequence = []
    dcm_file.ColorSpace = 'sRGB'
    Specimen_Description_Sequence = Dataset()
    Primary_Anatomic_Structure_Sequence = Dataset()
    Primary_Anatomic_Structure_Sequence.CodeValue = '32849002'
    Primary_Anatomic_Structure_Sequence.CodingSchemeDesignator = 'SCT'
    Primary_Anatomic_Structure_Sequence.CodeMeaning = ''
    Specimen_Description_Sequence.PrimaryAnatomicStructureSequence = Sequence([Primary_Anatomic_Structure_Sequence])
    Specimen_Description_Sequence.SpecimenIdentifier = 'Running Identifier(may be provided in YAML)'  # 'Unknown_0_20210527083404'
    Specimen_Description_Sequence.SpecimenUID = generate_uid()
    Specimen_Description_Sequence.IssuerOfTheSpecimenIdentifierSequence = []
    Specimen_Description_Sequence.SpecimenShortDescription = 'RMS'
    Specimen_Description_Sequence.SpecimenDetailedDescription = 'Rhabdomyosarcoma probability prediction'
    ######################################################
    ##########################################################
    ########################################################
    specimen_preparation_sequence1 = Dataset()
    a = Dataset()
    a.ValueType = 'TEXT'
    a_ConceptNameCodeSequence = Dataset()
    a_ConceptNameCodeSequence.CodeValue = '121041'
    a_ConceptNameCodeSequence.CodingSchemeDesignator = 'DCM'
    a_ConceptNameCodeSequence.CodeMeaning = 'Specimen Identifier'
    a.ConceptNameCodeSequence = Sequence([a_ConceptNameCodeSequence])
    a.TextValue = 'Array'  ##########################ARR Checken
    b = Dataset()
    b.ValueType = 'CODE'
    b_ConceptNameCodeSequence = Dataset()
    b_ConceptNameCodeSequence.CodeValue = '111701'
    b_ConceptNameCodeSequence.CodingSchemeDesignator = 'DCM'
    b_ConceptNameCodeSequence.CodeMeaning = 'Processing type'
    b.ConceptNameCodeSequence = Sequence([b_ConceptNameCodeSequence])
    b_Concept_Code_Sequence = Dataset()
    b_Concept_Code_Sequence.CodeValue = '9265001'
    b_Concept_Code_Sequence.CodingSchemeDesignator = 'SCT'
    b_Concept_Code_Sequence.CodeMeaning = 'Specimen processing'
    b.ConceptCodeSequence = Sequence([b_Concept_Code_Sequence])
    c = Dataset()
    c.ValueType = 'CODE'
    c_ConceptNameCodeSequence = Dataset()
    c_ConceptNameCodeSequence.CodeValue = '430864009'
    c_ConceptNameCodeSequence.CodingSchemeDesignator = 'SCT'
    c_ConceptNameCodeSequence.CodeMeaning = 'Tissue Fixative'
    c.ConceptNameCodeSequence = Sequence([c_ConceptNameCodeSequence])
    c_Concept_Code_Sequence = Dataset()
    c_Concept_Code_Sequence.CodeValue = '431510009'
    c_Concept_Code_Sequence.CodingSchemeDesignator = 'SCT'
    c_Concept_Code_Sequence.CodeMeaning = 'some description'
    c.ConceptCodeSequence = Sequence([c_Concept_Code_Sequence])
    specimen_preparation_sequence1.SpecimenPreparationStepContentItemSequence = Sequence([a, b, c])
    ################################################################
    specimen_preparation_sequence2 = Dataset()
    a2 = Dataset()
    a2.ValueType = 'TEXT'
    a2_ConceptNameCodeSequence = Dataset()
    a2_ConceptNameCodeSequence.CodeValue = '121041'
    a2_ConceptNameCodeSequence.CodingSchemeDesignator = 'DCM'
    a2_ConceptNameCodeSequence.CodeMeaning = 'Specimen Identifier'
    a2.ConceptNameCodeSequence = Sequence([a2_ConceptNameCodeSequence])
    a2.TextValue = 'Array'  ##########################ARR Checken
    b2 = Dataset()
    b2.ValueType = 'CODE'
    b2_ConceptNameCodeSequence = Dataset()
    b2_ConceptNameCodeSequence.CodeValue = '111701'
    b2_ConceptNameCodeSequence.CodingSchemeDesignator = 'DCM'
    b2_ConceptNameCodeSequence.CodeMeaning = 'Processing type'
    b2.ConceptNameCodeSequence = Sequence([b2_ConceptNameCodeSequence])
    b2_Concept_Code_Sequence = Dataset()
    b2_Concept_Code_Sequence.CodeValue = '9265001'
    b2_Concept_Code_Sequence.CodingSchemeDesignator = 'SCT'
    b2_Concept_Code_Sequence.CodeMeaning = 'Specimen processing'
    b2.ConceptCodeSequence = Sequence([b2_Concept_Code_Sequence])
    c2 = Dataset()
    c2.ValueType = 'CODE'
    c2_ConceptNameCodeSequence = Dataset()
    c2_ConceptNameCodeSequence.CodeValue = '430863003'
    c2_ConceptNameCodeSequence.CodingSchemeDesignator = 'SCT'
    c2_ConceptNameCodeSequence.CodeMeaning = 'Embedding medium'
    c2.ConceptNameCodeSequence = Sequence([c2_ConceptNameCodeSequence])
    c2_Concept_Code_Sequence = Dataset()
    c2_Concept_Code_Sequence.CodeValue = '311731000'
    c2_Concept_Code_Sequence.CodingSchemeDesignator = 'SCT'
    c2_Concept_Code_Sequence.CodeMeaning = 'some medium'  # 'Paraffin wax'
    c2.ConceptCodeSequence = Sequence([c2_Concept_Code_Sequence])
    specimen_preparation_sequence2.SpecimenPreparationStepContentItemSequence = Sequence([a2, b2, c2])
    #########################################################
    #############################################################
    specimen_preparation_sequence3 = Dataset()
    a3 = Dataset()
    a3.ValueType = 'TEXT'
    a3_ConceptNameCodeSequence = Dataset()
    a3_ConceptNameCodeSequence.CodeValue = '121041'
    a3_ConceptNameCodeSequence.CodingSchemeDesignator = 'DCM'
    a3_ConceptNameCodeSequence.CodeMeaning = 'Specimen Identifier'
    a3.ConceptNameCodeSequence = Sequence([a3_ConceptNameCodeSequence])
    a3.TextValue = 'Array'  ##########################ARR Checken
    b3 = Dataset()
    b3.ValueType = 'CODE'
    b3_ConceptNameCodeSequence = Dataset()
    b3_ConceptNameCodeSequence.CodeValue = '111701'
    b3_ConceptNameCodeSequence.CodingSchemeDesignator = 'DCM'
    b3_ConceptNameCodeSequence.CodeMeaning = 'Processing type'
    b3.ConceptNameCodeSequence = Sequence([b3_ConceptNameCodeSequence])
    b3_Concept_Code_Sequence = Dataset()
    b3_Concept_Code_Sequence.CodeValue = '127790008'
    b3_Concept_Code_Sequence.CodingSchemeDesignator = 'SCT'
    b3_Concept_Code_Sequence.CodeMeaning = 'Staining'
    b3.ConceptCodeSequence = Sequence([b3_Concept_Code_Sequence])
    c3 = Dataset()
    c3.ValueType = 'CODE'
    c3_ConceptNameCodeSequence = Dataset()
    c3_ConceptNameCodeSequence.CodeValue = '424361007'
    c3_ConceptNameCodeSequence.CodingSchemeDesignator = 'SCT'
    c3_ConceptNameCodeSequence.CodeMeaning = 'Using substance'
    c3.ConceptNameCodeSequence = Sequence([c3_ConceptNameCodeSequence])
    c3_Concept_Code_Sequence = Dataset()
    c3_Concept_Code_Sequence.CodeValue = '12710003'
    c3_Concept_Code_Sequence.CodingSchemeDesignator = 'SCT'
    c3_Concept_Code_Sequence.CodeMeaning = 'H&E or IHC'
    c3.ConceptCodeSequence = Sequence([c3_Concept_Code_Sequence])
    d3 = Dataset()
    d3.ValueType = 'CODE'
    d3_Concept_Name_Code_Sequence = Dataset()
    d3_Concept_Name_Code_Sequence.CodeValue = '424361007'
    d3_Concept_Name_Code_Sequence.CodingSchemeDesignator = 'SCT'
    d3_Concept_Name_Code_Sequence.CodeMeaning = 'Using substance'
    d3.ConceptNameCodeSequence = Sequence([d3_Concept_Name_Code_Sequence])
    d3_Concept_Code_Sequence = Dataset()
    d3_Concept_Code_Sequence.CodeValue = '36879007'
    d3_Concept_Code_Sequence.CodingSchemeDesignator = 'SCT'
    d3_Concept_Code_Sequence.CodeMeaning = 'further description'
    d3.ConceptCodeSequence = Sequence([d3_Concept_Code_Sequence])
    specimen_preparation_sequence3.SpecimenPreparationStepContentItemSequence = Sequence([a3, b3, c3, d3])
    #######################
    Specimen_Description_Sequence.SpecimenPreparationSequence = Sequence([specimen_preparation_sequence1, specimen_preparation_sequence2, specimen_preparation_sequence3])
    dcm_file.SpecimenDescriptionSequence = Sequence([Specimen_Description_Sequence])
    ######################################
    #####################################
    dcm_file.ImagedVolumeWidth = volume_width
    dcm_file.ImagedVolumeHeight = volume_height
    dcm_file.ImagedVolumeDepth = 1.0
    dcm_file.TotalPixelMatrixColumns = int(columns)
    dcm_file.TotalPixelMatrixRows = int(rows)
    dcm_file_Total_Pixel_Matrix_Origin_Sequence = Dataset()
    dcm_file_Total_Pixel_Matrix_Origin_Sequence.XOffsetInSlideCoordinateSystem = 0.0  # float(linearxoffset)#float(slide.properties.get('aperio.LineAreaXOffset'))#float(slide.properties.get('aperio.Left'))
    dcm_file_Total_Pixel_Matrix_Origin_Sequence.YOffsetInSlideCoordinateSystem = 0.0  # float(linearyoffset)#float(slide.properties.get('aperio.LineAreaYOffset'))#float(slide.properties.get('aperio.Left'))
    dcm_file.TotalPixelMatrixOriginSequence = Sequence([dcm_file_Total_Pixel_Matrix_Origin_Sequence])
    ####################################################
    ####################################################
    dcm_file.SpecimenLabelInImage = 'NO'
    dcm_file.FocusMethod = 'AUTO'
    dcm_file.ExtendedDepthOfField = 'NO'
    dcm_file.ImageOrientationSlide = ['-1', '0', '0', '0', '-1', '0']
    dcm_file_Optical_Path_Sequence = Dataset()
    dcm_file_Illumination_Type_Code_Sequence1 = Dataset()
    dcm_file_Illumination_Type_Code_Sequence1.CodeValue = '111744'
    dcm_file_Illumination_Type_Code_Sequence1.CodingSchemeDesignator = 'DCM'
    dcm_file_Illumination_Type_Code_Sequence1.CodeMeaning = 'Brightfield illumination'
    dcm_file_Optical_Path_Sequence.IlluminationTypeCodeSequence = Sequence([dcm_file_Illumination_Type_Code_Sequence1])
    #dcm_file_Optical_Path_Sequence.ICCProfile = ICC_Profil
    dcm_file_Optical_Path_Sequence.OpticalPathIdentifier = '0'
    dcm_file_Illumination_Type_Code_Sequence2 = Dataset()
    dcm_file_Illumination_Type_Code_Sequence2.CodeValue = '414298005'
    dcm_file_Illumination_Type_Code_Sequence2.CodingSchemeDesignator = 'SCT'
    dcm_file_Illumination_Type_Code_Sequence2.CodeMeaning = 'Full Spectrum'
    dcm_file_Optical_Path_Sequence.IlluminationColorCodeSequence = Sequence([dcm_file_Illumination_Type_Code_Sequence2])
    dcm_file.OpticalPathSequence = Sequence([dcm_file_Optical_Path_Sequence])
    dcm_file.NumberOfOpticalPaths = 1
    dcm_file.TotalPixelMatrixFocalPlanes = 1
    dcm_file_Shared_Functional_Groups = Dataset()
    dcm_file_Pixel_Measures_Sequence = Dataset()
    dcm_file_Pixel_Measures_Sequence.SliceThickness = '0.001'  # '0.0010000002384'
    dcm_file_Pixel_Measures_Sequence.SpacingBetweenSlices = '0.006'  # '0.0006'
    print('original pixel size:',OriginalPixelSize)
    dcm_file_Pixel_Measures_Sequence.PixelSpacing = [str(OriginalPixelSize), str(OriginalPixelSize)]
    dcm_file_Shared_Functional_Groups.PixelMeasuresSequence = Sequence([dcm_file_Pixel_Measures_Sequence])
    dcm_file_Whole_Slide_Microscopy_Image_Frame_Type_Sequence = Dataset()

    dcm_file_Whole_Slide_Microscopy_Image_Frame_Type_Sequence.FrameType = ['DERIVED', 'PRIMARY', 'PROBABILITY_MAP','RESAMPLED']
    dcm_file_Shared_Functional_Groups.WholeSlideMicroscopyImageFrameTypeSequence = Sequence([dcm_file_Whole_Slide_Microscopy_Image_Frame_Type_Sequence])
    dcm_file_Optical_Path_Identification_Sequence = Dataset()
    dcm_file_Optical_Path_Identification_Sequence.OpticalPathIdentifier = '0'
    dcm_file_Shared_Functional_Groups.OpticalPathIdentificationSequence = Sequence([dcm_file_Optical_Path_Identification_Sequence])
    dcm_file.SharedFunctionalGroupsSequence = Sequence([dcm_file_Shared_Functional_Groups])
 
    encoded_frames = []
    instance_byte_string_buffer = io.BytesIO()
    image = Image.fromarray(map)
    profile = image.info.get('icc_profile')
    # removed ICC profile since this is probability data
    #image.save(instance_byte_string_buffer, "JPEG", quality=95, icc_profile=profile, progressive=False)
    image.save(instance_byte_string_buffer, "JPEG", quality=95, progressive=False)

    t = instance_byte_string_buffer.getvalue()
    encoded_frames.append(t)
    capsulated = encapsulate(encoded_frames, has_bot=True)
    pixeL_data = capsulated
    data_elem_tag = pydicom.tag.TupleTag((0x7FE0, 0x0010))
    pd_ele = DataElement(data_elem_tag, 'OB', pixeL_data, is_undefined_length=True)
    dcm_file.add(pd_ele)
    store_path = out_path + file_name
    dcm_file.save_as(store_path, write_like_original=False)
    return 0




#imagePath = '/Volumes/CurtData/RMS-work/sample_images/Sample_WSI_Image.sys'
# this file seems broken
#imagePath = '/media/clisle/CurtData/RMS-work/sample_images/PAWDLM-0BLLXP_A2_RAW/DCM_0'
# this image just barely processes in 64GB RAM
#imagePath = '/media/clisle/Imaging/IDC/sample_data_from_pixelmed/PARNED-0BNNF4_B2_Q30/DCM_0'
# this image requires > 64GB RAM to process
#imagePath = '/media/clisle/CurtData/RMS-work/public-wsi-images/outputPARNJS/level-1-frames-0-2496.dcm'
# this one just fits in 64GB RAM plus swap
#imagePath = '/media/clisle/CurtData/RMS-work/public-wsi-images/output/level-0-frames-0-6072.dcm'

imagePath =  '/home/clisle/proj/slicer/PW39/images/P0005006/1.2.826.0.1.3680043.8.498.26978701885807404641007366994220544128.dcm'


outfile,outstats = infer_rhabdo(imagePath)