import numpy as np



#---------------- DICOM export --------


from pathlib import Path
import os
import highdicom as hd
import dicomslide
from pydicom.sr.codedict import codes
from pydicom.filereader import dcmread
from pydicom import Dataset
from dicomweb_client import DICOMfileClient
from tempfile import TemporaryDirectory
from typing import Tuple




#  This is based on the hidicom output example listed in the readthedocs documentation

CHANNEL_DESCRIPTION = {}
CHANNEL_DESCRIPTION['chan_1'] = 'Necrosis'
CHANNEL_DESCRIPTION['chan_2'] = 'Stroma'
CHANNEL_DESCRIPTION['chan_3'] = 'ARMS'
CHANNEL_DESCRIPTION['chan_4'] = 'ERMS'

def writeDicomSegObject(image_path, seg_image, out_path):

    # Path to multi-frame SM image instance stored as PS3.10 file
    image_file = Path(image_path)

   # Read SM Image data set from PS3.10 files on disk.  This will provide the 
    # reference image size and other dicom header information
    image_dataset = dcmread(str(image_file))

   # function stolen from idc-pan-cancer-archive repository to re-tile the numpy to match the tiling
   # from the source image
    mask = disassemble_total_pixel_matrix(seg_image[:,:,0],image_dataset)

    # make the derived image header information
    derived_plane_positions,derived_pixel_measures = _compute_derived_image_attributes(image_dataset, mask)

    # Describe the algorithm that created the segmentation
    algorithm_identification = hd.AlgorithmIdentificationSequence(
        name='FNLCR_RMS_seg_iou_0.7343_epoch_60',
        version='v1.0',
        family=codes.cid7162.ArtificialIntelligence
    )

    # Describe the segment
    description_segment_1 = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label=CHANNEL_DESCRIPTION['chan_1'],
        segmented_property_category=codes.cid7150.Tissue,
        segmented_property_type=codes.cid7166.ConnectiveTissue,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=hd.UID(),
        tracking_id='RMS segmentation_'+str(CHANNEL_DESCRIPTION['chan_1'])
    )

 # Describe the segment
    description_segment_2 = hd.seg.SegmentDescription(
        segment_number=2,
        segment_label=CHANNEL_DESCRIPTION['chan_2'],
        segmented_property_category=codes.cid7150.Tissue,
        segmented_property_type=codes.cid7166.ConnectiveTissue,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=hd.UID(),
        tracking_id='RMS segmentation_'+str(CHANNEL_DESCRIPTION['chan_2'])
    )

     # Describe the segment
    description_segment_3 = hd.seg.SegmentDescription(
        segment_number=3,
        segment_label=CHANNEL_DESCRIPTION['chan_3'],
        segmented_property_category=codes.cid7150.Tissue,
        segmented_property_type=codes.cid7166.ConnectiveTissue,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=hd.UID(),
        tracking_id='RMS segmentation_'+str(CHANNEL_DESCRIPTION['chan_3'])
    )

     # Describe the segment
    description_segment_4 = hd.seg.SegmentDescription(
        segment_number=4,
        segment_label=CHANNEL_DESCRIPTION['chan_4'],
        segmented_property_category=codes.cid7150.Tissue,
        segmented_property_type=codes.cid7166.ConnectiveTissue,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=hd.UID(),
        tracking_id='RMS segmentation_'+str(CHANNEL_DESCRIPTION['chan_4'])
    )

    # Create the Segmentation instance
    seg_dataset = hd.seg.Segmentation(
        source_images=[image_dataset],
        pixel_array=mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[description_segment_1,description_segment_2,description_segment_3,description_segment_4],
        series_instance_uid=hd.UID(),
        series_number=2,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        # the following two entries are added because the output resolution is different from the source
        pixel_measures=derived_pixel_measures,
        plane_positions= derived_plane_positions,
        manufacturer='Aperio',
        manufacturer_model_name='Unknown',
        software_versions='v1',
        device_serial_number='Unknown'
    )

    print(seg_dataset)
    # change output file with some function is needed
    outfileanme = out_path
    seg_dataset.save_as(outfileanme)


def writeDicomFractionalSegObject(image_path, seg_image, out_path):

    # Path to multi-frame SM image instance stored as PS3.10 file
    image_file = Path(image_path)

    # Read SM Image data set from PS3.10 files on disk.  This will provide the 
    # reference image size and other dicom header information
    image_dataset = dcmread(str(image_file))
    source_rows = image_dataset.TotalPixelMatrixRows
    source_cols = image_dataset.TotalPixelMatrixColumns

    # function stolen from idc-pan-cancer-archive repository to re-tile the numpy to match the tiling
    # from the source image
    print('passing in a numpy array of shape:',seg_image.shape)

    # create a mask that is exactly the same size as the source image
    mask = np.zeros((source_rows, source_cols), np.uint8)
    for i in range(seg_image.shape[0]):
        for j in range(seg_image.shape[1]):
            mask[i,j] = seg_image[i,j,3]*255
    #mask[:,:] = seg_image[ :, :, 3]
    #mask = seg_image[:,:,:]

    #mask = disassemble_total_pixel_matrix(seg_image[:,:,3],image_dataset)
    #print('disassembled dimensions:',mask.shape)
    mask_rolled = np.moveaxis(mask, -1, 0)
    #mask_rolled = mask_rolled[None, :, :, :]
    print('rolled dimensions:',mask_rolled.shape)


    # make the derived image header information
    #derived_plane_positions,derived_pixel_measures = _compute_derived_image_attributes(image_dataset, mask)

    # Describe the algorithm that created the segmentation
    algorithm_identification = hd.AlgorithmIdentificationSequence(
        name='FNLCR_RMS_probability_iou_0.7343_epoch_60',
        version='v1.0',
        family=codes.cid7162.ArtificialIntelligence
    )


    # Describe the segment
    description_segment_3 = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label=CHANNEL_DESCRIPTION['chan_3'],
        segmented_property_category=codes.cid7150.Tissue,
        segmented_property_type=codes.cid7166.ConnectiveTissue,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=hd.UID(),
        tracking_id='RMS segmentation_'+str(CHANNEL_DESCRIPTION['chan_3'])
    )

    '''
    # Describe the segment
    description_segment_4 = hd.seg.SegmentDescription(
        segment_number=4,
        segment_label=CHANNEL_DESCRIPTION['chan_4'],
        segmented_property_category=codes.cid7150.Tissue,
        segmented_property_type=codes.cid7166.ConnectiveTissue,
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=hd.UID(),
        tracking_id='RMS segmentation_'+str(CHANNEL_DESCRIPTION['chan_4'])
    )
    '''

    mask_arrays = [mask_rolled[0],mask_rolled[1],mask_rolled[2],mask_rolled[3]]

    # Create the Segmentation instance
    seg_dataset = hd.seg.create_segmentation_pyramid(
        source_images=[image_dataset],
        pixel_arrays=[mask],
        downsample_factors=[2.0, 4.0],
        #segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segmentation_type=hd.seg.SegmentationTypeValues.FRACTIONAL,
        #segment_descriptions=[description_segment_1,description_segment_2,description_segment_3,description_segment_4],
        segment_descriptions=[description_segment_3],
        series_instance_uid=hd.UID(),
        series_number=2,
        # the following two entries are added because the output resolution is different from the source
        #pixel_measures=derived_pixel_measures,
        #plane_positions= derived_plane_positions,
        manufacturer='Aperio',
        manufacturer_model_name='Unknown',
        software_versions='v1_probability',
        device_serial_number='Unknown'
    )

    #print(seg_dataset)
    # change output file with some function is needed
    outfilename = out_path
    if len(seg_dataset)>1:
        for i,seg in enumerate(seg_dataset):
            outfilename = os.path.join(out_path,'seg_'+str(i)+'.dcm')
            seg.save_as(outfilename)
        else:
            seg_dataset.save_as(outfilename)

#-----------------------------
import logging
from time import time
from typing import List, Union
import xml

import numpy as np
import highdicom as hd
import pydicom
from pydicom import Dataset
from pydicom.sr.codedict import codes
from pydicom.uid import JPEGLSLossless, ExplicitVRLittleEndian
import metadata_config


def convert_segmentation(
    source_images: List[pydicom.Dataset],
    segmentation_array: np.ndarray,
    create_pyramid: bool,
    segmentation_type: Union[hd.seg.SegmentationTypeValues, str],
    dimension_organization_type: Union[hd.DimensionOrganizationTypeValues, str],
    workers: int = 0,
) -> List[hd.seg.Segmentation]:
    
    """Store segmentation masks as DICOM segmentations.

    Parameters
    ----------
    source_images: Sequence[pydicom.Dataset]
        List of pydicom datasets containing the metadata of the image (already
        converted to DICOM format). Note that the metadata of the image at full
        resolution should appear first in this list. These can be the full image
        datasets, but the PixelData attributes are not required.
    segmentation_array: np.ndarray
        Segmentation output (before thresholding). Should have shape (rows,
        columns, 5), where 5 is the number of classes. Values are in the range
        0 to 1.
    create_pyramid: bool, optional
        Whether to create a full pyramid of segmentations (rather than a single
        segmentation at the highest resolution).
    segmentation_type: Union[hd.seg.SegmentationTypeValues, str], optional
        Segmentation type (BINARY or FRACTIONAL) for the Segmentation Image
        (if any).
    dimension_organization_type: Union[hd.DimensionOrganizationTypeValues, str], optional
        Dimension organization type of the output segmentations.
    workers: int
        Number of workers to use for frame compression.

    Returns
    -------
    segmentations: list[hd.seg.Segmentation]
        DICOM segmentation image(s) encoding the original annotations

    """


    segment_descriptions = []
    
    for number, (label, (prop_code, cat_code)) in enumerate(
        metadata_config.finding_codes.items(),
        start=1
    ):
        desc = hd.seg.SegmentDescription(
            segment_number=number,
            segment_label=label,
            segmented_property_category=cat_code,
            segmented_property_type=prop_code,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=metadata_config.algorithm_identification
        )
        segment_descriptions.append(desc)
   

    segmentation_type = hd.seg.SegmentationTypeValues(segmentation_type)
    dimension_organization_type = hd.DimensionOrganizationTypeValues(
        dimension_organization_type
    )

    # Compression method depends on what is possible given the chosen
    # segmentation type
    transfer_syntax_uid = {
        hd.seg.SegmentationTypeValues.BINARY: ExplicitVRLittleEndian,
        #hd.seg.SegmentationTypeValues.FRACTIONAL: JPEGLSLossless,
    }[segmentation_type]

    omit_empty_frames = dimension_organization_type.value != "TILED_FULL"

    if segmentation_type == hd.seg.SegmentationTypeValues.FRACTIONAL:
        # Add frame axis and remove background class
        mask = segmentation_array[None, :, :, 1:]
        print('mask.shape:',mask.shape)
    elif segmentation_type == hd.seg.SegmentationTypeValues.BINARY:
        print("Before", segmentation_array.shape)
        mask = np.argmax(segmentation_array, axis=2).astype(np.uint8)
        print("after", mask.shape)


    if create_pyramid:
        logging.info("Creating DICOM Segmentations")
        seg_start_time = time()
        segmentations = hd.seg.pyramid.create_segmentation_pyramid(
            source_images=source_images,
            pixel_arrays=[mask],
            segmentation_type=segmentation_type,
            segment_descriptions=segment_descriptions,
            series_instance_uid=hd.UID(),
            series_number=20,
            manufacturer=metadata_config.manufacturer,
            manufacturer_model_name=metadata_config.manufacturer_model_name,
            software_versions=metadata_config.software_versions,
            device_serial_number=metadata_config.device_serial_number,
            transfer_syntax_uid=transfer_syntax_uid,
            max_fractional_value=1,
            dimension_organization_type=dimension_organization_type,
            omit_empty_frames=omit_empty_frames,
            workers=workers,
        )
        seg_time = time() - seg_start_time
        logging.info(f"Created DICOM Segmentations in {seg_time:.1f}s.")
    else:
        logging.info("Creating DICOM Segmentation")
        seg_start_time = time()
        segmentation = hd.seg.Segmentation(
            source_images=source_images[0:1],
            pixel_array=mask,
            segmentation_type=segmentation_type,
            segment_descriptions=segment_descriptions,
            series_instance_uid=hd.UID(),
            series_number=20,
            sop_instance_uid=hd.UID(),
            instance_number=1,
            manufacturer=metadata_config.manufacturer,
            manufacturer_model_name=metadata_config.manufacturer_model_name,
            software_versions=metadata_config.software_versions,
            device_serial_number=metadata_config.device_serial_number,
            transfer_syntax_uid=transfer_syntax_uid,
            max_fractional_value=1,
            tile_pixel_array=True,
            dimension_organization_type=dimension_organization_type,
            omit_empty_frames=omit_empty_frames,
            workers=workers,
        )
        segmentations = [segmentation]
        seg_time = time() - seg_start_time
        logging.info(f"Created DICOM Segmentation in {seg_time:.1f}s.")

    return segmentations
#-----------------------------

# source image
#imagePath = '/media/clisle/KVisImagery/NCI/IDC/Oct2023_RMS_SamplesToIDC/RMS2397_source_level/3f40a518-bf63-40fd-bdae-4ee09e2d8865.dcm'
#outPath = '/media/clisle/KVisImagery/NCI/IDC/Oct2023_RMS_SamplesToIDC/PALMPL-0BMX5D-AKA-RMS2397/new_model_prediction_v2'
imagePath = '/home/clisle/proj/mhub/idc/input_data/image2/DCM_4.dcm'
outPath = '/home/clisle/proj/mhub/idc/output_data'

# open a numpy image file and write out the dicom equivalent.  This requires both 
# the original image (to get the header metadata) and the numpy (which came out from running the model)
#load_path = '/media/clisle/KVisImagery/NCI/IDC/Oct2023_RMS_SamplesToIDC/PALMPL-0BMX5D-AKA-RMS2397/new_model_prediction_v2'
#filename = '3f40a518-bf63-40fd-bdae-4ee09e2d8865.dcm_probability_seg.npy'
load_path = '/home/clisle/proj/mhub/idc/input_data/image2'
filename = 'DCM_4_prob_map_seg_stack.npy'

np_file_path = os.path.join(load_path, filename)
seg_image = np.load(np_file_path)

# create a mask that is exactly the same size as the source image. 
source_image_dataset = dcmread(str(imagePath))
source_rows = source_image_dataset.TotalPixelMatrixRows
source_cols = source_image_dataset.TotalPixelMatrixColumns
mask = np.zeros((source_rows, source_cols,5), np.float32)
for i in range(seg_image.shape[0]):
    for j in range(seg_image.shape[1]):
        mask[i,j,:] = seg_image[i,j,:]

# why does the image have to be the exact same size as the source image?  This is a bug?
segtype = 'BINARY'
#segtype = 'FRACTIONAL'
tiledtype = 'TILED_FULL'
# set true if multiresolution pyramid should be generated
multiresolution = False
segmentations = convert_segmentation([source_image_dataset], mask, multiresolution, segtype, tiledtype, 0)

# write out segmentations
for i,seg in enumerate(segmentations):
    outfilename = os.path.join(outPath,'seg_'+segtype+'_'+str(i)+'.dcm')
    seg.save_as(outfilename)

