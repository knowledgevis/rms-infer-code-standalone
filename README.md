# rms-infer-code-standalone

python scripts for RMS model inference. Working on DICOM export.

ImageScope-to-geojson.py - this file reads a directory containing XML annotations in ImageScope format and converts the annotations into geojson. 

rms-seg-with-dicom-output.py - this application processes a single H&E image in DICOM WSI format and generates a multi-class binary segmentation. 

rms-seg-fractional-dicom.py - This code is based on rms-seg-with-dicom-output.py and changes only the output function to create a fractional 
segmentaiton object.  This worked for a single class but it did not work for more than one output class at the time of writing.  
