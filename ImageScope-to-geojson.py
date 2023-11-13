
# read ImageScope XML and write out GeoJSon
# KnowledgeVis, LLC 2023

import sys
import random
import numpy as np

# for the import process
import xml.etree.ElementTree as ET
import json
import sys
import os
from math import sqrt
import argparse

# needed to compile to an executable
#from charset_normalizer import md__mypyc


def translate_annotation_to_geojson(region_type):
    # Translate the Aperio type to a GeoJSON type
    match region_type:
        case '0':
            text_type = "polygon"
        case '1':
            text_type = "rectangle"
        case '2':
            text_type = "ellipse"
        case '4':
            text_type = "line"
        case '5':
            text_type = "point"
        case other:
            print('no match for annotation type ',region_type)
    return text_type

def translate_region_to_geojson(region_type):
    # Translate the Aperio type to a GeoJSON type
    match region_type:
        case '0':
            text_type = "Polygon"
        case '1':
            text_type = "Polygon"
        case '2':
            text_type = "Polygon"
        case '4':
            text_type = "LineString"
        case '5':
            text_type = "Point"
        case other:
            print('no match for region type ',region_type)   
    return text_type


def convert_coordinates_according_to_region_type(region,coords,length,area):

    # For an ellipse, we receive only the top left and bottom right corners and we need to calulate a
    # bbox for the ellipse, since the ImageServer stores ellipses by their bounding coordinates
    if region == 'ellipse':
        print('coords',coords)
        p1 = coords[0]
        p2 = coords[1]
        xdiff = abs(p2[0]-p1[0]); ydiff = abs(p2[1]-p1[1])
        center_list = [(p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0]
        center = np.array(center_list)
        # calculate the four corner points of the bbox 
        major_axis = xdiff /  2.0
        minor_axis = ydiff / 2.0
        print('semimajor length:',major_axis)
        print('semiminor length:',minor_axis)

        # assume Y=0 is on the top, since it is an image. Calculate the bbox centered around the 
        # center of the ellipse.  
        tl = [-major_axis,-minor_axis]
        bl = [-major_axis,minor_axis]
        tr = [major_axis,-minor_axis]
        br = [major_axis,minor_axis]
        tl_arr = np.array(tl)
        bl_arr = np.array(bl)
        tr_arr = np.array(tr)
        br_arr = np.array(br)
        tl_p = tl_arr + center
        bl_p = bl_arr + center
        tr_p = tr_arr + center
        br_p = br_arr + center
        # copy into the output array, duplicating the first point
        newcoords = [[[tl_p[0],tl_p[1]], [bl_p[0],bl_p[1]],
                    [br_p[0],br_p[1]],[tr_p[0],tr_p[1]],[tl_p[0],tl_p[1]]]]
    # points and lines use one less level of containing lists
    elif region == 'point' or region == 'line':
        newcoords = coords
    else:
        # other cases are enclosed in another list
        newcoords = [coords]
    return newcoords

# we found an XML annotation file.  Read it and add annotation to corresponding Image Item record
def perform_conversion(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    geojson = []
    annotationCount = 1

    # Iterate through each annotation
    for annotation in root.iter('Annotation'):
        # Iterate through each region within the current annotation
        for region in annotation.iter('Region'):
            region_text_type = translate_region_to_geojson(region.attrib['Type'])
            #annotation_text_type = translate_annotation_to_geojson(annotation.attrib['Type'])
            annotation_text_type = translate_annotation_to_geojson(region.attrib['Type'])
            # if the name is blank, fill in a placeholder 'annptation2', etc.
            if annotation_text_type == 'line':
                #annotationName =  annotation_text_type+' '+str(annotationCount)+': length '+ region.attrib['LengthMicrons']
                annotationName =  region.attrib['LengthMicrons'] + ' microns'
            else:
                # if the region has been given a name in the annotation file, preserve the name
                if len(region.attrib['Text'])>0:
                    annotationName =  region.attrib['Text']
                else:
                    annotationName =  annotation_text_type+' '+str(annotationCount)
            region_to_geometry = {}
            vertice_to_geometry_coordinates = []

            # Iterate through each vertex within the current region. Handle points as special case, since there
            # is one less level of lists for a point
            if annotation_text_type == 'point':
                for vertex in region.iter('Vertex'):
                    vertice_to_geometry_coordinates = [float(vertex.attrib['X']),float(vertex.attrib['Y'])]
            else:
                for vertex in region.iter('Vertex'):
                    vertice_to_geometry_coordinates.append(
                        [float(vertex.attrib['X']), float(vertex.attrib['Y'])]
                    )

            # depending on the region type, we need to adjust the coordinates (e.g. ellipse). Ellipse needs the length value
            adjusted_coordinates = convert_coordinates_according_to_region_type(annotation_text_type,vertice_to_geometry_coordinates,region.attrib['LengthMicrons'],region.attrib['AreaMicrons'])

            # Create a GeoJSON object for the current region
            region_to_geometry.update({
                "geometry": {
                    "coordinates": adjusted_coordinates,
                    "type": region_text_type
                },
                "properties": {
                    "annotationId": annotationCount,
                    "annotationType": annotation_text_type,
                    "fillColor": '#f'+annotation.attrib['LineColor'],
                    #"name": annotation.attrib['Name'],
                    "name":annotationName,
                },
                "type": "Feature"
            }
            )
            geojson.append(region_to_geometry)
            annotationCount += 1
    
    # trim the .xml from the filename
    xml_file = xml_file[:-4]
    # save the geojson to a file
    with open(f"{xml_file}.geojson", "w") as f:
        # print(json.dumps(geojson, indent=4))
        json.dump(geojson, f, indent=4)
    


if __name__ == "__main__":
  
    # define a command line parser
    parser = argparse.ArgumentParser(
                    prog='imageScope-convert',
                    description='converts ImageScope XML annotations ',
                    epilog='')
    
    parser.add_argument('-d', '--directory',help='specify an absolute pathname to a directory containing images and annotations') 
   
    args = parser.parse_args()

    if args.directory:
        # Get the path to the files to convert
        path = args.directory
        uploadEnabled = True
    else:
        path = None
        print('please enter a directory with the -d or --directory options')
        uploadEnabled = False

    # Get all the xml files from within a path and it's descendants
    convertCount = 0
    if uploadEnabled:
        print('upload enabled')
        print('path',path)
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.xml'):
                    try:
                        perform_conversion(os.path.join(root, file))
                        convertCount += 1
                    except:
                        pass

        print('')
        print('Job has finished.',convertCount,' annotations were converted')

    else:
        print('** No files were converted. Please review the command line options')
        print('If needed, please use the -h flag for help')
