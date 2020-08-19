# Here we have imported the required directory.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
from time import sleep
import pickle
from parameters import *
import cv2


# 3. The main function.
def main(args):
# Output directory as we have passed in argumnets.
    output_dir = os.path.expanduser(args.output_dir)
    path_dict = {}
# If the output directory not exist, then a new directory will be created.    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# It extract the source path of file.   
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)

    print('Creating networks and loading parameters')
# Tensorflow works as graph, and its uses gpu. Here the key is that we have to pass that how much of gpu we have to use for same, by default its uses 100% gpu.
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.compat.v1.Session()
        with sess.as_default():
# It extract pnet, rnet, onet by the help of mtcnn.        
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

# minsize meant that it will only detect the face atleat size of 20
# threshold and factor are the parameter for better detecting the face in detect_face() function below.
    minsize = 20 
    threshold = [ 0.6, 0.7, 0.7 ] 
    factor = 0.709 

# Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
# Here we are defining the name of text file for the entry of images with an random key for each by the help random_key.
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

# Here we are opening that writable text file we made at line 52
    with open(bounding_boxes_filename, "w") as text_file:
# Here we total numbers of images and aligned images as 0.
        nrof_images_total = 0
        nrof_successfully_aligned = 0
# If we passed the shuffle argument in terminal dataset image will be shuffled.        
        if args.random_order:
            random.shuffle(dataset)
# This for loop runs for each class in the dataset (for each individual person in dataset). This loop will run for 7 times because we have data of 7 different person in our case.           
        for cls in dataset:
# This will join the output directory with the class name.        
            output_class_dir = os.path.join(output_dir, cls.name)
# This will make the path dictionary of output class directory.            
            path_dict[cls.name] = output_class_dir
            
# If the output class directory not exist, then a new directory will be created.            
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
# Same as line 59.                
                if args.random_order:
                    random.shuffle(cls.image_paths)
                    
# This loop will run of every single image in a single class.                    
            for image_path in cls.image_paths:
                nrof_images_total += 1
# This will extract the file from the image path.
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
# This will create output filename with .png file format.                
                output_filename = os.path.join(output_class_dir, filename+'.png')
                print(image_path)
                
                
# For every image file which exist.                
                if not os.path.exists(output_filename):
# This is a try-catch exception, if image is not readble by imread() function then it throw an exception message.                
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
# Here it will check for colour images, if it is true the good. But if image dimention is less the 2 then print message else convert it to rgb.                      
                    else:
                        if img.ndim<2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
# Now this is colour image.                            
                        img = img[:,:,0:3]

# This will detect the face with the help of dectect_face() function.
                        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

# This will count the total numbers face in single image.                        
                        nrof_faces = bounding_boxes.shape[0]
# This will carry-on only if atleast 1 face should be their.                        
                        if nrof_faces>0:
# This will create a variable det which stores..........                        
                            det = bounding_boxes[:,0:4]
                            
# Empty detection list.                            
                            det_arr = []
                            
# Image size on the basis of image shape.                            
                            img_size = np.asarray(img.shape)[0:2]
# If their will faces more than 1, condition passed.                            
                            if nrof_faces>1:
# If we have passed the detect_multiple_faces argument in terminal then, it will go for for loop.                           
                                if args.detect_multiple_faces:

# This loop will run for count of total number of faces, then it will append the face array of each face in detection list after removing single dimentional. This dimention removal is done by numpy squeez() function.                                
                                    for i in range(nrof_faces):
                                        det_arr.append(np.squeeze(det[i]))
                                else:
                                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                    det_arr.append(det[index,:])
# This will run only when image has 1 face in it, then it will directly append the face array to the detection list after removing single dimentional. This dimention removal is done by numpy squeez() function.                                                          
                            else:
                                det_arr.append(np.squeeze(det))



                            for i, det in enumerate(det_arr):
                                det = np.squeeze(det)
                                bb = np.zeros(4, dtype=np.int32)
# These are the 4 point corner of face.                                
                                bb[0] = np.maximum(det[0]-args.margin/2, 0)
                                bb[1] = np.maximum(det[1]-args.margin/2, 0)
                                bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                                bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
# The face is cropped from img image.                                
                                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
# Then the image is scaled according to the requirment of our keras pre-trained model.                                
                                scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
# This will make the increase in count of number of successfully alinged faces by 1.                                
                                nrof_successfully_aligned += 1
                                
# It will extract the filename and extension by the help of output filename.                       
                                filename_base, file_extension = os.path.splitext(output_filename)
# If the detect_multiple_faces argument is passed by terminal and image has more then 1 image then filename will alloted with extension and name will end with integer value.                                  
                                if args.detect_multiple_faces:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                    
# Else directly output file name will be generated with extension.                                    
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
# This will save the image on the device in given directory.                                    
                                misc.imsave(output_filename_n, scaled)
# This will make the entry on the text file.                                
                                text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                        
# This else case will run if image has no face with print message. And write the entry on the text file.                        
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))

# This will print total number of images and total numbers of alined images.
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

# This will save the path dictionary as pickle file named as path_dict.p.
    with open('path_dict.p', 'wb') as f:
        pickle.dump(path_dict, f)
# Code ends here.




# 2. Here we pass the input and output directory as arguments in command line for now. But we pass many more arguments like image size, margin, gpu memory fraction.
# Example on command line -  python align_dataset_mtcnn.py ./input_directory ./output_directory
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=IMAGE_SIZE)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order',
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)


# 1. The code begins from here, its ask for arguments by parse_arguments() function.
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
