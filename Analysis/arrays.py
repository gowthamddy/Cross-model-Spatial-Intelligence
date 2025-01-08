
import glob
import natsort
from imageai.Detection import VideoObjectDetection

import numpy as np
import os
import sys



def detection_of_vehicles_from_video(folder1,folder2,findex):

    def forFrame(frame_number, output_array, output_count):
            
            bboxes = []
            
            for i in range(len(output_array)):
                bboxes.append(list(output_array[i]['box_points']))
                
            B.append(bboxes)
    
    videos = glob.glob(folder1+'/video*.MOV')
    videos = natsort.natsorted(videos)

    execution_path = os.getcwd()
    detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet() 
    detector.setModelPath(os.path.join(execution_path,"/users/gowtham/downloads/majorproject/Data/resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    custom_objects = detector.CustomObjects(bicycle=True, motorcycle=True,car=True,truck=True)


    for video in videos:
        print('processing' + video )
        B = []
        detector.detectCustomObjectsFromVideo(
            save_detected_video=False,
            custom_objects = custom_objects,
            input_file_path=os.path.join(execution_path, video),
            frames_per_second=30,
            per_frame_function=forFrame,
            minimum_percentage_probability=40)
        B = np.array(B)
        print('saving array for video' + video + '\n shape of array: ' + str(B.shape))
        np.save(folder2+'/array'+str(findex),B)
        findex = findex + 1

detection_of_vehicles_from_video('/users/gowtham/downloads/majorproject/results',77)

print('saved arrays for videos_new')

