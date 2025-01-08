import cv2
import os
import pickle
import numpy as np

user = 'gowtham'

if user == 'gowtham':
    path_videos = '/users/gowtham/downloads/majorproject/Data/Videos/'
    path_labels_csv = '/users/gowtham/downloads/majorproject/Data/labels_framewise_csv.csv'
    path_labels_list = '/users/gowtham/downloads/majorproject/Data/labels_framewise_list.pkl'
    path_frames = '/users/gowtham/downloads/majorproject/Data/Frames/'

video_ids = list(range(1,105))

for id in video_ids:

    cam = cv2.VideoCapture(path_videos + "video" + str(id) + ".MOV")

    try: 
      
        if not os.path.exists(path_frames + "/video" + str(id)): 
            os.makedirs(path_frames + "/video" + str(id)) 
    
    except OSError: 
        print ('Error: Creating directory of data')

    currentframe = 0
    print("starting " + path_videos + "video" + str(id) + ".MOV")
    while(True): 
        
        ret,frame = cam.read() 
    
        if ret: 
            name = path_frames + "/video" + str(id) + "/frame" + str(currentframe) + '.jpg'
            #print ('Creating...' + name) 
    
            cv2.imwrite(name, frame) 
    
            currentframe += 1
        else: 
            break
    
    cam.release() 
    cv2.destroyAllWindows() 


def get_labels_from_video(no_frames, safe_duration_list):

    labels = [0]*no_frames
    no_safe_durations = int(len(safe_duration_list)/2)
    if(no_safe_durations == 0):
        return labels,-1
    else:

        for i in range(no_safe_durations):
            safe_start = max(safe_duration_list[i*2] - 1, 0)
            safe_end = min(safe_duration_list[i*2 +1] - 1, no_frames-1)
            labels[safe_start:safe_end+1] = [1]*(safe_end-safe_start+1) # marking the value b/w safe_start and safe_end with 1

    if len(labels) > no_frames: 
        raise Exception('Check the labels assigned in CSV file!')
    return labels,1


open_file = open(path_labels_list, "rb")
labels_list = pickle.load(open_file)
open_file.close()
print(len(labels_list))
video_ids = list(range(1, 105))

for id in video_ids:

    video = path_videos + "video" + str(id) + ".MOV"

    print("starting " + video)
    cap = cv2.VideoCapture(video)
    no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    l, f = get_labels_from_video(no_frames, labels_list[id-1])
    print(len(l))

    labels = np.array(l)
    name = path_frames + "video" + str(id) + "/labels" + str(id) + ".npy"

    np.save(name, labels)
