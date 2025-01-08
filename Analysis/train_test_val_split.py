import numpy as np
from random import randint
import random

user = 'gowtham'

if user == 'siddhi':
    path_videos = '/users/gowtham/downloads/majorproject/Data/Videos/'
    path_labels_csv = '/users/gowtham/downloads/majorproject/Data/labels_framewise_csv.csv'
    path_labels_list = '/users/gowtham/downloads/majorproject/Data/labels_framewise_list.pkl'
    path_frames = '/users/gowtham/downloads/majorproject/Data/Frames/'

x = np.arange(1, 105)
np.random.seed(42)
np.random.shuffle(x)
videos_validation = x[:16]
videos_test = x[16: 16+22]
videos_train = x[16+22: ]

print(videos_train, len(videos_train))
print(videos_test, len(videos_test))
print(videos_validation, len(videos_validation))