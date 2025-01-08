import torch
import torch.nn as nn


class YOLOv1(nn.Module):
   
    
    def __init__(self, split_size, num_boxes, num_classes):
        
        super(YOLOv1, self).__init__()
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.darkNet = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3, stride=2, bias=False), # 3,448,448 -> 64,224,224
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2), # -> 64,112,112
            
            nn.Conv2d(64, 192, 3, padding=1, bias=False), # -> 192,112,112
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2), # -> 192,56,56
            
            nn.Conv2d(192, 128, 1, bias=False), # -> 192,56,56
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), # -> 256,56,56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 1, bias=False), # -> 256,56,56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False), # -> 512,56,56
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2), # -> 512,28,28
            
            nn.Conv2d(512, 256, 1, bias=False), # -> 256,28,28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False), # -> 512,28,28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1, bias=False), # -> 256,28,28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False), # -> 512,28,28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1, bias=False), # -> 256,28,28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False), # -> 512,28,28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1, bias=False), # -> 256,28,28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False), # -> 512,28,28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 1, bias=False), # -> 512,28,28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1, bias=False), # -> 1024,28,28
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2), # -> 1024,14,14
            
            nn.Conv2d(1024, 512, 1, bias=False), # -> 512,14,14
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1, bias=False), # -> 1024,14,14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1, bias=False), # -> 512,14,14
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1, bias=False), # -> 1024,14,14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False), # -> 1024,14,14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False), # -> 1024,14,14 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False), # -> 1024,14,14 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False), # -> 1024,14,14 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            )
        self.fc = nn.Sequential(
            nn.Linear(1024 * split_size * split_size, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, split_size * split_size * (num_classes + num_boxes*5)),
            nn.Sigmoid()
            )
        
        
    def forward(self, x):
        
        x = self.darkNet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = x.view(x.shape[0], self.split_size, self.split_size, 
                   self.num_boxes*5 + self.num_classes)
        return x