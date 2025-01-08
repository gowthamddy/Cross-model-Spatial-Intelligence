import torch
import torch.nn as nn


class YOLOv9(nn.Module):
    
    def __init__(self, split_size, num_boxes, num_classes):
       
        
        super(YOLOv9, self).__init__()
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.darkNet = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3, stride=2, bias=False), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(64, 192, 3, padding=1, bias=False), 
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(192, 128, 1, bias=False), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 1, bias=False), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(512, 256, 1, bias=False), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1, bias=False), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1, bias=False), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1, bias=False), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 1, bias=False), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1, bias=False), 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(1024, 512, 1, bias=False), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1, bias=False), 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1, bias=False), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1, bias=False), 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False), 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False), 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False), 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False), 
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