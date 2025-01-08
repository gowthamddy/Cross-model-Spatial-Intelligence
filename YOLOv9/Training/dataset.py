import torch
from torchvision import transforms
from os import listdir
from PIL import Image
import json
import random


class DataLoader():
    

    def __init__(self, img_files_path, target_files_path, category_list, split_size, 
                 batch_size, load_size):
        
        self.img_files_path = img_files_path
        self.target_files_path = target_files_path       
        self.category_list = category_list        
        self.num_classes = len(category_list)       
        self.split_size = split_size        
        self.batch_size = batch_size      
        self.load_size = load_size
        
        self.img_files = [] 
        self.target_files = [] 
        
        self.data = [] 
        self.img_tensors = [] 
        self.target_tensors = [] 
        self.transform = transforms.Compose([
            transforms.Resize((448,448), Image.NEAREST),
            transforms.ToTensor(),
            ])
    

    def LoadFiles(self):          
        self.img_files = listdir(self.img_files_path)
        f = open(self.target_files_path)
        self.target_files = json.load(f)
        
        
    def LoadData(self):
        self.data = []    
        self.img_tensors = [] 
        self.target_tensors = [] 

        for i in range(len(self.img_files)):
            if len(self.img_tensors) == self.batch_size:
                self.data.append((torch.stack(self.img_tensors), 
                                  torch.stack(self.target_tensors)))
                self.img_tensors = []
                self.target_tensors = []
                print('Loaded batch ', len(self.data), 'of ', self.load_size)
                print('Percentage Done: ', round(len(self.data)/self.load_size*100., 2), '%')
                print('')
            
            if len(self.data) == self.load_size: 
                break 
            self.extract_image_and_label() 


    def extract_image_and_label(self):
       
        img_tensor, chosen_image = self.extract_image()
        target_tensor = self.extract_json_label(chosen_image)
        
        if target_tensor is not None: # Checks if the label contains any data
            self.img_tensors.append(img_tensor)
            self.target_tensors.append(target_tensor)
        else:
            print("No label found for " + chosen_image) # Log the image without label
            print("")

        
    def extract_image(self):   
       
        f = random.choice(self.img_files)
        self.img_files.remove(f)
        
        global img
        img = Image.open(self.img_files_path + f)
        img_tensor = self.transform(img) # Apply the transform to the image.
        return img_tensor, f


    def extract_json_label(self, chosen_image):
       
        for json_el in self.target_files:
            if json_el['name'] == chosen_image:
                img_label = json_el
                if img_label["labels"] is None: # Checks if a label exists for the given image
                    break
                target_tensor = self.transform_label_to_tensor(img_label)
                return target_tensor

        print("No label found for " + chosen_image) # Log the image without label
        print("")


    def transform_label_to_tensor(self, img_label):
        target_tensor = torch.zeros(self.split_size, self.split_size, 5+self.num_classes)

        for labels in range(len(img_label["labels"])):
            category = img_label["labels"][labels]["category"]         
            if category not in self.category_list:
                continue
            ctg_idx = self.category_list.index(category)

            x1 = img_label["labels"][labels]["box2d"]["x1"] * (448/img.size[0])
            y1 = img_label["labels"][labels]["box2d"]["y1"] * (448/img.size[1])
            x2 = img_label["labels"][labels]["box2d"]["x2"] * (448/img.size[0])
            y2 = img_label["labels"][labels]["box2d"]["y2"] * (448/img.size[1])

            x_mid = abs(x2 - x1) / 2 + x1
            y_mid = abs(y2 - y1) / 2 + y1
            width = abs(x2 - x1) 
            height = abs(y2 - y1) 

            cell_dim = int(448 / self.split_size)

            cell_pos_x = int(x_mid // cell_dim)
            cell_pos_y = int(y_mid // cell_dim)

            if target_tensor[cell_pos_y][cell_pos_x][0] == 1:
                continue

            target_tensor[cell_pos_y][cell_pos_x][0] = 1
            target_tensor[cell_pos_y][cell_pos_x][1] = (x_mid % cell_dim) / cell_dim
            target_tensor[cell_pos_y][cell_pos_x][2] = (y_mid % cell_dim) / cell_dim
            target_tensor[cell_pos_y][cell_pos_x][3] = width / 448
            target_tensor[cell_pos_y][cell_pos_x][4] = height / 448
            target_tensor[cell_pos_y][cell_pos_x][ctg_idx+5] = 1

        return target_tensor