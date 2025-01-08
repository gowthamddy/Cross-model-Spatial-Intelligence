import os
import time
import torch
import argparse
import torch.optim as optim
from model import YOLOv9
from loss import YOLO_Loss
from dataset import DataLoader
from utils import load_checkpoint, save_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("-tip", "--train_img_files_path", default="majorproject/data/videos/", 
                help="majorproject/images")
ap.add_argument("-ttp", "--train_target_files_path", 
                default="majorproject/labels.json", 
                help="Path to JSON file containing the train labels")
ap.add_argument("-lr", "--learning_rate", default=1e-5, help="Learning rate")
ap.add_argument("-bs", "--batch_size", default=10, help="Batch size")
ap.add_argument("-ne", "--number_epochs", default=100, help="Number of epochs")
ap.add_argument("-nb", "--number_boxes", default=2, help="Number of bounding boxes to predict")
ap.add_argument("-lc", "--lambda_coord", default=5, help="Hyperparameter for bounding box loss")
ap.add_argument("-ln", "--lambda_noobj", default=0.5, help="Hyperparameter for no object loss")
ap.add_argument("-lm", "--load_model", default=1, help="1 to load model weights, else 0")
ap.add_argument("-lmf", "--load_model_file", default="YOLOv9_weights.pt", 
                help="File containing the model weights")
args = ap.parse_args()

learning_rate = float(args.learning_rate)
batch_size = int(args.batch_size)
num_epochs = int(args.number_epochs)
num_boxes = int(args.number_boxes)
lambda_coord = float(args.lambda_coord)
lambda_noobj = float(args.lambda_noobj)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLOv9(num_boxes=num_boxes).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

if int(args.load_model):
    load_checkpoint(torch.load(args.load_model_file), model, optimizer)

def train_network(num_epochs, model, device, optimizer):
    model.train()
    data_loader = DataLoader(args.train_img_files_path, args.train_target_files_path, batch_size)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for img_data, target_data in data_loader:
            img_data, target_data = img_data.to(device), target_data.to(device)
            optimizer.zero_grad()
            predictions = model(img_data)
            loss = YOLO_Loss(predictions, target_data, lambda_coord, lambda_noobj)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(data_loader):.4f}')
        save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, 
                        filename=args.load_model_file)

        time.sleep(1)

def main():
    print("Starting training...")
    train_network(num_epochs, model, device, optimizer)

if __name__ == "__main__":
    main()