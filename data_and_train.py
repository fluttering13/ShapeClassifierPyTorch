import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from train.data_loader import *
import pickle



class SimpleCNN(nn.Module):
    def __init__(self, image_height, image_width, number_classes):
        super(SimpleCNN, self).__init__()
        #batch_size
        fc_input_size = 32 * (image_height // 4) * (image_width // 4) #two maxPool://4 and second conv channel: 32 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, number_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x) #out: (batch_size, 16, cov_height, cov_width)
        x = self.batchnorm1(x) 
        x = self.relu1(x)
        x = self.pool1(x) #out: (batch_size, 16, cov_height/2, cov_width/2)

        x = self.conv2(x) #out: (batch_size, 32, cov_height/2, cov_width/2)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.pool2(x) #out: (batch_size, 32, cov_height/4, cov_width/4)

        x = x.squeeze()
        fc_input_size = 32 * (x.size(2)) * (x.size(3))
        x = x.view(-1, fc_input_size) # out: (batch_size, 32 * (image_height // 4) * (image_width // 4))

        x = F.relu(self.fc1(x)) #out: (batch_size, 128)

        x = self.fc2(x)
        
        return x
def validate_model(model, criterion, val_loader, device):
    model.eval()  
    val_loss = torch.tensor(0.0)
    correct = 0
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            if device!=torch.device('cpu'):
                images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            val_loss += criterion(outputs, labels).item() * images.size(0)
            _, predicted = torch.max(outputs, 1) #get predictions from the final layer

            correct += (predicted == labels).sum().item()
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())    

    val_loss /= len(val_loader.dataset)
    accuracenter_y = correct / len(val_loader.dataset) * 100

    confusionMatrix = confusion_matrix(true_labels, predicted_labels)
    return val_loss, accuracenter_y, confusionMatrix    

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, checkpoint_interval, device, result_save_name):
    os.makedirs('checkpoints', exist_ok=True)#make directory of checkpoint folder
    epoch_list=[]
    training_loss_list=[]
    validation_loss_list=[]
    val_acc_list=[]
    confusion_matrix_list=[]

    for epoch in range(num_epochs):# begin training
        epoch_list.append(epoch)
        model.train()  
        running_loss = 0.0 # the training loss summing over the batches in training 

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  
            outputs = model(images)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        training_loss_list.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Validate the model
        if (epoch + 1) % checkpoint_interval == 0: # save and validate the model per 2 epoch
            val_loss, val_acc, confusion_matrix = validate_model(model, criterion, val_loader, device)
            print(f"Validation Loss: {val_loss:.4f}, Accuracenter_y: {val_acc:.2f}%")
            print(f'confusion_matrix:\n {confusion_matrix}')
            # Save model checkpoint
            checkpoint_path = f"checkpoints/{result_save_name}_model_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
            validation_loss_list.append(val_loss.cpu().numpy())
            val_acc_list.append(val_acc)
            confusion_matrix_list.append(confusion_matrix)

    result_dict={'epoch_list':epoch_list,
                 'training_loss_list':training_loss_list,
                 'validation_loss_list':validation_loss_list,
                 'val_acc_list':val_acc_list,
                 'cm_list':confusion_matrix_list
                 } # the summary dict
    with open('./checkpoints/'+result_save_name+'.pkl', 'wb') as f:
        pickle.dump(result_dict, f)
    return result_dict

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_height =200  
    image_width= 200
    batch_size=64
    num_epochs = 10
    checkpoint_interval = 2

    folder_path='./pic'
    number_classes=len([f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))])

    model = SimpleCNN(image_height,image_width,number_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    ### read if intestsed https://arxiv.org/abs/1711.05101
    ### decoupled weight decay is wildly useful in different framework.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader=exp_data_loader(image_height, image_width, batch_size)
    result_dict=train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, checkpoint_interval, device, result_save_name='simple_dataset')
