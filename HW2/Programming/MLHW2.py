'''********************************************* 
  Import packages
 *********************************************'''
import os
import random
import glob
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.optim import Adam
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import argparse
from tqdm import trange
import wandb
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix


"""********************************************* 
  Self-defined
 *********************************************"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='Total training epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=128, help='Batch size for validation.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')

    parser.add_argument('--weight_d', type=float, default=1e-2, help='Adjust weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')  #'./model/1025_4_4Level/epoch30_acc0.6753.pth'
    parser.add_argument('--mode', type=str, default='train', help='train or test mode')
    parser.add_argument('--gamma', type=float, default=0.5, help='Initial gamma for scheduler and the default is 0.8.')
    parser.add_argument('--step', type=int, default=20, help='Initial step for scheduler and the default is 10.')
    parser.add_argument('--channel_num', type=int, default=64, help='Revise channel quantity in network')

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('-p', '--plot_cm', action='store_true', help='Ploting confusion matrix.')
    parser.add_argument('--data_aug', action='store_true', help='Use data augmentation or not.')
    parser.add_argument('--early_stop', action='store_true', help='Use early stopping technique or not.')
    return parser.parse_args()

def wandb_update():
    config = wandb.config
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.val_batch_size = args.val_batch_size
    config.optimizer = args.optimizer

    config.weight_d = args.weight_d
    config.momentum = args.momentum
    config.checkpoint = args.checkpoint
    config.gamma = args.gamma
    config.step = args.step
    config.channel_num = args.channel_num

def plot_confusion_matrix(cm, labels_name, title, acc):
    cm = cm / cm.sum(axis=1)[:, np.newaxis]  # Normalization
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "white", fontsize = 'large')
    plt.imshow(cm, interpolation='none')  # Show image nearest on specific window
    plt.title(title)  # Image Title
    plt.colorbar()
    num_class = np.array(range(len(labels_name)))  # Get interval number of labels
    plt.xticks(num_class, labels_name, rotation=45, fontsize=16)  # Print label on x-axis
    plt.yticks(num_class, labels_name, fontsize=16)  # Print label on y-axis
    plt.ylabel('Target')
    plt.xlabel('Prediction')
    plt.imshow(cm, interpolation='none', cmap=plt.get_cmap('Blues'))
    # plt.tight_layout()
    plt.savefig(os.path.join('./Confusion_matrix', title + "_acc" + str(round(acc, 4)) + ".png"), format='png')
    plt.show()


"""********************************************* 
  Set arguments and random seed
 *********************************************"""
TRA_PATH = './dataset/train/'
TST_PATH = './dataset/test/'
LABEL_PATH = './dataset/train.csv'
DEVICE_ID = 0
SEED = 5566
args = parse_args()
NUM_ECPOCH = args.epochs


torch.cuda.set_device(DEVICE_ID)
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)


"""********************************************* 
  Process data
 *********************************************"""
def load_train_data(img_path, label_path, valid_ratio=0.12):
    train_label = pd.read_csv(label_path)['label'].values.tolist()
    train_image = [f'{img_path}/{i+10000}.jpg' for i in range(len(train_label)-1)]
    
    train_data = list(zip(train_image, train_label))
    random.shuffle(train_data)
    
    split_len = int(len(train_data) * valid_ratio)
    train_set = train_data[split_len:]
    valid_set = train_data[:split_len]
    
    return train_set, valid_set

def load_test_data(img_path):
    test_set = [f'{img_path}/{i}.jpg' for i in range(7000, 10000)]
    return test_set
    
def compute_statistics(dataset):
    data = []
    for (img_path, label) in dataset:
        data.append(np.array(Image.open(img_path)))
    data = np.array(data)
    return data.mean(), data.std()

train_set, valid_set = load_train_data(TRA_PATH, LABEL_PATH)
test_set = load_test_data(TST_PATH)

'''Aumentation'''
transform_aug = None
if args.data_aug:
    transform_set = [
        transforms.RandomHorizontalFlip(p=0.5),   # Horizontal Flip in random
        # transforms.RandomVerticalFlip(p=0.5),   # Vertical Flip in random
        transforms.ColorJitter(brightness=(0, 5), contrast=(0, 5), saturation=(0, 5), hue=(-0.1, 0.1)),  # Adjust image brightness, contrast, satuation and hue in random
        transforms.RandomRotation(30, center=(0, 0), expand=False),]   # expand only for center rotation
    # size = 48
    # transform_aug = transforms.Compose([
    #     transforms.RandomChoice(transform_set),
    #     transforms.CenterCrop(size),     # Cutting image by original center to a fitting size
    #     transforms.Pad((64 - size)//2, fill=0, padding_mode="constant"),])
    transform_aug = transforms.Compose([
        transforms.RandomChoice(transform_set),
        transforms.Resize(224)])



"""********************************************* 
  Customize dataset
 *********************************************"""
class FaceExpressionDataset(Dataset):
    def __init__(self, data, augment=None):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)
    
    def normalize(self, data):
        # TODO: do normalization there
        transform_nor = transforms.Normalize(mean=0.5085, std=0.2644)
        return transform_nor(data)
    
    def read_img(self, idx):
        img = Image.open(self.data[idx][0])
        if not self.augment is None:
            img = self.augment(img)
        img = torch.from_numpy(np.array(img)).float()
        img = img.unsqueeze(0).float()
        img = self.normalize(img)
        return img
    
    def __getitem__(self, idx):
        img = self.read_img(idx)
        label = self.data[idx][1]
        return img, label
    
class TestingDataset(Dataset):
    def __init__(self, data, augment=None):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)
    
    def normalize(self, data):
        # TODO: do normalization there
        transform_nor = transforms.Normalize(mean=0.5085, std=0.2644)
        return transform_nor(data)
    
    def read_img(self, idx):
        img = Image.open(self.data[idx])
        if not self.augment is None:
            img = self.augment(img)
        img = torch.from_numpy(np.array(img)).float()
        img = img.unsqueeze(0).float()
        img = self.normalize(img)
        return img, self.data[idx].split('/')[-1][:-4]
        
    def __getitem__(self, idx):
        img, name = self.read_img(idx)
        
        return img, name

train_dataset = FaceExpressionDataset(train_set, transform_aug)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

valid_dataset = FaceExpressionDataset(valid_set)
valid_loader = DataLoader(valid_dataset, batch_size=args.val_batch_size, shuffle=False)

test_dataset = TestingDataset(test_set)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


"""********************************************* 
  Define module class
 *********************************************"""
class FaceExpressionNet(nn.Module):
    def __init__(self, n_chansl):
        super(FaceExpressionNet, self).__init__()
        # TODO
        self.conv_0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, affine=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d((2, 2)),
        )
        
        self.conv_4layer = nn.Sequential(
            nn.Conv2d(1, n_chansl, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_chansl, eps=1e-05, affine=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(n_chansl, n_chansl*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_chansl*2, eps=1e-05, affine=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(n_chansl*2, n_chansl*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_chansl*4, eps=1e-05, affine=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(n_chansl*4, n_chansl*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_chansl*2, eps=1e-05, affine=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d((2, 2)),
        )
        self.fc_4layer = nn.Sequential(
            nn.Linear(n_chansl*2 * 4 * 4, 7),
        )

        self.conv_3layer = nn.Sequential(
            nn.Conv2d(1, n_chansl, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_chansl, eps=1e-05, affine=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d((2, 2)),   # (Batch_size, 32, 32, 32)->(B, C, H, W)

            nn.Conv2d(n_chansl, n_chansl//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_chansl//2, eps=1e-05, affine=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d((2, 2)),   # (Batch_size, 64, 16, 16)->(B, C, H, W)

            nn.Conv2d(n_chansl//2, n_chansl//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_chansl//4, eps=1e-05, affine=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d((2, 2)),   # (Batch_size, 128, 8, 8)->(B, C, H, W)
        )
        self.fc_3layer = nn.Sequential(
            nn.Linear(n_chansl//4 * 8 * 8, 7),
        )

    def forward(self, x):
        # #image size (64,64) -> Shape: [Batch_size, 1, 64, 64]
        # x = self.conv_3layer(x)
        # x = x.flatten(start_dim=1)  # Shape: [Batch_size, channel_num*8*8]->[B, C*H*W]
        # x = self.fc_3layer(x)  # Shape: [Batch_size, 7]

        x = self.conv_4layer(x)
        x = x.flatten(start_dim=1)  # Shape: [Batch_size, channel_num*4*4]->[B, C*H*W]
        x = self.fc_4layer(x)  # Shape: [Batch_size, 7]
        return x


"""********************************************* 
  Define training and testing process
 *********************************************"""
def train(train_loader, model, loss_fn, scheduler, use_gpu=True):
    model.train()
    train_loss = []
    train_acc = []
    for (img, label) in train_loader:
        if use_gpu:
            img = img.to(device)
            label = label.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, label)
        loss.backward()            
        optimizer.step()
        with torch.no_grad():
            predict = torch.argmax(output, dim=-1)
            acc = np.mean((label == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())
    print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))

    if args.wandb:
        wandb.log({"lr": optimizer.param_groups[0]['lr'],
                   "train_loss": np.mean(train_loss),
                   "train_acc": np.mean(train_acc),})
    scheduler.step()
    
def valid(valid_loader, model, loss_fn, use_gpu=True, mode=None):
    model.eval()
    with torch.no_grad():
        valid_loss = []
        valid_acc = []
        pre_labels = []  # For plotting confusion matrix
        gt_labels = []  # For plotting confusion matrix
        for idx, (img, label) in enumerate(valid_loader):
            gt_labels += label.tolist()   # For plotting confusion matrix
            if use_gpu:
                img = img.to(device)
                label = label.to(device)
            output = model(img)
            loss = loss_fn(output, label)
            predict = torch.argmax(output, dim=-1)
            acc = (label == predict).cpu().tolist()
            valid_loss.append(loss.item())
            valid_acc += acc
            pre_labels += predict.cpu().tolist()  # For plotting confusion matrix
       
        valid_acc = np.mean(valid_acc)
        valid_loss = np.mean(valid_loss)
        if mode != 'val':
            print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, valid_loss, valid_acc))
        else:
            print("valid Loss: {:.4f}, valid Acc: {:.4f}".format(valid_loss, valid_acc))

        if args.wandb:
            wandb.log({"val_loss": valid_loss, "val_acc": valid_acc})
    return valid_acc, valid_loss, gt_labels, pre_labels

def test(test_loader, model, file_name='./testing_result/predict.csv'):
    with torch.no_grad():
        predict_result = []
        predict_name = []
        for img, name in test_loader:
            if use_gpu:
                img = img.to(device)
            output = model(img)
            predict = torch.argmax(output, dim=-1).tolist()
            predict_result += predict
            predict_name += name
        
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for id, r in zip(predict_name, predict_result):
            writer.writerow([id, r])

def save_checkpoint(valid_acc, acc_record, epoch, optimizer_state_dict, prefix):
    # you can define the condition to save model :)
    if valid_acc >= np.mean(acc_record[-5:]):
        torch.save({'iter': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_state_dict},
                 os.path.join(prefix, "res18_epoch" + str(epoch) + "_acc" + str(round(valid_acc, 4)) + ".pth"))
        print('model saved...')

def better(acc_record):
    if max(acc_record) == acc_record[-1]: return True
    return False

if __name__ == '__main__':
    if args.mode == 'train':
        '''Wandb Prepare'''
        if args.wandb:
            wandb.init(project='MLHW2')
            wandb_update()
        

        '''Model & Param Prepare'''
        # model = FaceExpressionNet(n_chansl=args.channel_num)
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if args.checkpoint:
            print("Loading pretrained weights...", args.checkpoint)
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if use_gpu:
            model.to(device)


        '''Optim Prepare'''
        params = model.parameters()
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, weight_decay=args.weight_d, lr=args.lr)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weight_d)
        else:
            raise ValueError("Optimizer not supported.")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
        loss_fn = nn.CrossEntropyLoss()
        

        '''Start Training'''
        acc_record = [0.62] # The initial acc is 0.55, so it must greater than 0.55 before saving
        the_last_loss = 100
        patience = 5    # If early-stopping trigger time >= patience, then stop training
        trigger_times = 0
        for epoch in trange(NUM_ECPOCH):
            train(train_loader, model, loss_fn, scheduler, use_gpu)
            valid_acc, valid_loss, _, _ = valid(valid_loader, model, loss_fn, use_gpu=True)
            acc_record.append(valid_acc)
            

            # Early-Stopping
            if args.early_stop:
                if valid_loss > the_last_loss:
                    trigger_times += 1
                    print('trigger times:', trigger_times)

                    if trigger_times >= patience:
                        print('Early stopping!\nStart to test process.')
                        break

                else:
                    print('trigger times: 0')
                    trigger_times = 0

                the_last_loss = valid_loss


            if better(acc_record):
                save_checkpoint(valid_acc, acc_record, epoch, optimizer.state_dict(), prefix='./model')
            
            print('########################################################')
        del model
        
    '''Plot Confusion Matrix'''
    if args.plot_cm and args.mode=='val':
        model = FaceExpressionNet(n_chansl=args.channel_num)
        model.load_state_dict(torch.load(args.checkpoint)["model_state_dict"], strict=False)
        model = model.cuda()
        loss_fn = nn.CrossEntropyLoss()
        if use_gpu:
            model.to(device)
        valid_acc, valid_loss, gt_labels, pre_labels = valid(valid_loader, model, loss_fn, use_gpu=True, mode='val')

        labels_name = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        cm_face = confusion_matrix(gt_labels, pre_labels)
        cm_face = np.array(cm_face)
        plot_confusion_matrix(cm_face, labels_name, 'MLHW2', valid_acc)


    if args.mode == 'test':
        model = FaceExpressionNet(n_chansl=args.channel_num)
        model.load_state_dict(torch.load(args.checkpoint)["model_state_dict"], strict=False)
        model = model.cuda()
        test(test_loader, model)