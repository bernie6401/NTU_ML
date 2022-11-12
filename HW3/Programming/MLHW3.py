'''********************************************* 
  Import packages
 *********************************************'''
import csv
import time
import sys
import os
import random

# other library
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PyTorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data 
from torchvision import transforms
# from tqdm.notebook import tqdm

# Self-defined
import argparse
from tqdm import trange
import wandb
# import matplotlib.pyplot as plt
# import itertools
# from sklearn.metrics import confusion_matrix


"""********************************************* 
  Self-defined
 *********************************************"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Total training epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate for sgd.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')

    parser.add_argument('--weight_d', type=float, default=1e-5, help='Adjust weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--mode', type=str, default='train', help='train or test mode')
    parser.add_argument('--gamma', type=float, default=0.5, help='Initial gamma for scheduler and the default is 0.8.')
    parser.add_argument('--step', type=int, default=20, help='Initial step for scheduler and the default is 10.')
    # parser.add_argument('--channel_num', type=int, default=64, help='Revise channel quantity in network')

    parser.add_argument('--latent_dim', type=int, default=32, help='Edit laten dimension.')
    parser.add_argument('--reduced_dim', type=int, default=8, help='Edit reduced dimension.')
    parser.add_argument('--num_iter', type=int, default=1000, help='Edit cluster iteration.')
    parser.add_argument('--reduced_method', type=str, default='pca', help='Edit reduced method(PCA or tSNE).')

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('-p', '--plot_cm', action='store_true', help='Ploting confusion matrix.')
    parser.add_argument('--data_aug', action='store_true', help='Use data augmentation or not.')
    parser.add_argument('--scheduler', action='store_true', help='Use early stopping technique or not.')
    return parser.parse_args()

def wandb_update():
    config = wandb.config
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.optimizer = args.optimizer

    config.weight_d = args.weight_d
    config.momentum = args.momentum
    config.checkpoint = args.checkpoint
    config.gamma = args.gamma
    config.step = args.step

    config.latent_dir = args.latent_dim
    config.reduced_dim = args.reduced_dim
    config.num_iter = args.num_iter
    config.reduced_method = args.reduced_method

    config.data_aug = args.data_aug
    config.scheduler = args.scheduler
    # config.channel_num = args.channel_num


'''********************************************* 
  Fix random seed
 *********************************************'''
SEED = 5566 # Do not modify
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)


#   TODO: Modified the hyper-parameter
args = parse_args()
NUM_EPOCH = args.epochs
BATCH_SIZE = args.batch_size
LATENT_DIM = args.latent_dim
REDUCED_DIM = args.reduced_dim 
NUM_ITER = args.num_iter
REDUCED_METHOD =  args.reduced_method   # 'pca' or 'tsne'
lr = args.lr

DATA_PATH = './dataset/trainX.npy'


'''********************************************* 
  Define Dataset
 *********************************************'''
class Dataset(data.Dataset):
    def __init__(self, data_path):
        self.total_img = torch.from_numpy(np.load(data_path)).float()
        self.total_img = self.total_img.permute(0, 3, 1, 2)
        self.total_img = self.total_img/255
        
    def normalize(self, img):
        # TODO: normalize the dataset 
        # transform_nor = transforms.Normalize(mean=(0.4749, 0.4805, 0.4395), std=(0.2377, 0.2324, 0.2646))
        # return transform_nor(img)
        return img
    
    def augment(self, img):
        # TODO: do augmentation while loading image
        return img
    
    def __len__(self):
        return len(self.total_img)

    def __getitem__(self, index):
        img = self.total_img[index]
        img_aug = self.augment(img)
        
        img_aug = self.normalize(img_aug)
        img = self.normalize(img)
        return img_aug, img


#Please finish this block to run this code!**
'''********************************************* 
  Define Model Architerchure
 *********************************************'''
class Net(nn.Module):
    def __init__(self, image_channels=3, latent_dim=128, n_chansl=32):
        super(Net, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = 32
        
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, n_chansl, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # TODO: define your own structure
        )
        
        # TODO: check the dimension if you modified the structure
        self.fc1 = nn.Linear(n_chansl * (self.img_size//2)**2, self.latent_dim)

        # TODO: check the dimension if you modified the structure
        self.fc2 = nn.Linear(self.latent_dim, n_chansl * (self.img_size//2)**2)

        self.decoder = nn.Sequential(
           # TODO: define yout own structure
           # Hint: nn.ConvTranspose2d(...)
           nn.ConvTranspose2d(n_chansl, image_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
           nn.ReLU()
        )
                
    def forward(self, x):
        feature_map = self.encoder(x)   # Input:32x3x32x32 Output:32x32x16x16
        latent_vec = self.fc1(feature_map.reshape(feature_map.shape[0], -1))    # 32x32
        feature_map2 = self.fc2(latent_vec) # 32*8192
        x_res = self.decoder(feature_map2.reshape(feature_map2.shape[0], 32, 16, 16))
        
        return latent_vec, x_res

class Net1(nn.Module):
    def __init__(self, image_channels=3, latent_dim=128):
        super(Net, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = 32
        
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2048, 4096, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4096, 8192, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # TODO: define your own structure
        )
        
        # TODO: check the dimension if you modified the structure
        self.fc1 = nn.Linear(8192, self.latent_dim) 

        # TODO: check the dimension if you modified the structure
        self.fc2 = nn.Linear(self.latent_dim, 8192)

        self.decoder = nn.Sequential(
           # TODO: define yout own structure
           # Hint: nn.ConvTranspose2d(...)
           nn.ConvTranspose2d(8192, 4096, kernel_size=3, stride=2, padding=1),
           nn.ReLU(),
           nn.ConvTranspose2d(4096, 2048, kernel_size=3, stride=2, padding=1),
           nn.ReLU(),
           nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1),
           nn.ReLU(),
           nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
           nn.ReLU(),
           nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1),
           nn.ReLU(),
           nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
           nn.ReLU(),
           nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
           nn.ReLU(),
           nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
           nn.ReLU(),
           nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=2, padding=1),
           nn.ReLU(),
        )
                
    def forward(self, x):
      
        feature_map = self.encoder(x)
        latent_vec = self.fc1(feature_map.reshape(feature_map.shape[0], -1))
        feature_map2 = self.fc2(latent_vec)
        # x_res = self.decoder(feature_map2)
        x_res = self.decoder(feature_map2.reshape(feature_map2.shape[0], 8192, 1, 1))
        return latent_vec, x_res

'''********************************************* 
  Define Training Process
 *********************************************'''
def training(train, val, model, device, n_epoch, batch, lr):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('=== start training, parameter total:%d, trainable:%d' % (total, trainable))
    criterion = nn.MSELoss()

    '''Optim Prepare'''
    params = model.parameters()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, weight_decay=args.weight_d, lr=lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weight_d)
    else:
        raise ValueError("Optimizer not supported.")
    if args.scheduler == True:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    best_loss = 100
    for epoch in trange(n_epoch):
        train_total_loss = 0
        
        # training set
        model.train()
        idx = 0
        for image_aug, image in train:
            image = image.to(device, dtype=torch.float)
            image_aug = image_aug.to(device, dtype=torch.float)
            _, reconsturct = model(image_aug)
            loss = criterion(reconsturct, image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total_loss += (loss.item() / len(train))

            print('[Epoch %d | %d/%d] loss: %.4f' % ((epoch+1), idx*batch, len(train)*batch, loss.item()), end='\r')
            idx += 1 
        print("\n  Training  | Loss:%.4f " % train_total_loss)

        # validation set
        model.eval()
        val_total_loss = 0
        idx = 0 
        with torch.no_grad():
            for image_aug, image in val:
                image = image.to(device, dtype=torch.float)
                image_aug = image_aug.to(device, dtype=torch.float)
                _, reconstruct = model(image_aug)

                loss = criterion(reconstruct, image)
                val_total_loss += (loss.item() / len(val))
                idx += 1
            print(" Validation | Loss:%.4f " % val_total_loss)
        
        # save model
        if val_total_loss < best_loss:
                best_loss = val_total_loss
                print("saving model with loss %.4f...\n" % val_total_loss)
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                            os.path.join('./model', "epoch" + str(epoch) + "_" + str(round(val_total_loss, 5)) + ".pth"))
        if args.wandb:
            wandb.log({"lr": optimizer.param_groups[0]['lr'],
                        "train_loss": np.mean(train_total_loss),
                        "val_loss": val_total_loss})
        if args.scheduler == True:
            scheduler.step()


'''********************************************* 
  Define Clustering Process
 *********************************************'''
def clustering(model, device, loader, n_iter, reduced_method, reduced_dim, perplexity):
    assert reduced_method in ['pca', 'tsne', None]
    
    model.eval()
    latent_vec = torch.tensor([]).to(device, dtype=torch.float)
    for idx, (image_aug, image) in enumerate(loader):
        print("predict %d / %d" % (idx, len(loader)) , end='\r')
        image = image.to(device, dtype=torch.float)
        latent, r = model(image)
        latent_vec = torch.cat((latent_vec, latent), dim=0)

    latent_vec = latent_vec.cpu().detach().numpy()
    
    if reduced_method == 'tsne':
        tsne = TSNE(n_components=reduced_dim, verbose=1, method='exact', perplexity=perplexity, n_iter=n_iter)
        latent_vec = tsne.fit_transform(latent_vec)
    elif reduced_method == 'pca':
        pca = PCA(n_components=reduced_dim, copy=False, whiten=True, svd_solver='full')
        latent_vec = pca.fit_transform(latent_vec)

    kmeans = KMeans(n_clusters=2, random_state=0, max_iter=n_iter).fit(latent_vec)
    return kmeans.labels_


'''********************************************* 
  Define write function
 *********************************************'''
def write_output(predict_result, file_name='predict.csv'):      
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for i in range(len(predict_result)):
            writer.writerow([str(i), str(predict_result[i])])


'''********************************************* 
  Main Process
 *********************************************'''
if __name__ == '__main__':
    if args.mode == 'train':
        if args.wandb:
            wandb.init(project='MLHW3')
            wandb_update()
        # Build dataset
        dataset = Dataset(DATA_PATH)
        # print(len(dataset))

        # Random split
        train_set_size = int(len(dataset) * 0.85)
        valid_set_size = len(dataset) - train_set_size
        train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size])

        # set data loader
        train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)
        valid_loader = data.DataLoader(valid_set, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)

        model = Net(latent_dim=LATENT_DIM).to(device)
        if args.checkpoint:
            print("Loading pretrained weights...", args.checkpoint)
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(model)
        training(train_loader, valid_loader, model, device, NUM_EPOCH, BATCH_SIZE, lr)


    if args.mode == 'test':
        '''********************************************* 
        Inference
        *********************************************'''
        dataset = Dataset(DATA_PATH)
        
        test_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        model = Net(latent_dim=LATENT_DIM).to(device)
        model = model.cuda()
        model.load_state_dict(torch.load(args.checkpoint)["model_state_dict"], strict=False)
        predicted = clustering(model, device, test_loader, NUM_ITER, reduced_method=REDUCED_METHOD, reduced_dim=REDUCED_DIM, perplexity=15)
        predicted = 1 - predicted
        write_output(predicted, './testing_result/pred.csv')