import random
from re import T
import pandas as pd
from PIL import Image
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn


TRA_PATH = './dataset/train/'
TST_PATH = './dataset/test/'
LABEL_PATH = './dataset/train.csv'

def load_train_data(img_path, label_path, valid_ratio=0.12):
    train_label = pd.read_csv(label_path)['label'].values.tolist()
    train_image = [f'{img_path}/{i+10000}.jpg' for i in range(len(train_label)-1)]
    
    train_data = list(zip(train_image, train_label))
    random.shuffle(train_data)
    
    split_len = int(len(train_data) * valid_ratio)
    train_set = train_data[split_len:]
    valid_set = train_data[:split_len]
    
    return train_set, valid_set

def data_distribution(label_path):
    train_label = pd.read_csv(label_path)['label'].values.tolist()
    data_dist = np.zeros(7)
    for i in range(len(train_label)):
        data_dist[train_label[i]] += 1

    return data_dist

def load_test_data(img_path):
    test_set = [f'{img_path}/{i}.jpg' for i in range(7000, 10000)]
    return test_set

class FaceExpressionDataset(Dataset):
    def __init__(self, data, augment=None):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)
    
    def normalize(self, data):
        # TODO: do normalization there
        transform = transforms.Normalize(mean=0.5085, std=0.2644)
        return transform(data)
    
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

class FaceExpressionNet(nn.Module):
    def __init__(self, n_chansl=256):
        super(FaceExpressionNet, self).__init__()
        # TODO
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, affine=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d((2, 2)),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_chansl, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_chansl, eps=1e-05, affine=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d((2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_chansl, n_chansl//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_chansl//2, eps=1e-05, affine=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d((2, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(n_chansl//2, n_chansl//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_chansl//4, eps=1e-05, affine=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d((2, 2)),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(n_chansl//4, n_chansl//8, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_chansl//8, eps=1e-05, affine=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d((2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(n_chansl//4 * 8 * 8, 7),
        )

    def forward(self, x):
        #image size (64,64) -> Shape: [Batch_size, 1, 64, 64]
        x = self.conv1(x)   # (Batch_size, 256, 32, 32)->(B, C, H, W)
        x = self.conv2(x)   # (Batch_size, 128, 16, 16)->(B, C, H, W)
        x = self.conv3(x)   # (Batch_size, 64, 8, 8)->(B, C, H, W)
        # x = self.conv4(x)   # (Batch_size, 32, 4, 4)->(B, C, H, W)
        x = x.flatten(start_dim=1)  # Shape: [Batch_size, 4096]->[B, C*H*W]
        x = self.fc(x)  # Shape: [Batch_size, 7]
        return x


'''    
class TestingDataset(Dataset):
    def __init__(self, data, augment=None):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)
    
    def normalize(self, data):
        # TODO: do normalization there
        pass
    
    def read_img(self, idx):
        img = Image.open(self.data[idx])
        if not self.augment is None:
            img = self.augment(img)
        img = torch.from_numpy(np.array(img)).float()
        img = img.unsqueeze(0).float()
        # img = self.normalize(img)
        return img, self.data[idx].split('/')[-1][:-4]
        
    def __getitem__(self, idx):
        img, name = self.read_img(idx)
        
        return img, name

X = [[0, 1], [1, 2]]
Y = [ [ int(x1+x2 < 1) ] for (x1, x2) in X ]
print(type(Y))
Z=[]
for x1, x2 in X:
    Z.append([int(x1+x2 < 1)])
print(type(Z))

data_path = './temp'
preprocess = transforms.Compose([transforms.ToTensor()])
cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform=preprocess)
img1 = [cifar10_img for cifar10_img, _ in cifar10]

# print(type(img))
img = torch.stack(img1, dim=3)
img2 = []
for i, (cifar10_img, _) in enumerate(cifar10):
    img2.append(cifar10_img)
    if i == 0:
        print(cifar10_img.shape)
        print(cifar10_img.size)
        print(type(cifar10_img))
print(img2.__sizeof__)
img3 = torch.stack(img2, dim=3)
print(img.view(3, -1).mean(dim=1))
print(img.view(3, -1).std(dim=1))
print(img3.shape)
print(img3.view(3, -1).mean(dim=1))
print(img3.view(3, -1).std(dim=1))'''


'''Aumentation'''
transform_set = [
    # transforms.Resize((32, 32)),    # Resize image to a fitting size
    # transforms.RandomResizedCrop(size=224, scale=(0.5, 0.5)), # Resized crop image in random
    # transforms.CenterCrop(32),     # Cutting image by original center to a fitting size
    # transforms.RandomHorizontalFlip(p=0.5),   # Horizontal Flip in random
    # transforms.RandomVerticalFlip(p=0.5),   # Vertical Flip in random
    # transforms.ColorJitter(brightness=(0, 5), contrast=(0, 5), saturation=(0, 5), hue=(-0.1, 0.1)),  # Adjust image brightness, contrast, satuation and hue in random
    transforms.RandomRotation(30, center=(0, 0), expand=False),   # expand only for center rotation
]
size = 48
transform_aug = transforms.Compose([
    transforms.RandomChoice(transform_set),
    transforms.CenterCrop(size),     # Cutting image by original center to a fitting size
    transforms.Pad((64 - size)//2, fill=0, padding_mode="constant"), 
])


'''Computing mean and std of training dataset'''
# train_set, valid_set = load_train_data(TRA_PATH, LABEL_PATH)
# transform = transforms.Compose([transforms.ToTensor()])
# train_dataset = FaceExpressionDataset(train_set, transform)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
# img4 = []
# for i, (img, _) in enumerate(train_loader):
#     img4.append(img)
# img5 = torch.stack(img4, dim=-1)
# print(img5.view(1, -1).mean(dim=1))
# print(img5.view(1, -1).std(dim=1))


'''Implement Augmentation'''
# train_set, valid_set = load_train_data(TRA_PATH, LABEL_PATH)
# train_dataset = FaceExpressionDataset(train_set, transform_aug)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# model = FaceExpressionNet()
# for i, (img, _) in enumerate(train_loader):
#     output = model(img)


'''Plot Data Distribution'''
# labels_name = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# data_dist = data_distribution(LABEL_PATH)
# dict = {
#     'Emotion': labels_name,
#     'Numbers': data_dist 
# }
# select_df = pd.DataFrame(dict)
# plt.rcParams["figure.figsize"] = (12, 8)
# plt.rcParams['savefig.dpi'] = 100   # Image Pixel
# plt.rcParams['figure.dpi'] = 100    # Resolution
# ax = select_df[['Emotion', 'Numbers']].plot(x='Emotion', kind='bar', color='#5887ff')   # Get column in select_df and let Emotion be x-axis

# '''Info of chart'''
# plt.title("Data Distribution", fontsize="18")
# plt.xlabel("Emotion", fontsize="14")
# plt.ylabel("Numbers", fontsize="14", rotation=360, horizontalalignment='right', verticalalignment='top')

# '''Scale and Range'''
# plt.xticks(rotation=30)   # Size of x-axis scale

# '''Put each class number on each bar'''
# x = select_df['Emotion'].tolist()
# y= select_df['Numbers'].tolist()
# l=[i for i in range(len(select_df))]
# for i,(_x,_y) in enumerate(zip(l, y)):
#     plt.text(_x, _y, str(int(y[i])), ha='center', va= 'bottom', color='black', fontsize=12)
# plt.show()
# plt.savefig('./Img/data_distribution.png', format='png')


'''Check the image after transforming'''
# import PIL.Image as Image
# import torchvision
# import matplotlib.pyplot as plt
# import numpy as np
# import warnings
# warnings.filterwarnings("ignore")
# imagepath='./dataset/train/10000.jpg' 
# # read image with PIL module
# img_pil = Image.open(imagepath, mode='r')
# img_pil = img_pil.convert('RGB')
# from torchvision import transforms
# from torchvision.transforms import functional as TF
# trans_toPIL = transforms.ToPILImage() # 將  "pytoch tensor" 或是  "numpy.ndarray" 轉換成 PIL Image.
# img_np = np.asarray(img_pil) # 將PIL image轉換成  "numpy.ndarray" 
# print('image type before convert:{}'.format(type(img_np)))
# img_pil = trans_toPIL(img_np)
# print('image type after convert:{}'.format(type(img_pil)))
# size = 224
# # transform = transforms.Resize(size)
# transform = transforms.RandomRotation(30, center=(0, 0), expand=True)
# img_pil_normal = transform(img_pil)
# img_pil_normal.show()