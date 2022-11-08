from torch.utils import data 
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn


class Dataset(data.Dataset):
    def __init__(self, data_path):
        self.total_img = torch.from_numpy(np.load(data_path)).float()
        self.total_img = self.total_img.permute(0, 3, 1, 2)
        self.total_img = self.total_img/255
        
    def normalize(self, img):
        # TODO: normalize the dataset 
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

NUM_EPOCH = 5
BATCH_SIZE = 32
LATENT_DIM = 32
REDUCED_DIM = 8
NUM_ITER = 1000
REDUCED_METHOD = 'pca'   # 'pca' or 'tsne'
lr = 5e-4

''' ********************************************* 
    Create image
********************************************* '''
# DATA_PATH = './dataset/trainX.npy'
# dataset = Dataset(DATA_PATH)

# # Random split
# train_set_size = int(len(dataset) * 0.85)
# valid_set_size = len(dataset) - train_set_size
# train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size])

# # set data loader
# train_loader = data.DataLoader(train_set, batch_size=1, num_workers=1, shuffle=True)
# valid_loader = data.DataLoader(valid_set, batch_size=1, num_workers=1, shuffle=False)
# use_gpu = torch.cuda.is_available()
# device = torch.device("cuda" if use_gpu else "cpu")
# transform = T.ToPILImage()

# if __name__ == '__main__':
    # for i, (image_aug, image) in enumerate(train_loader):
    #     image = torch.squeeze(image, 0)
    #     img = transform(image)
    #     img.save("./img/hwq2_" + str(i) + ".png")


''' ********************************************* 
    Report Qrestion 2
********************************************* '''
if __name__ == '__main__':
    DATA_PATH = './dataset/trainX.npy'
    dataset = Dataset(DATA_PATH)
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    # Random split
    train_set_size = int(len(dataset) * 0.85)
    valid_set_size = len(dataset) - train_set_size
    train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size])

    # set data loader
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)
    valid_loader = data.DataLoader(valid_set, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)

    model = Net(latent_dim=LATENT_DIM).to(device)
    print("Loading pretrained weights...", args.checkpoint)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    for image_aug, image in train:
        _, reconsturct = model(image_aug)