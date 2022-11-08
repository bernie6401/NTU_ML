from torch.utils import data 
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
import random
# from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

SEED = 5566
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

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

    # kmeans = KMeans(n_clusters=2, random_state=0, max_iter=n_iter).fit(latent_vec)
    return latent_vec#kmeans.labels_


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
'''
DATA_PATH = './dataset/trainX.npy'
dataset = Dataset(DATA_PATH)

# Random split
train_set_size = int(len(dataset) * 1)
valid_set_size = len(dataset) - train_set_size
train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size])

# set data loader
train_loader = data.DataLoader(train_set, batch_size=1, num_workers=1, shuffle=False)
valid_loader = data.DataLoader(valid_set, batch_size=1, num_workers=1, shuffle=False)
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
transform = T.ToPILImage()

if __name__ == '__main__':
    for i, (image_aug, image) in enumerate(train_loader):
        image = torch.squeeze(image, 0)
        img = transform(image)
        img.save("./dataset/img/hwq2_" + str(i) + ".png")
'''

''' ********************************************* 
    Report Qrestion 2
********************************************* '''
'''Try to transform image to numpy'''
'''
if __name__ == '__main__':
    DATA_PATH = './img/hwq2_'
    image1 = Image.open(DATA_PATH + '0.png')
    image1 = np.asarray(image1)
    image2 = Image.open(DATA_PATH + '1.png')
    image2 = np.asarray(image2)
    total_img = np.zeros((2, 32, 32, 3))
    # total_img = np.append(total_img, image1)
    # total_img = np.append(total_img, image2)
    # print(total_img[0].size())
    total_img[0] = image1
    total_img[1] = image2
    
    # total_img1 = np.load('./dataset/trainX.npy')
    # image3 = total_img1[0]
    # image4 = total_img1[1]
    np.save('./img_npy/hwq2', total_img)
if __name__ == '__main__':
    # total_img = np.load('./dataset/trainX.npy')
    # img1 = total_img[0]
    # img2 = total_img[1]
    DATA_PATH = './img_npy/hwq2.npy'
    dataset = Dataset(DATA_PATH)
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    # Random split
    train_set_size = int(len(dataset) * 1)
    valid_set_size = len(dataset) - train_set_size
    train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size])

    # set data loader
    train_loader = data.DataLoader(train_set, batch_size=1, num_workers=1, shuffle=False)
    valid_loader = data.DataLoader(valid_set, batch_size=1, num_workers=1, shuffle=False)

    model = Net(latent_dim=LATENT_DIM).to(device)
    checkpoint = './model/1102_1_origin/epoch9_0.0107.pth'
    print("Loading pretrained weights...", checkpoint)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    transform = T.ToPILImage()
    criterion = nn.MSELoss()
    for i, (image_aug, image) in enumerate(train_loader):
        image = image.to(device, dtype=torch.float)
        image_aug = image_aug.to(device, dtype=torch.float)
        _, reconsturct = model(image_aug)
        image = torch.squeeze(image, 0)
        img = transform(image)
        loss = criterion(reconsturct, image)
        img.save("./img/hwq2_reconstruct_" + str(i) + ".png")
        print(loss)
'''


''' ********************************************* 
    Report Qrestion 3
********************************************* '''
if __name__ == '__main__':
    DATA_PATH = './dataset/visualization_X.npy'
    dataset = Dataset(DATA_PATH)
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    # Random split
    test_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = Net(latent_dim=LATENT_DIM).to(device)
    checkpoint = './model/original/epoch4.pth'
    print("Loading pretrained weights...", checkpoint)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    predicted = clustering(model, device, test_loader, NUM_ITER, reduced_method=REDUCED_METHOD, reduced_dim=REDUCED_DIM, perplexity=15)
    print(predicted)
    print(len(predicted))
    # transform = T.ToPILImage()
    # criterion = nn.MSELoss()
    # for i, (image_aug, image) in enumerate(train_loader):
    #     image = image.to(device, dtype=torch.float)
    #     image_aug = image_aug.to(device, dtype=torch.float)
    #     _, reconsturct = model(image_aug)
    #     image = torch.squeeze(image, 0)
    #     img = transform(image)
    #     loss = criterion(reconsturct, image)
    #     img.save("./img/hwq2_reconstruct_" + str(i) + ".png")
    #     print(loss)