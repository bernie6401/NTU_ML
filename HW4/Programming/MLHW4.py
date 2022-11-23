"""********************************************* 
  Import packages.
 *********************************************"""
# other library
import os
import csv
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=UserWarning)
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

# PyTorch library
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# Self-defined
import argparse
from tqdm import trange
import wandb
import ipdb
import re


SEED = 1124 # Set your lucky number as the random seed
MODEL_DIR = './model/RNN/model'
#MODEL_DIR = './drive/MyDrive/Colab Notebooks/MLHW4/model/RNN/model'

"""********************************************* 
  Basic setup of hyperparameters
 *********************************************"""
EPOCH_NUM = 10
lr = 1e-5
BATCH_SIZE = 256
OPTIMIZER = 'sgd'

weight_d = 1e-5
momentum = 0.9
CHECKPOINT = '' #MODEL_DIR + '_55.33854.pth' + '_61.46889.pth'
gamma = 0.8
step = 20

MAX_POSITIONS_LEN = 100
w2v_dim = 250
embedding_dim = 250
net_hidden_dim = 150
net_num_layers = 1
dropout = 0.5
header_hidden_dim = 150

mode = 'train'
WANDB = False
DATA_AUG = False
SCHEDULER = False

"""********************************************* 
  Do not changed
 *********************************************"""
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

w2v_config = {'path': './model/w2v/w2v_local_' + str(w2v_dim) + '.model', 'dim': w2v_dim}
w2v_config = {'path': './drive/MyDrive/Colab Notebooks/MLHW4/model/w2v/w2v_local_' + str(w2v_dim) + '_parse_text_cbow.model', 'dim': w2v_dim}
# net_config = {'hidden_dim': net_hidden_dim, 'num_layers': net_num_layers, 'bidirectional': False, 'fix_embedding': True}
net_config = {'embedding_dim':embedding_dim, 'hidden_dim': net_hidden_dim, 'num_layers': net_num_layers, 'bidirectional': False, 'fix_embedding': True}
header_config = {'dropout': dropout, 'hidden_dim': header_hidden_dim}
assert header_config['hidden_dim'] == net_config['hidden_dim'] or header_config['hidden_dim'] == net_config['hidden_dim'] * 2


"""********************************************* 
  Self-defined
 *********************************************"""
def wandb_update():
    config = wandb.config
    config.epochs = EPOCH_NUM
    config.learning_rate = lr
    config.batch_size = BATCH_SIZE
    config.optimizer = OPTIMIZER

    config.weight_d = weight_d
    config.momentum = momentum
    config.checkpoint = CHECKPOINT
    config.gamma = gamma
    config.step = step

    config.max_position_len = MAX_POSITIONS_LEN
    config.w2v_dim = w2v_dim
    config.embedding_dim = embedding_dim
    config.net_hidden_dim = net_hidden_dim
    config.net_num_layers = net_num_layers
    config.dropout = dropout
    config.header_hidden_dim = header_hidden_dim

    config.data_aug = DATA_AUG
    config.scheduler = SCHEDULER


"""********************************************* 
  Auxiliary functions and classes definition
 *********************************************"""
def parsing_text(text):
    # TODO: do data processing
    # print(text)
    text = text.split(' ')
    at_someone = re.compile('^@')
    http_str = re.compile('^http')
    www_str = re.compile('^www')
    hashtag_str = re.compile('^#')
    text = [string for string in text if not re.match(at_someone, string)]  # Remove @.... string
    text = [string for string in text if not re.match(http_str, string)]    # Remove http... string(url)
    text = [string for string in text if not re.match(www_str, string)]     # Remove www... string(url)
    text = [string for string in text if not re.match(hashtag_str, string)]     # Remove #... string(hashtag)
    pass_str = ['.', '-', '&lt;', '&gt;', '&amp;', '&quot;', '!!!']         # Replace blacklist string to white space
    for i in range(len(text)):
        for j in range(len(pass_str)):
            text[i] = text[i].replace(pass_str[j], ' ')
    text = ' '.join(text)
    # print(text)
    return text

def load_train_label(path='HW4_dataset/train.csv'):
    tra_lb_pd = pd.read_csv(path)
    label = torch.FloatTensor(tra_lb_pd['label'].values)
    idx = tra_lb_pd['id'].tolist()
    text = [parsing_text(s).split(' ') for s in tra_lb_pd['text'].tolist()]
    return idx, text, label

def load_train_nolabel(path='HW4_dataset/train_nolabel.csv'):
    tra_nlb_pd = pd.read_csv(path)
    text = [parsing_text(s).split(' ') for s in tra_nlb_pd['text'].tolist()]
    return None, text, None

def load_test(path='HW4_dataset/test.csv'):
    tst_pd = pd.read_csv(path)
    idx = tst_pd['id'].tolist()
    text = [parsing_text(s).split(' ') for s in tst_pd['text'].tolist()]
    return idx, text

class Preprocessor:
    def __init__(self, sentences, w2v_config):
        self.sentences = sentences
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
        self.build_word2vec(sentences, **w2v_config)
        
    def build_word2vec(self, x, path, dim):
        if os.path.isfile(path):
            print("loading word2vec model ...")
            w2v_model = Word2Vec.load(path)
        else:
            print("training word2vec model ...")
            w2v_model = Word2Vec(x, size=dim, window=5, min_count=5, workers=12, iter=10, sg=0)
            print("saving word2vec model ...")
            w2v_model.save(path)
            
        self.embedding_dim = w2v_model.vector_size
        for i, word in enumerate(w2v_model.wv.vocab):
            #e.g. self.word2index['he'] = 1 
            #e.g. self.index2word[1] = 'he'
            #e.g. self.vectors[1] = 'he' vector
            
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(w2v_model[word])
        
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        self.add_embedding('<PAD>')
        self.add_embedding('<UNK>')
        print("total words: {}".format(len(self.embedding_matrix)))
        
    def add_embedding(self, word):
        # 把 word 加進 embedding，並賦予他一個隨機生成的 representation vector
        # word 只會是 "<PAD>" 或 "<UNK>"
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)   
        
    def sentence2idx(self, sentence):
        sentence_idx = []
        for word in sentence:
            if word in self.word2idx.keys():
                sentence_idx.append(self.word2idx[word])
            else:
                sentence_idx.append(self.word2idx["<UNK>"])
        return torch.LongTensor(sentence_idx)
    
class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, id_list, sentences, labels, preprocessor):
        self.id_list = id_list
        self.sentences = sentences
        self.labels = labels
        self.preprocessor = preprocessor
    
    def __getitem__(self, idx):
        if self.labels is None: return self.id_list[idx], self.preprocessor.sentence2idx(self.sentences[idx])
        return self.id_list[idx], self.preprocessor.sentence2idx(self.sentences[idx]), self.labels[idx]
    
    def __len__(self):
        return len(self.sentences)
    
    def collate_fn(self, data):
        id_list = torch.LongTensor([d[0] for d in data])
        lengths = torch.LongTensor([len(d[1]) for d in data])
        texts = pad_sequence(
            [d[1] for d in data], batch_first=True).contiguous()
     
        if self.labels == None: 
            return id_list, lengths, texts
        else:
          labels = torch.FloatTensor([d[2] for d in data])
          return id_list, lengths, texts, labels


train_idx, train_label_text, label = load_train_label('HW4_dataset/train.csv')
preprocessor = Preprocessor(train_label_text, w2v_config)


train_idx, valid_idx, train_label_text, valid_label_text, train_label, valid_label = train_test_split(train_idx, train_label_text, label, test_size=0.5)
train_dataset, valid_dataset = TwitterDataset(train_idx, train_label_text, train_label, preprocessor), TwitterDataset(valid_idx, valid_label_text, valid_label, preprocessor)

test_idx, test_text = load_test('HW4_dataset/test.csv')
test_dataset = TwitterDataset(test_idx, test_text, None, preprocessor)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn = train_dataset.collate_fn, num_workers = 8)
valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = BATCH_SIZE, shuffle = False, collate_fn = valid_dataset.collate_fn, num_workers = 8)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False, collate_fn = test_dataset.collate_fn, num_workers = 8)


"""********************************************* 
  Definition of RNN network
 *********************************************"""
class Backbone(torch.nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, bidirectional, fix_embedding=True):
        super(Backbone, self).__init__()
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        self.embedding.weight.requires_grad = False if fix_embedding else True  # 是否將embedding固定住，如果fix_embedding爲False，在訓練過程中，embedding也會跟着被訓練
        
        # self.net = torch.nn.RNN(embedding.size(1), hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.net = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.net(inputs)
        return x
    
class Header(torch.nn.Module):
    def __init__(self, dropout, hidden_dim):
        super(Header, self).__init__()
        # TODO: you should design your classifier module
        self.classifier = torch.nn.Sequential(torch.nn.Linear(hidden_dim, 1),
                            torch.nn.Sigmoid())
        
    @ torch.no_grad()
    def _get_length_masks(self, lengths):
        # lengths: (batch_size, ) in cuda
        ascending = torch.arange(MAX_POSITIONS_LEN)[:lengths.max().item()].unsqueeze(0).expand(len(lengths), -1).to(lengths.device)
        length_masks = (ascending < lengths.unsqueeze(-1)).unsqueeze(-1)
        return length_masks
    
    def forward(self, inputs, lengths):
        # the input shape should be (N, L, D∗H)
        pad_mask = self._get_length_masks(lengths)
        inputs = inputs * pad_mask
        inputs = inputs.sum(dim=1)
        out = self.classifier(inputs).squeeze()
        return out


"""********************************************* 
  Trainer
 *********************************************"""
def train(train_loader, backbone, header, optimizer, criterion, device, epoch):

    total_loss = []
    total_acc = []
    
    for i, (idx_list, lengths, texts, labels) in enumerate(train_loader):
        lengths, inputs, labels = lengths.to(device), texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        if not backbone is None:
            inputs = backbone(inputs)
        soft_predicted = header(inputs, lengths)
        loss = criterion(soft_predicted, labels)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        if WANDB:
          wandb.log({"train_loss": np.mean(total_loss)})
        
        # ipdb.set_trace()
        with torch.no_grad():
            hard_predicted = (soft_predicted >= 0.5).int()
            correct = sum(hard_predicted == labels).item()
            batch_size = len(labels)
            acc = correct * 100 / len(labels)
            total_acc.append(acc)

            if WANDB:
              wandb.log({"lr": optimizer.param_groups[0]['lr'],
                          "train_acc": np.mean(total_acc),})
    backbone.train()
    header.train()
    return np.mean(total_loss), np.mean(total_acc)

def valid(valid_loader, backbone, header, criterion, device, epoch):
    backbone.eval()
    header.eval()
    with torch.no_grad():
        total_loss = []
        total_acc = []
        
        for i, (idx_list, lengths, texts, labels) in enumerate(valid_loader):
            lengths, inputs, labels = lengths.to(device), texts.to(device), labels.to(device)

            if not backbone is None:
                inputs = backbone(inputs)
            soft_predicted = header(inputs, lengths)
            loss = criterion(soft_predicted, labels)
            total_loss.append(loss.item())
            
            hard_predicted = (soft_predicted >= 0.5).int()
            correct = sum(hard_predicted == labels).item()
            acc = correct * 100 / len(labels)
            total_acc.append(acc)
            
            # print('[Validation in epoch {:}] loss:{:.3f} acc:{:.3f}'.format(epoch+1, np.mean(total_loss), np.mean(total_acc)), end='\r')

            if WANDB:
                wandb.log({"val_loss": np.mean(total_loss),
                            "val_acc": np.mean(total_acc),})
    backbone.train()
    header.train()
    return np.mean(total_loss), np.mean(total_acc)
          
def run_training(train_loader, valid_loader, backbone, header, epoch_num, lr, device, model_dir): 
    def check_point(backbone, header, loss, acc, model_dir):
        # TODO
        torch.save({'backbone': backbone}, model_dir + "_backbone_" + str(round(acc, 5)) + '.pth')
        torch.save({'header': header}, model_dir + "_header_" + str(round(acc, 5)) + '.pth')
    def is_stop(loss, acc):
        # TODO
        return False
    
    if backbone is None:
        trainable_paras = header.parameters()
    else:
        trainable_paras = list(backbone.parameters()) + list(header.parameters())
        
    '''Optim Prepare'''
    if OPTIMIZER == 'adam':
      optimizer = torch.optim.Adam(trainable_paras, weight_decay=weight_d, lr=lr)
    elif OPTIMIZER == 'sgd':
      optimizer = torch.optim.SGD(trainable_paras, lr=lr, momentum=momentum, weight_decay=weight_d)
    else:
      raise ValueError("Optimizer not supported.")

    if SCHEDULER == True:
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
    
    backbone.train()
    header.train()
    backbone = backbone.to(device)
    header = header.to(device)
    criterion = torch.nn.BCELoss()
    best_acc = 78
    for epoch in range(epoch_num):
        train_loss, train_acc = train(train_loader, backbone, header, optimizer, criterion, device, epoch)
        loss, acc = valid(valid_loader, backbone, header, criterion, device, epoch)
        print('[Training in epoch {:}] loss:{:.3f} acc:{:.3f}'.format(epoch+1, train_loss, train_acc))
        print('[Validation in epoch {:}] loss:{:.3f} acc:{:.3f} '.format(epoch+1, loss, acc))
        if acc > best_acc:
            best_acc = acc
            check_point(backbone, header, loss, acc, model_dir)
        if is_stop(loss, acc):
            break
        if SCHEDULER == True:
            scheduler.step()


"""********************************************* 
  Training
 *********************************************"""
backbone = Backbone(preprocessor.embedding_matrix, **net_config).to(device)
header = Header(**header_config).to(device)

if mode == 'train':
    if WANDB:
        wandb.init(project='MLHW4')
        wandb_update()
    if os.path.isfile(MODEL_DIR + '_backbone_' + CHECKPOINT + '.pth') and os.path.isfile(MODEL_DIR + '_header_' + CHECKPOINT + '.pth'):
        print('Loading RNN model...')
        backbone = torch.load(MODEL_DIR + '_backbone_' + CHECKPOINT + '.pth')
        header = torch.load(MODEL_DIR + '_header_' + CHECKPOINT + '.pth')
    run_training(train_loader, valid_loader, backbone, header, EPOCH_NUM, lr, device, MODEL_DIR)


"""********************************************* 
  Testing
 *********************************************"""
def run_testing(test_loader, backbone, header, device, output_path):
  with open(output_path, 'w') as f:
    backbone.eval()
    header.eval()
    writer = csv.writer(f)
    writer.writerow(['id', 'label'])
    with torch.no_grad():
      for i, (idx_list, lengths, texts) in enumerate(test_loader):
        lengths, inputs = lengths.to(device), texts.to(device)
        if not backbone is None:
          inputs = backbone(inputs)
        soft_predicted = header(inputs, lengths)
        hard_predicted = (soft_predicted >= 0.5).int()
        for i, p in zip(idx_list, hard_predicted):
          writer.writerow([str(i.item()), str(p.item())])

if os.path.isfile(MODEL_DIR + '_backbone_' + CHECKPOINT + '.pth') and os.path.isfile(MODEL_DIR + '_header_' + CHECKPOINT + '.pth'):
    print('Loading RNN model...')
    backbone = torch.load(MODEL_DIR + '_backbone_' + CHECKPOINT + '.pth')
    header = torch.load(MODEL_DIR + '_header_' + CHECKPOINT + '.pth')
pred_file = './testing_result/pred_local.csv'
run_testing(test_loader, backbone, header, device, pred_file)
# from google.colab import files
# files.download(pred_file)