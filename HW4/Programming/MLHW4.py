'''********************************************* 
  Import packages
 *********************************************'''
# other library
from gensim.models import Word2Vec
import os
import csv
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# PyTorch library
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# Self-defined
import argparse
from tqdm import trange
import wandb

"""********************************************* 
  Self-defined
 *********************************************"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='Total training epochs(default is 1).')
    parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate for sgd(default is 1e-5).')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.(default is 256)')
    parser.add_argument('--optimizer', type=str, default="sgd", help='Optimizer, adam or sgd(default is sgd).')

    parser.add_argument('--weight_d', type=float, default=1e-5, help='Adjust weight decay(default is 1e-5)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd(default is 0.9)')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--mode', type=str, default='test', help='train or test mode(default is train)')
    parser.add_argument('--gamma', type=float, default=0.8, help='Initial gamma for scheduler and the default is 0.8.')
    parser.add_argument('--step', type=int, default=20, help='Initial step for scheduler and the default is 20.')

    parser.add_argument('--max_position_len', type=int, default=100, help='Edit position length.(default is 100)')
    parser.add_argument('--w2v_dim', type=int, default=64, help='Try to tune word to vector dimension(default is 64)')
    parser.add_argument('--net_hidden_dim', type=int, default=32, help='Try to tune network hidden dimension(default is 32)')
    parser.add_argument('--net_num_layers', type=int, default=1, help='Try to tune network # layer(default is 1)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Set network dropout config(default is 0.5).')
    parser.add_argument('--header_hidden_dim', type=int, default=32, help='Try to tune header # layer(default is 32)')

    parser.add_argument('--wandb', action='store_true')
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

    config.max_position_len = args.max_position_len
    config.w2v_dim = args.w2v_dim
    config.net_hidden_dim = args.net_hidden_dim
    config.net_num_layers = args.net_num_layers
    config.dropout = args.dropout
    config.header_hidden_dim = args.header_hidden_dim

    config.data_aug = args.data_aug
    config.scheduler = args.scheduler

'''********************************************* 
  Basic setup of hyperparameters
 *********************************************'''
args = parse_args()
BATCH_SIZE = args.batch_size
EPOCH_NUM = args.epochs
MAX_POSITIONS_LEN = args.max_position_len
SEED = 1124 # Set your lucky number as the random seed
MODEL_DIR = './model/RNN/model.pth'
lr = args.lr

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

w2v_config = {'path': './model/w2v/w2v.model', 'dim': args.w2v_dim}
net_config = {'hidden_dim': args.net_hidden_dim, 'num_layers': args.net_num_layers, 'bidirectional': False, 'fix_embedding': True}
header_config = {'dropout': args.dropout, 'hidden_dim': args.header_hidden_dim}
assert header_config['hidden_dim'] == net_config['hidden_dim'] or header_config['hidden_dim'] == net_config['hidden_dim'] * 2


'''********************************************* 
  Auxiliary functions and classes definition
 *********************************************'''
def parsing_text(text):
    # TODO: do data processing
    return text

def load_train_label(path='./dataset/train.csv'):
    tra_lb_pd = pd.read_csv(path)
    label = torch.FloatTensor(tra_lb_pd['label'].values)
    idx = tra_lb_pd['id'].tolist()
    text = [parsing_text(s).split(' ') for s in tra_lb_pd['text'].tolist()]
    return idx, text, label

def load_train_nolabel(path='./dataset/train_nolabel.csv'):
    tra_nlb_pd = pd.read_csv(path)
    text = [parsing_text(s).split(' ') for s in tra_nlb_pd['text'].tolist()]
    return None, text, None

def load_test(path='./dataset/test.csv'):
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
            w2v_model = Word2Vec(x, vector_size=dim, window=5, min_count=2, workers=12, epochs=2, sg=1)
            print("saving word2vec model ...")
            w2v_model.save(path)
            
        self.embedding_dim = w2v_model.vector_size
        for i, word in enumerate(list(w2v_model.wv.index_to_key)):   #for i, word in enumerate(w2v_model.wv.vocab):
            #e.g. self.word2index['he'] = 1 
            #e.g. self.index2word[1] = 'he'
            #e.g. self.vectors[1] = 'he' vector
            
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(w2v_model.wv[word])
        
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

train_idx, train_label_text, label = load_train_label('./dataset/train.csv')

preprocessor = Preprocessor(train_label_text, w2v_config)

train_idx, valid_idx, train_label_text, valid_label_text, train_label, valid_label = train_test_split(train_idx, train_label_text, label, test_size=0.5)
train_dataset, valid_dataset = TwitterDataset(train_idx, train_label_text, train_label, preprocessor), TwitterDataset(valid_idx, valid_label_text, valid_label, preprocessor)

test_idx, test_text = load_test('./dataset/test.csv')
test_dataset = TwitterDataset(test_idx, test_text, None, preprocessor)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = BATCH_SIZE,
                                            shuffle = True,
                                            collate_fn = train_dataset.collate_fn,
                                            num_workers = 8)
valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset,
                                            batch_size = BATCH_SIZE,
                                            shuffle = False,
                                            collate_fn = valid_dataset.collate_fn,
                                            num_workers = 8)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = BATCH_SIZE,
                                            shuffle = False,
                                            collate_fn = test_dataset.collate_fn,
                                            num_workers = 8)


'''********************************************* 
  Definition of RNN network
 *********************************************'''
class Backbone(torch.nn.Module):
    def __init__(self, embedding, hidden_dim, num_layers, bidirectional, fix_embedding=True):
        super(Backbone, self).__init__()
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        self.embedding.weight.requires_grad = False if fix_embedding else True
        
        self.net = torch.nn.RNN(embedding.size(1), hidden_dim, num_layers=num_layers, \
                                  bidirectional=bidirectional, batch_first=True)
        
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
        ascending = torch.arange(MAX_POSITIONS_LEN)[:lengths.max().item()].unsqueeze(
            0).expand(len(lengths), -1).to(lengths.device)
        length_masks = (ascending < lengths.unsqueeze(-1)).unsqueeze(-1)
        return length_masks
    
    def forward(self, inputs, lengths):
        # the input shape should be (N, L, D∗H)
        pad_mask = self._get_length_masks(lengths)
        inputs = inputs * pad_mask
        inputs = inputs.sum(dim=1)
        out = self.classifier(inputs).squeeze()
        return out


'''********************************************* 
  Trainer
 *********************************************'''
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
        
        with torch.no_grad():
            hard_predicted = (soft_predicted >= 0.5).int()
            correct = sum(hard_predicted == labels).item()
            batch_size = len(labels)
            acc = correct * 100 / batch_size
            total_acc.append(acc)
        
            print('[Training in epoch {:}] loss:{:.3f} acc:{:.3f}'.format(epoch+1, np.mean(total_loss), np.mean(total_acc)), end='\r')
        
            if args.wandb:
                wandb.log({"lr": optimizer.param_groups[0]['lr'],
                            "train_loss": np.mean(total_loss),
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
            
            print('[Validation in epoch {:}] loss:{:.3f} acc:{:.3f}'.format(epoch+1, np.mean(total_loss), np.mean(total_acc)), end='\r')

            if args.wandb:
                wandb.log({"val_loss": np.mean(total_loss),
                            "val_acc": np.mean(total_acc),})
    backbone.train()
    header.train()
    return np.mean(total_loss), np.mean(total_acc)

            
def run_training(train_loader, valid_loader, backbone, header, epoch_num, lr, device, model_dir): 
    def check_point(backbone, header, loss, acc, model_dir):
        # TODO
        torch.save({'backbone': backbone, 'header': header}, model_dir)
    def is_stop(loss, acc):
        # TODO
        return False
    
    if backbone is None:
        trainable_paras = header.parameters()
    else:
        trainable_paras = list(backbone.parameters()) + list(header.parameters())


    '''Optim Prepare'''
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(trainable_paras, weight_decay=args.weight_d, lr=lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(trainable_paras, lr=lr, momentum=args.momentum, weight_decay=args.weight_d)
    else:
        raise ValueError("Optimizer not supported.")
    if args.scheduler == True:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    
    backbone.train()
    header.train()
    backbone = backbone.to(device)
    header = header.to(device)
    criterion = torch.nn.BCELoss()
    for epoch in range(epoch_num):
        train_total_loss, train_total_acc = train(train_loader, backbone, header, optimizer, criterion, device, epoch)
        loss, acc = valid(valid_loader, backbone, header, criterion, device, epoch)
        print('[Validation in epoch {:}] loss:{:.3f} acc:{:.3f} '.format(epoch+1, loss, acc))
        check_point(backbone, header, loss, acc, model_dir)
        if is_stop(loss, acc):
            break

        if args.scheduler == True:
            scheduler.step()


'''********************************************* 
  Testing
 *********************************************'''
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


'''********************************************* 
  Main Process
 *********************************************'''
if __name__ == '__main__':
    if args.mode == 'train':
        if args.wandb:
            wandb.init(project='MLHW4')
            wandb_update()
        '''********************************************* 
        Training
        *********************************************'''
        backbone = Backbone(preprocessor.embedding_matrix, **net_config)
        header = Header(**header_config)

        run_training(train_loader, valid_loader, backbone, header, EPOCH_NUM, lr, device, MODEL_DIR)

    if args.mode == 'test':
        '''********************************************* 
        Make a submission file
        (Note: In principle, you don't need to modify this part, and please make sure that you follow the correct format of the produced files.)
        *********************************************'''
        pred_file = os.path.join('./testing_result/', 'pred.csv')
        backbone = Backbone(preprocessor.embedding_matrix, **net_config)
        header = Header(**header_config)
        backbone = backbone.to(device)
        header = header.to(device)
        run_testing(test_loader, backbone, header, device, pred_file)

        # from google.colab import files
        # files.download(pred_file)