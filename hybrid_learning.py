import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertModel
from transformers import BertTokenizer

from util import get_args


class HybridModel(nn.Module):

    def __init__(self, num_features, freeze_bert=False, is_regressor=True):
        super(HybridModel, self).__init__()
        # Instantiating BERT model object
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        self.num_features = num_features
        self.is_regressor = is_regressor

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # classification layers
        self.classification_layer = nn.Linear(768, self.num_features)
        self.sigmoid = nn.Sigmoid()
        # self.attention_layer = nn.Sequential(nn.Linear(768, self.num_features), nn.Softmax(dim=1))
        self.regression_layer = nn.Linear(768, 1)

    def set_as_classifier(self):
        self.is_regressor = False

    def set_as_regressor(self):
        self.is_regressor = True

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        # Feeding the input to BERT model to obtain contextualized representations
        cont_reps, _ = self.bert_layer(seq, attention_mask=attn_masks)

        # Obtaining the representation of [CLS] head
        cls_rep = cont_reps[:, 0]

        # Feeding cls_rep to the regression (when evaluating)/classification (when fine-tuning) layer
        if self.is_regressor:
            regression_score = self.regression_layer(cls_rep)
            return regression_score
        else:
            affinity = self.classification_layer(cls_rep)
            probs = self.sigmoid(affinity)
            return probs


class SymmetryHybridDataset(Dataset):

    def __init__(self, df, ling_fts, maxlen):

        # Store the contents of the file in a pandas dataframe
        self.df = df.reset_index(drop=True)
        self.ling_fts = ling_fts

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # Selecting the sentence and label at the specified index in the data frame
        sentence = self.df['sentence'][index]
        sent_score = self.df['sentence average score'][index]
        features = self.ling_fts[index]

        # Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(sentence)  # Tokenize the sentence
        tokens = ['[CLS]'] + tokens + [
            '[SEP]']  # Insering the CLS and SEP token in the beginning and end of the sentence
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]  # Padding sentences
        else:
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']  # Prunning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(
            tokens)  # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids)  # Converting the list to a pytorch tensor

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, sent_score, features


def train(model, criterion, opti, train_loader, val_loader, max_eps=5, print_every=10, is_regression=False):
    best_eval_loss = -1
    best_model = model
    # print("len of training data: ", len(train_loader.dataset.df))
    for ep in range(max_eps):
        model.train()
        for it, (seq, attn_masks, sent_scores, features) in enumerate(train_loader):

            # Clear gradients
            opti.zero_grad()
            # Converting these to cuda tensors
            seq, attn_masks, sent_scores, features = seq.cuda(), attn_masks.cuda(), sent_scores.cuda(), features.cuda()  # features: shape (batch_size, num_features)

            # Computing loss
            if is_regression:
                scores = model(seq, attn_masks)
                loss = criterion(scores.squeeze(-1), sent_scores.float())
            else:
                pred_features = model(seq, attn_masks)
                loss = criterion(pred_features.squeeze(-1), features.float())

            # Backpropagating the gradients
            loss.backward()

            # Optimization step
            opti.step()

            # print("Iteration {} of epoch {} complete.".format(it+1, ep+1))
            if (it + 1) % print_every == 0:
                print("Iteration {} of epoch {} complete. Train Loss : {}".format(it + 1, ep + 1, loss.item()))
        # evaluation step after each epoch is done
        eval_loss = evaluate(model, val_loader, criterion, is_regression)
        print("Epoch {}, validation loss: {}".format(ep + 1, eval_loss))
        if best_eval_loss == -1:
            best_eval_loss = eval_loss
            best_model = deepcopy(model)
        elif eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_model = deepcopy(model)
    print("best eval loss: ", best_eval_loss)
    return best_model


def evaluate(model, val_loader, criterion, is_regression=True):
    model.eval()
    total_loss = []
    for it, (seq, attn_masks, sent_scores, features) in enumerate(val_loader):
        seq, attn_masks, sent_scores, features = seq.cuda(), attn_masks.cuda(), sent_scores.cuda(), features.cuda()  # features: shape (batch_size, num_features)
        if is_regression:
            scores = model(seq, attn_masks)
            loss = criterion(scores.squeeze(-1), sent_scores.float())
        else:
            pred_features = model(seq, attn_masks)
            loss = criterion(pred_features.squeeze(-1), features.float())
        total_loss.append(loss.item())
    return np.array(total_loss).mean()


def predict_sent_scores(model, val_loader):
    model.eval()
    model.set_as_regressor()
    total_loss = 0
    scores = []
    for it, (seq, attn_masks, sent_scores, features) in enumerate(val_loader):
        seq, attn_masks, sent_scores, features = seq.cuda(), attn_masks.cuda(), sent_scores.cuda(), features.cuda()  # features: shape (batch_size, num_features)
        scores.append(model(seq, attn_masks).squeeze().cpu().detach().numpy())
    return np.concatenate(scores)


def main(args):
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        print("using gpu acceleration")

    ft_criterion = nn.BCELoss()
    reg_criterion = nn.MSELoss()

    num_features = 17
    new_df = pd.read_csv('./data/sis.csv')
    ling_fts = pickle.load(open('./data/sis_ling_fts.p', 'rb'))
    pred_ids = list(set(list(new_df['pred_id'])))

    # first training task: predicting feature labels
    m = len(list(new_df['sentence']))
    num_train = int(0.7 * m)
    # print(num_train)

    shuffled_idx = np.arange(m)
    np.random.shuffle(shuffled_idx)
    train_idx = shuffled_idx[:num_train]
    eval_idx = shuffled_idx[num_train:]

    train_df = new_df.loc[train_idx]
    eval_df = new_df.loc[eval_idx]

    train_ling_fts = ling_fts[train_idx]
    eval_ling_fts = ling_fts[eval_idx]

    train_set = SymmetryHybridDataset(train_df, train_ling_fts, maxlen=30)
    eval_set = SymmetryHybridDataset(eval_df, eval_ling_fts, maxlen=30)

    train_loader = DataLoader(train_set, batch_size=32, num_workers=4)
    val_loader = DataLoader(eval_set, batch_size=32, num_workers=4)

    classifier = HybridModel(num_features=num_features, freeze_bert=False).cuda()
    classifier.set_as_classifier()
    opti = optim.Adam(classifier.parameters(), lr=1e-5)
    classifier = train(model=classifier, criterion=ft_criterion, opti=opti, train_loader=train_loader,
                       val_loader=val_loader, max_eps=20, is_regression=False)
    torch.save(classifier, 'fine_tuned_bert_classifier.pth')

    # second training task: sentence score regression

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        print("using gpu acceleration")

    prediction_scores = []
    classifier.set_as_regressor()

    for pred_id in tqdm(pred_ids):
        regressor = deepcopy(classifier)
        regressor.set_as_regressor()

        train_idx = list(new_df.index[new_df['pred_id'] != pred_id])
        eval_idx = list(new_df.index[new_df['pred_id'] == pred_id])

        train_df = new_df.loc[train_idx]
        eval_df = new_df.loc[eval_idx]

        train_ling_fts = ling_fts[train_idx]
        eval_ling_fts = ling_fts[eval_idx]

        train_set = SymmetryHybridDataset(train_df, train_ling_fts, maxlen=30)
        eval_set = SymmetryHybridDataset(eval_df, eval_ling_fts, maxlen=30)

        train_loader = DataLoader(train_set, batch_size=32, num_workers=4)
        val_loader = DataLoader(eval_set, batch_size=32, num_workers=4)

        regressor.set_as_regressor()
        opti = optim.Adam(regressor.parameters(), lr=1e-4)
        regressor = train(model=regressor, criterion=reg_criterion, opti=opti, train_loader=train_loader,
                          val_loader=val_loader, max_eps=30, is_regression=True)
        # predict sentence scores for the eval verb
        prediction_scores.append(predict_sent_scores(regressor, val_loader))
        del regressor, opti
        torch.cuda.empty_cache()

    prediction_scores = np.concatenate(prediction_scores)
    new_df['hybrid prediction score'] = pd.Series(prediction_scores)
    new_df.to_csv('./data/sis.csv', index=False)
    human_scores = np.array(list(new_df['sentence average score']))
    mse = np.square(human_scores - prediction_scores).mean()
    correlation = pearsonr(human_scores, prediction_scores)
    print("Training {} model is complete.".format(args.model_type))
    print("Mean square error between model prediction and human judgement: {}".format(mse))
    print("Correlation between model prediction and human judgement: {}".format(correlation))


if __name__ == '__main__':
    args = get_args()
    main(args)
