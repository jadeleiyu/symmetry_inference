from copy import deepcopy
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertModel
from transformers import BertTokenizer

from util import get_args


class BertRegressor(nn.Module):

    def __init__(self, freeze_bert=True):
        super(BertRegressor, self).__init__()
        # Instantiating BERT model object
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Linear layer
        self.cls_layer = nn.Linear(768, 1)
        # self.sigmoid = nn.Sigmoid()

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

        # Feeding cls_rep to the regression layer
        scores = self.cls_layer(cls_rep)
        # scores = self.sigmoid(scores)

        return scores


def train(model, criterion, opti, train_loader, val_loader, max_eps=5, print_every=10):
    best_eval_loss = -1
    best_model = model
    # print("len of training data: ", len(train_loader.dataset.df))
    for ep in range(max_eps):
        model.train()
        for it, (seq, attn_masks, labels) in enumerate(train_loader):
            # print("shape of labels:", labels.shape)
            # Clear gradients
            opti.zero_grad()
            # Converting these to cuda tensors
            seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()

            # Obtaining the logits from the model
            scores = model(seq, attn_masks)

            # Computing loss
            loss = criterion(scores.squeeze(-1), labels.float())

            # Backpropagating the gradients
            loss.backward()

            # Optimization step
            opti.step()

            # print("Iteration {} of epoch {} complete.".format(it+1, ep+1))
            if (it + 1) % print_every == 0:
                print("Iteration {} of epoch {} complete. Train Loss : {}".format(it + 1, ep + 1, loss.item()))
        # evaluation step after each epoch is done
        eval_loss = evaluate(model, val_loader, criterion)
        print("Epoch {}, validation loss: {}".format(ep + 1, eval_loss))
        if best_eval_loss == -1:
            best_eval_loss = eval_loss
            best_model = deepcopy(model)
        elif eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_model = deepcopy(model)
    print("best eval loss: ", best_eval_loss)
    return best_model


class SymmetryDataset(Dataset):

    def __init__(self, df, maxlen):

        # Store the contents of the file in a pandas dataframe
        self.df = df.reset_index(drop=True)

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # Selecting the sentence and label at the specified index in the data frame

        sentence = self.df['sentence'][index]
        label = self.df['sentence average score'][index]

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

        return tokens_ids_tensor, attn_mask, label


def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    for it, (seq, attn_masks, labels) in enumerate(val_loader):
        seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()
        scores = model(seq, attn_masks)
        loss = criterion(scores.squeeze(-1), labels.float())
        total_loss += loss.item()
    return total_loss


def predict(model, val_loader):
    model.eval()
    predictions = []
    for it, (seq, attn_masks, labels) in enumerate(val_loader):
        seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()
        scores = model(seq, attn_masks).squeeze().cpu().detach().numpy()
        predictions.append(scores)
    return np.concatenate(predictions)


def main(args):
    # turn on cuda if necessary
    cuda_available = torch.cuda.is_available()
    if args.cuda() and cuda_available:
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        print("using gpu acceleration")

    mse_loss = nn.MSELoss()
    sis_df = pd.read_csv('./data/sis.csv')
    pred_ids = list(set(list(sis_df['pred_id'])))
    bert_predictions = []
    # eval_losses = []
    for pred_id in tqdm(pred_ids):
        train_df = sis_df.loc[sis_df['pred_id'] != pred_id]
        eval_df = sis_df.loc[sis_df['pred_id'] == pred_id]
        train_set = SymmetryDataset(train_df, maxlen=30)
        eval_set = SymmetryDataset(eval_df, maxlen=30)

        train_loader = DataLoader(train_set, batch_size=32, num_workers=4)
        val_loader = DataLoader(eval_set, batch_size=32, num_workers=4)

        encoder = BertRegressor(freeze_bert=False).cuda()
        opti = optim.Adam(encoder.parameters(), lr=2e-5)
        encoder = train(model=encoder, criterion=mse_loss, opti=opti, train_loader=train_loader,
                        val_loader=val_loader)

        predictions = predict(encoder, val_loader)
        print(predictions)
        bert_predictions.append(predictions)

        # model_fn = './bert_regressor/bert_regressor_pred' + str(pred_id) + '.pt'
        # torch.save(encoder.state_dict(), model_fn)
        del encoder, opti
        torch.cuda.empty_cache()

    bert_predictions = np.concatenate(bert_predictions)
    sis_df['bert tuned prediction'] = pd.Series(bert_predictions)
    sis_df.to_csv('./data/sis.csv', index=False)
    human_scores = np.array(list(sis_df['sentence average score']))
    mse = np.square(human_scores - bert_predictions).mean()
    correlation = pearsonr(human_scores, bert_predictions)
    print("Training {} model is complete.".format(args.model_type))
    print("Mean square error between model prediction and human judgement: {}".format(mse))
    print("Correlation between model prediction and human judgement: {}".format(correlation))


if __name__ == '__main__':
    args = get_args()
    main(args)

