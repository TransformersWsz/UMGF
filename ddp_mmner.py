# -*- encoding: utf-8 -*-
'''
@File     : ddp_mner.py
@DateTime : 2020/08/31 00:06:36
@Author   : Swift
@Desc     : reset twitter2015 model
'''


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms
from transformers import BertTokenizer
from transformers import BertModel
from torchcrf import CRF
from PIL import Image
import os
import glob
import numpy as np
import time
import random
import argparse
from tqdm import tqdm
import warnings

from model.utils import *
from metric import evaluate_pred_file
from config import tag2idx, idx2tag, max_len, max_node, log_fre

warnings.filterwarnings("ignore")
pre_file = "./output/twitter2017/{}/epoch_{}.txt"
device = torch.device("cuda:3")


def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class MMNerDataset(Dataset):
    def __init__(self, textdir, imgdir="./data/twitter2015/image"):
        self.X_files = sorted(glob.glob(os.path.join(textdir, "*_s.txt")))
        self.Y_files = sorted(glob.glob(os.path.join(textdir, "*_l.txt")))
        self.P_files = sorted(glob.glob(os.path.join(textdir, "*_p.txt")))
        self._imgdir = imgdir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def __len__(self):
        return len(self.X_files)

    def construct_inter_matrix(self, word_num, pic_num=max_node):
        mat = np.zeros((max_len, pic_num), dtype=np.float32)
        mat[:word_num, :pic_num] = 1.0
        return mat

    def __getitem__(self, idx):
        with open(self.X_files[idx], "r", encoding="utf-8") as fr:
            s = fr.readline().split("\t")

        with open(self.Y_files[idx], "r", encoding="utf-8") as fr:
            l = fr.readline().split("\t")

        with open(self.P_files[idx], "r", encoding="utf-8") as fr:
            imgid = fr.readline()
            picpaths = [os.path.join(self._imgdir, "{}/{}_{}.jpg".format(imgid, entity, imgid))
                        for entity in ["crop_person", "crop_miscellaneous", "crop_location", "crop_organization"]]

        ntokens = ["[CLS]"]
        label_ids = [tag2idx["CLS"]]
        for word, label in zip(s, l):    # iterate every word
            tokens = self.tokenizer._tokenize(word)    # one word may be split into several tokens
            ntokens.extend(tokens)
            for i, _ in enumerate(tokens):
                label_ids.append(tag2idx[label] if i == 0 else tag2idx["X"])
        ntokens = ntokens[:max_len-1]
        ntokens.append("[SEP]")
        label_ids = label_ids[:max_len-1]
        label_ids.append(tag2idx["SEP"])

        matrix = self.construct_inter_matrix(len(label_ids), len(picpaths))

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        mask = [1] * len(input_ids)
        segment_ids = [0] * max_len

        pad_len = max_len - len(input_ids)
        rest_pad = [0] * pad_len    # pad to max_len
        input_ids.extend(rest_pad)
        mask.extend(rest_pad)
        label_ids.extend(rest_pad)

        # pad ntokens
        ntokens.extend(["pad"] * pad_len)

        return {
            "ntokens": ntokens,
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "mask": mask,
            "label_ids": label_ids,
            "picpaths": picpaths,
            "matrix": matrix
        }


def collate_fn(batch):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    label_ids = []

    b_ntokens = []
    b_matrix = []
    b_img = torch.zeros(len(batch)*max_node, 3, 224, 224)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    for idx, example in enumerate(batch):
        b_ntokens.append(example["ntokens"])
        input_ids.append(example["input_ids"])
        token_type_ids.append(example["segment_ids"])
        attention_mask.append(example["mask"])
        label_ids.append(example["label_ids"])
        b_matrix.append(example["matrix"])

        for i, picpath in enumerate(example["picpaths"]):
            try:
                b_img[idx*max_node+i] = preprocess(Image.open(picpath).convert('RGB'))
            except:
                print("========={} error!===============".format(picpath))
                exit(1)
            

    return {
        "b_ntokens": b_ntokens,
        "x": {
            "input_ids": torch.tensor(input_ids).to(device),
            "token_type_ids": torch.tensor(token_type_ids).to(device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.uint8).to(device)
        },
        "b_img": torch.tensor(b_img).to(device),
        "b_matrix": torch.tensor(b_matrix).to(device),
        "y": torch.tensor(label_ids).to(device)
    }


class MMNerModel(nn.Module):

    def __init__(self, d_model=512, d_hidden=256, n_heads=8, dropout=0.4, layer=6, tag2idx=tag2idx):
        super(MMNerModel, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.resnet = models.resnet152(pretrained=True)
        self.crf = CRF(len(tag2idx), batch_first=True)
        # self.hidden2tag = nn.Linear(2*d_model, len(tag2idx))
        self.hidden2tag = nn.Linear(1280, len(tag2idx))

        objcnndim = 2048
        fc_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(
            in_features=fc_feats, out_features=objcnndim, bias=True)

        self.layer = layer
        self.dp = dropout
        self.d_model = d_model
        self.hid = d_hidden

        self.trans_txt = nn.Linear(768, d_model)
        self.trans_obj = nn.Sequential(Linear(objcnndim, d_model), nn.ReLU(), nn.Dropout(dropout),
                                       Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout))

        # text
        self.mhatt_x = clone(MultiHeadedAttention(
            n_heads, d_model, dropout), layer)
        self.ffn_x = clone(PositionwiseFeedForward(d_model, d_hidden), layer)
        self.res4ffn_x = clone(SublayerConnectionv2(d_model, dropout), layer)
        self.res4mes_x = clone(SublayerConnectionv2(d_model, dropout), layer)

        # img
        self.mhatt_o = clone(MultiHeadedAttention(
            n_heads, d_model, dropout, v=0, output=0), layer)
        self.ffn_o = clone(PositionwiseFeedForward(d_model, d_hidden), layer)
        self.res4mes_o = clone(SublayerConnectionv2(d_model, dropout), layer)
        self.res4ffn_o = clone(SublayerConnectionv2(d_model, dropout), layer)

        self.mhatt_x2o = clone(Linear(d_model * 2, d_model), layer)
        self.mhatt_o2x = clone(Linear(d_model * 2, d_model), layer)
        self.xgate = clone(SublayerConnectionv2(d_model, dropout), layer)
        self.ogate = clone(SublayerConnectionv2(d_model, dropout), layer)

    def log_likelihood(self, x, b_img, inter_matrix, text_mask, tags):
        """
        inter_matrix: batch, max_len, 4
        text_mask: batch, max_len
        """
        batch_size, xn, objn = inter_matrix.size(0), inter_matrix.size(1), inter_matrix.size(2)
        inter_matrix = inter_matrix.unsqueeze(-1)
        matrix4obj = torch.transpose(inter_matrix, 1, 2)
        crf_mask = x["attention_mask"]
        x = self.bert(**x)[0]
        o = self.resnet(b_img).view(batch_size, max_node, -1)
        o = self.trans_obj(o)

        bert_x = x.clone()    # reserve origin bert output (9, 48, 768)
        x = self.trans_txt(x)    # 9, 48, 512
        
        for i in range(self.layer):
            # Text self-attention: batch, max_len, dim
            newx = self.res4mes_x[i](x, self.mhatt_x[i](
                x, x, x, text_mask.unsqueeze(1)))

            # Visual self-attention: batch, 4, odim
            newo = self.res4mes_o[i](o, self.mhatt_o[i](o, o, o, None))

            # Text to Image Gating
            newx_ep = newx.unsqueeze(2).expand(
                batch_size, max_len, objn, newx.size(-1))
            o_ep = newo.unsqueeze(1).expand(batch_size, xn, objn, o.size(-1))
            # batch, xn, objn, dmodel
            x2o_gates = torch.sigmoid(
                self.mhatt_x2o[i](torch.cat((newx_ep, o_ep), -1)))
            x2o = (x2o_gates * inter_matrix * o_ep).sum(2)

            # Image to Text Gating
            x_ep = newx.unsqueeze(1).expand(batch_size, objn, xn, newx.size(-1))
            newo_ep = newo.unsqueeze(2).expand(batch_size, objn, xn, o.size(-1))
            # B O T H
            o2x_gates = torch.sigmoid(
                self.mhatt_o2x[i](torch.cat((x_ep, newo_ep), -1)))
            o2x = (o2x_gates * matrix4obj * x_ep).sum(2)

            newx = self.xgate[i](newx, x2o)
            newo = self.ogate[i](newo, o2x)

            # 9, 48, 512
            x = self.res4ffn_x[i](newx, self.ffn_x[i](newx))
            o = self.res4ffn_o[i](newo, self.ffn_o[i](newo))
        x = torch.cat((bert_x, x), dim=2)
        x = self.hidden2tag(x)
        return -self.crf(x, tags, mask=crf_mask)

    def forward(self, x, b_img, inter_matrix, text_mask, tags):
        """
        inter_matrix: batch, max_len, 4
        text_mask: batch, max_len
        """
        batch_size, xn, objn = inter_matrix.size(0), inter_matrix.size(1), inter_matrix.size(2)
        inter_matrix = inter_matrix.unsqueeze(-1)
        matrix4obj = torch.transpose(inter_matrix, 1, 2)
        crf_mask = x["attention_mask"]
        x = self.bert(**x)[0]
        o = self.resnet(b_img).view(batch_size, max_node, -1)
        o = self.trans_obj(o)

        bert_x = x.clone()    # reserve origin bert output (9, 48, 768)
        x = self.trans_txt(x)
        for i in range(self.layer):
            # Text self-attention: batch, max_len, dim
            newx = self.res4mes_x[i](x, self.mhatt_x[i](
                x, x, x, text_mask.unsqueeze(1)))

            # Visual self-attention: batch, 4, odim
            newo = self.res4mes_o[i](o, self.mhatt_o[i](o, o, o, None))

            # Text to Image Gating
            newx_ep = newx.unsqueeze(2).expand(
                batch_size, max_len, objn, newx.size(-1))
            o_ep = newo.unsqueeze(1).expand(batch_size, xn, objn, o.size(-1))
            # batch, xn, objn, dmodel
            x2o_gates = torch.sigmoid(
                self.mhatt_x2o[i](torch.cat((newx_ep, o_ep), -1)))
            x2o = (x2o_gates * inter_matrix * o_ep).sum(2)

            # Image to Text Gating
            x_ep = newx.unsqueeze(1).expand(batch_size, objn, xn, newx.size(-1))
            newo_ep = newo.unsqueeze(2).expand(batch_size, objn, xn, o.size(-1))
            # B O T H
            o2x_gates = torch.sigmoid(
                self.mhatt_o2x[i](torch.cat((x_ep, newo_ep), -1)))
            o2x = (o2x_gates * matrix4obj * x_ep).sum(2)

            newx = self.xgate[i](newx, x2o)
            newo = self.ogate[i](newo, o2x)

            x = self.res4ffn_x[i](newx, self.ffn_x[i](newx))
            o = self.res4ffn_o[i](newo, self.ffn_o[i](newo))
        x = torch.cat((bert_x, x), dim=2)
        x = self.hidden2tag(x)
        return self.crf.decode(x, mask=crf_mask)


def save_model(model, model_path="./model.pt"):
    torch.save(model.state_dict(), model_path)
    print("Current Best mmner model has beed saved!")


def predict(epoch, model, dataloader, mode="val", res=None):
    model.eval()
    with torch.no_grad():
        filepath = pre_file.format(mode, epoch)
        with open(filepath, "w", encoding="utf8") as fw:
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Predicting"):
                b_ntokens = batch["b_ntokens"]

                x = batch["x"]
                b_img = batch["b_img"]
                inter_matrix = batch["b_matrix"]
                text_mask = x["attention_mask"]
                y = batch["y"]
                output = model(x, b_img, inter_matrix, text_mask, y)

                # write into file
                for idx, pre_seq in enumerate(output):
                    ground_seq = y[idx]
                    for pos, (pre_idx, ground_idx) in enumerate(zip(pre_seq, ground_seq)):
                        if ground_idx == tag2idx["PAD"] or ground_idx == tag2idx["X"] or ground_idx == tag2idx["CLS"] or ground_idx == tag2idx["SEP"]:
                            continue
                        else:
                            predict_tag = idx2tag[pre_idx] if idx2tag[pre_idx] not in [
                                "PAD", "X", "CLS", "SEP"] else "O"
                            true_tag = idx2tag[ground_idx.data.item()]
                            line = "{}\t{}\t{}\n".format(b_ntokens[idx][pos], predict_tag, true_tag)
                            fw.write(line)
        print("=============={} -> {} epoch eval done=================".format(mode, epoch))
        cur_f1 = evaluate_pred_file(filepath)
        to_save = False
        if mode == "test":
            if res["best_f1"] < cur_f1:
                res["best_f1"] = cur_f1
                res["epoch"] = epoch
                to_save = True
            print("current best f1: {}, epoch: {}".format(res["best_f1"], res["epoch"]))
        return to_save


def train(args):
    # make initialized weight of each model replica same
    seed_torch(args.seed)

    train_textdir = os.path.join(args.txtdir, "train")
    train_dataset = MMNerDataset(textdir=train_textdir, imgdir=args.imgdir)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)

    val_textdir = os.path.join(args.txtdir, "valid")
    val_dataset = MMNerDataset(textdir=val_textdir, imgdir=args.imgdir)
    val_dataloader = DataLoader(val_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)

    test_textdir = os.path.join(args.txtdir, "test")
    test_dataset = MMNerDataset(textdir=test_textdir, imgdir=args.imgdir)
    test_dataloader = DataLoader(test_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn)

    model = MMNerModel()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.8)

    res = {}
    res["best_f1"] = 0.0
    res["epoch"] = -1
    start = time.time()
    for epoch in range(args.num_train_epoch):
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = batch["x"]
            b_img = batch["b_img"]
            inter_matrix = batch["b_matrix"]
            text_mask = x["attention_mask"]
            y = batch["y"]

            loss = model.log_likelihood(x, b_img, inter_matrix, text_mask, y)
            loss.backward()
            optimizer.step()

            if i % log_fre == 0:
                print("EPOCH: {} Step: {} Loss: {}".format(epoch, i, loss.data))

        scheduler.step()
        predict(epoch, model, val_dataloader, mode="val", res=res)
        to_save = predict(epoch, model, test_dataloader, mode="test", res=res)
        if to_save:    # whether to save the best checkpoint
            save_model(model, args.ckpt_path)

    print("================== train done! ================")
    end = time.time()
    hour = int((end-start)//3600)
    minute = int((end-start)%3600//60)
    print("total time: {} hour - {} minute".format(hour, minute))


def test(args):
    model = MMNerModel().to(device)
    model.load_state_dict(torch.load(args.ckpt_path))

    test_textdir = os.path.join(args.txtdir, "test")
    test_dataset = MMNerDataset(textdir=test_textdir, imgdir=args.imgdir)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=collate_fn)

    model.eval()
    with torch.no_grad():
        filepath = "./test_output.txt"
        with open(filepath, "w", encoding="utf8") as fw:
            for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing"):
                b_ntokens = batch["b_ntokens"]

                x = batch["x"]
                b_img = batch["b_img"]
                inter_matrix = batch["b_matrix"]
                text_mask = x["attention_mask"]
                y = batch["y"]
                output = model(x, b_img, inter_matrix, text_mask, y)

                # write into file
                for idx, pre_seq in enumerate(output):
                    ground_seq = y[idx]
                    for pos, (pre_idx, ground_idx) in enumerate(zip(pre_seq, ground_seq)):
                        if ground_idx == tag2idx["PAD"] or ground_idx == tag2idx["X"] or ground_idx == tag2idx["CLS"] or ground_idx == tag2idx["SEP"]:
                            continue
                        else:
                            predict_tag = idx2tag[pre_idx] if idx2tag[pre_idx] not in [
                                "PAD", "X", "CLS", "SEP"] else "O"
                            true_tag = idx2tag[ground_idx.data.item()]
                            line = "{}\t{}\t{}\n".format(b_ntokens[idx][pos], predict_tag, true_tag)
                            fw.write(line)
        evaluate_pred_file(filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--txtdir', 
                        type=str, 
                        default="./my_data/twitter2015/", 
                        help="text dir")
    parser.add_argument('--imgdir', 
                        type=str, 
                        default="./data/twitter2015/image/", 
                        help="image dir")
    parser.add_argument('--ckpt_path', 
                        type=str, 
                        default="./model.pt", 
                        help="path to save or load model")
    parser.add_argument("--num_train_epoch",
                        default=30,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--test_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--lr",
                        default=0.0001,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=2021,
                        help="random seed for initialization")
    args = parser.parse_args()
    if args.do_train:
        train(args)
    elif args.do_test:
        test(args)
    else:
        raise ValueError('At least one of `do_train`, `do_eval` must be True.')


if __name__ == "__main__":
    main()
