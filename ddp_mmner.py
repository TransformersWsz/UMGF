# -*- encoding: utf-8 -*-
'''
@File     : ddp_mner.py
@DateTime : 2020/08/31 00:06:36
@Author   : Swift
@Desc     : reset twitter2015 model
'''


from config import tags, tag2idx, idx2tag, max_len, max_node, EPOCH, BATCH_SIZE, log_fre
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import models
from torchvision import transforms
from transformers import BertTokenizer
from transformers import BertModel
from torchcrf import CRF
from model.util import *

from PIL import Image
import os
import glob
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

model_path = "./ddp_mner.pt"
TEXTDIR = "/home/swift/mmner/var/{}"
IMGDIR = "/home/swift/mmner/data/twitter2015/image"
pre_file = "/home/swift/mmner/output/ddp/{}/{}_{}.txt"
max_node = 4
BATCH_SIZE = 8
EPOCH = 50

# initialization
torch.distributed.init_process_group(backend="nccl")

# allocate gpu for each replicated model
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def seed_torch(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class MMNerDataset(Dataset):
    def __init__(self, textdir, imgdir=IMGDIR):
        self.X_files = sorted(glob.glob(os.path.join(textdir, "*_s.txt")))
        self.Y_files = sorted(glob.glob(os.path.join(textdir, "*_l.txt")))
        self.P_files = sorted(glob.glob(os.path.join(textdir, "*_p.txt")))
        self._imgdir = imgdir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def __len__(self):
        return len(self.X_files)

    def construct_inter_matrix(self, word_num, pic_num=max_node):
        mat = np.zeros((max_len, pic_num), dtype=np.float)
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
        for word, label in zip(s, l):    # 遍历每个单词
            tokens = self.tokenizer._tokenize(word)    # 一个单词可能会被分成多个token
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
        rest_pad = [0] * pad_len    # pad成max_len
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


def save_model(model):
    torch.save(model.state_dict(), model_path)
    print("MMNer model has beed saved!")


def predict(epoch, model, val_dataloader, mode="val"):
    model.eval()
    with open(pre_file.format(mode, local_rank, epoch), "w", encoding="utf8") as fw:
        for i, batch in enumerate(val_dataloader):
            b_ntokens = batch["b_ntokens"]

            x = batch["x"]
            b_img = batch["b_img"]
            inter_matrix = batch["b_matrix"]
            text_mask = x["attention_mask"]
            y = batch["y"]
            output = model(x, b_img, inter_matrix, text_mask, y)

            # 写入文件
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
    if local_rank == 0:
        print("=============={} -> {} epoch eval done=================".format(mode, epoch))


if __name__ == "__main__":

    # make initialized weight of each model replica same
    seed_torch(2019)

    train_dataset = MMNerDataset(textdir=TEXTDIR.format("train"))
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn)

    val_dataset = MMNerDataset(textdir=TEXTDIR.format("valid"))
    val_sampler = DistributedSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, collate_fn=collate_fn)

    test_dataset = MMNerDataset(textdir=TEXTDIR.format("test"))
    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, collate_fn=collate_fn)

    model = MMNerModel()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.8)

    for epoch in range(EPOCH):
        train_sampler.set_epoch(epoch)
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

            if i % log_fre == 0 and local_rank == 0:
                print("EPOCH: {} Step: {} Loss: {}".format(epoch, i, loss.data))

        scheduler.step()
        # predict(epoch, model, val_dataloader, mode="val")
        predict(epoch, model, test_dataloader, mode="test")
    print("================== train done! ================")
    save_model(model)
