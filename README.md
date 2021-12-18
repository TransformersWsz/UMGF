# UMGF: Multi-modal Graph Fusion for Named Entity Recognition with Targeted Visual Guidance

This repository contains the source code for the paper: **UMGF: Multi-modal Graph Fusion for Named Entity Recognition with Targeted Visual
Guidance**


## Install

- python3.7
- transformers==3.4.0
- torch==1.7.1
- pytorch-crf==0.7.2
- pillow==7.1.2


## Dataset

- You can download original data from [UMT](https://github.com/jefferyYu/UMT/)

## Preprocess

### Image
1. Download twitter images from [UMT](https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view)
2. To detect visual objects, please follow [onestage_grounding
](https://github.com/TransformersWsz/onestage_grounding/blob/master/my_readme.md) or you can directly download them from [twitter2015_img.tar.gz(password: l75t)](https://pan.baidu.com/s/1DCACHmDKYiW21Vnmn6YIvQ) and [twitter2017_img.tar.gz(password: kvc5)](https://pan.baidu.com/s/1UCdUgyspUBHiM8DvoF-R_A)
3. Unzip and put the images under the corresponding folder(e.g. `./data/twitter2015/image`)

### Text
- The proprocessed text has been put under `./my_data/` folder

## Run

### Train

```bash
python ddp_mmner.py --do_train --txtdir=./my_data/twitter2015 --imgdir=./data/twitter2015/image --ckpt_path=./model.pt --num_train_epoch=30 --train_batch_size=16 --lr=0.0001 --seed=2019
```

### Test

```bash
python ddp_mmner.py --do_test --txtdir=./my_data/twitter2015 --imgdir=./data/twitter2015/image --ckpt_path=./ddp_mner.pt --test_batch_size=32
```
- [Checkpoint on twitter2015(password: j9ib)](https://pan.baidu.com/s/1pa7xRJofE3oru3EsE2NB2w) has beed provided.

## Acknowledgements
- Using these two datasets means you have read and accepted the copyrights set by Twitter and dataset providers.
- Part of the code are from:
    - [GMNMT](https://github.com/middlekisser/GMNMT)
    - [UMT](https://github.com/jefferyYu/UMT/)
    - [onestage_grounding](https://github.com/zyang-ur/onestage_grounding)