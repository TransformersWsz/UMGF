max_len = 128
max_node = 4
log_fre = 10
tag2idx = {
    "PAD": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-OTHER": 7,
    "I-OTHER": 8,
    "O": 9,
    "X": 10,
    "CLS": 11,
    "SEP": 12
}

idx2tag = {idx: tag for tag, idx in tag2idx.items()}
