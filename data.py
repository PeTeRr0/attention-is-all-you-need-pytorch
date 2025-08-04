import torch
import datasets
import spacy
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# !pip install -q datasets spacy
# !python -m spacy download en_core_web_sm
# !python -m spacy download de_core_news_sm

# Tokenizer
spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

# Data loading
raw_data = datasets.load_dataset("opus_books", "de-en")
raw_data = raw_data["train"].train_test_split(test_size=0.1)
train_data = raw_data["train"]
val_data = raw_data["test"]

# Max length of sentence
MAX_LEN = 40

# Data preprocessing
def preprocess(data):
    src_tokenized = []
    tgt_tokenized = []
    for ex in data:
        src = tokenize_en(ex["translation"]["en"])
        tgt = tokenize_de(ex["translation"]["de"])
        if len(src) <= MAX_LEN and len(tgt) <= MAX_LEN:
            src_tokenized.append(["<sos>"] + src + ["<eos>"])
            tgt_tokenized.append(["<sos>"] + tgt + ["<eos>"])
    return src_tokenized, tgt_tokenized

src_sentences, tgt_sentences = preprocess(train_data)

# Generating vocabulary
def build_vocab(sentences, min_freq=2):
    counter = Counter()
    for sent in sentences:
        counter.update(sent)
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab

src_vocab = build_vocab(src_sentences)
tgt_vocab = build_vocab(tgt_sentences)

# Tensor transformation
def numericalize(sent, vocab):
    return [vocab.get(tok, vocab["<unk>"]) for tok in sent]

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
        self.src = src_sentences
        self.tgt = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_tensor = torch.tensor(numericalize(self.src[idx], self.src_vocab))
        tgt_tensor = torch.tensor(numericalize(self.tgt[idx], self.tgt_vocab))
        return src_tensor, tgt_tensor

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=src_vocab["<pad>"])
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_vocab["<pad>"])
    return src_batch, tgt_batch

# DataLoader
train_dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Verification
for src, tgt in train_loader:
    print("SRC (input):", src.shape)  # [batch_size, src_len]
    print("TGT (target):", tgt.shape)  # [batch_size, tgt_len]
    break

# Total number of vocabulary
print(f"Source Vocab Size: {len(src_vocab)}")
print(f"Target Vocab Size: {len(tgt_vocab)}")
