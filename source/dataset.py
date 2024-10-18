from datasets import load_dataset
import os
import torch
from torch.utils.data import DataLoader
from source.utils import load_data, save_data
from tqdm import tqdm

class KoEnTranslation():
    def __init__(self, vocab_path, merge_path, max_seq_len=256, sos_idx=0, eos_idx=2, pad_idx=1, unk_idx=3):
        self.dataset_name = 'aihub-koen-translation-integrated-base-1m'
        self.hf_url = 'traintogpb/aihub-koen-translation-integrated-base-1m'
        self.max_seq_len = max_seq_len
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.tokenizer = self.build_tokenizer(vocab_path, merge_path)

        self.ds_train, self.ds_val, self.ds_test = None, None, None
        self.build_dataset()

        self.train = None
        self.val = None
        self.test = None
        self.build_vocab()



    def build_tokenizer(self, vocab_path, merge_path):
        from tokenizers import ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer(vocab_path, merge_path)
        special_tokens = [self.sos_idx, self.eos_idx, self.pad_idx, self.unk_idx]
        tokenizer.add_special_tokens([tokenizer.id_to_token(i) for i in special_tokens])
        return tokenizer


    def build_dataset(self, cache_dir='./dataset'):
        cache_dir = os.path.join(cache_dir, self.dataset_name)
        os.makedirs(cache_dir, exist_ok=True)

        ds = load_dataset(self.hf_url)
        self.ds_train = ds['train']
        self.ds_val = ds['validation']
        self.ds_test = ds['test']


    def encode_and_transform(self, vocab_tup):
        ids_list = []
        for ko, en in tqdm(vocab_tup):
            # Ko text tokenizing 후 sos, eos 토큰 추가
            ko_ids = self.tokenizer.encode(ko).ids
            ko_ids = ko_ids[:self.max_seq_len-2]
            ko_ids.insert(0, self.sos_idx)
            ko_ids.append(self.eos_idx)
            while len(ko_ids) < self.max_seq_len:
                ko_ids.append(self.pad_idx)

            # En text tokenizing 후 sos, eos 토큰 추가
            en_ids = self.tokenizer.encode(en).ids
            en_ids = en_ids[:self.max_seq_len-2]
            en_ids.insert(0, self.sos_idx)
            en_ids.append(self.eos_idx)
            while len(en_ids) < self.max_seq_len:
                en_ids.append(self.pad_idx)

            ids_list.append((ko_ids, en_ids))
        return ids_list


    def build_vocab(self, cache_dir='./dataset'):
        cache_dir = os.path.join(cache_dir, self.dataset_name)
        os.makedirs(cache_dir, exist_ok=True)

        train_filename = os.path.join(cache_dir, 'train_vocab.pkl')
        val_filename = os.path.join(cache_dir, 'val_vocab.pkl')
        test_filename = os.path.join(cache_dir, 'test_vocab.pkl')

        if os.path.exists(train_filename):
            vocab_train = load_data(train_filename)
        else:
            vocab_train = zip(self.ds_train['ko'], self.ds_train['en'])
            vocab_train = self.encode_and_transform(vocab_train)
            save_data(vocab_train, train_filename)

        if os.path.exists(val_filename):
            vocab_val = load_data(val_filename)
        else:
            vocab_val = zip(self.ds_val['ko'], self.ds_val['en'])
            vocab_val = self.encode_and_transform(vocab_val)
            save_data(vocab_val, val_filename)

        if os.path.exists(test_filename):
            vocab_test = load_data(test_filename)
        else:
            vocab_test = zip(self.ds_test['ko'], self.ds_test['en'])
            vocab_test = self.encode_and_transform(vocab_test)
            save_data(vocab_test, test_filename)

        self.train = vocab_train
        self.val = vocab_val
        self.test = vocab_test


    def collate_fn(self, pairs):
        ko = torch.Tensor([pair[0] for pair in pairs])
        en = torch.Tensor([pair[1] for pair in pairs])
        return (ko, en)


    def get_loaders(self, batch_size=64):
        train_loader = DataLoader(self.train ,collate_fn=self.collate_fn, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val ,collate_fn=self.collate_fn, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test ,collate_fn=self.collate_fn, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader