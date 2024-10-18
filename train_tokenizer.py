from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
import re
import random
import os
from tokenizer_config import *


def load_koen_dataset(url):
    ds = load_dataset(url)
    sent_list = []
    for ko_sent, en_sent in tqdm(zip(ds['train']['ko'], ds['train']['en'])):
        ko_sent = re.sub('\n+', ' ', ko_sent, flags=re.I)
        en_sent = re.sub('\n+', ' ', en_sent, flags=re.I)
        sent_list.extend([ko_sent.strip(), en_sent.strip()])

    return sent_list

def save_koen_dataset(dataset, cache_dir, dataset_name):
    file_path = os.path.join(cache_dir, dataset_name)
    
    os.makedirs(cache_dir, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in dataset:
            file.write(f'{item}\n')
            
            

if __name__ == '__main__':
    filepath = os.path.join(CACHE_DIR, DATASET_NAME)

    print('Loading Dataset..')
    dataset = load_koen_dataset(DATASET_URL)
    random.seed(SEED)
    random.shuffle(dataset)
    save_koen_dataset(dataset, CACHE_DIR, DATASET_NAME)

    print('Training Toeknizer..')
    tokenizer = ByteLevelBPETokenizer(unicode_normalizer="nfkc")
    tokenizer.train(files=filepath, vocab_size=VOCAB_SIZE, min_frequency=MIN_FREQUENCY,
                    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>",])

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save_model(TOKENIZER_DIR)