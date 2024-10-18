import os, sys, time
import logging

import torch
import torch.nn.functional as F
from torch import nn, optim
from source.utils import remove_surplus, set_seed
from source.dataset import KoEnTranslation
from source.model import Transformer, create_lookahead_mask, create_padding_mask
from source.model import Encoder, Decoder, EncoderLayer, DecoderLayer, Embedding
from tqdm import tqdm
from configs import *

logging.basicConfig(level=logging.INFO)

def train(model, data_loader, optimizer, criterion, epoch, checkpoint_dir, save):
    model.train()
    epoch_loss = 0

    n_steps = 0
    for idx, (X_ko, X_en) in tqdm(enumerate(data_loader)):
        X_ko = X_ko.to(torch.int).to(model.device) # (batch, seq_len)
        X_en = X_en.to(torch.int).to(model.device) # (batch, seq_len)
        X_ko = remove_surplus(X_ko)
        X_en = remove_surplus(X_en)
        Y_en = X_en[:, 1:]
        X_en = X_en[:, :-1]

        optimizer.zero_grad()
        output = model(X_ko, X_en) # (batch, seq_len, vocab_size)
        output = output.reshape(-1, output.shape[-1])
        Y_en = Y_en.reshape(-1).type(torch.long)

        loss = criterion(output, Y_en)
        epoch_loss += loss.item()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n_steps += 1

    epoch_loss /= n_steps

    if save:
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, f'{epoch:04d}.pt')
        torch.save({
            'epoch': epoch,
            'loss': epoch_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)

    return epoch_loss

def translate_from_model(model, sentence, tkn):
    model.eval()
    sentence = tkn.encode('<s>' + sentence + '</s>').ids
    max_length = len(sentence) + 50

    # Encode
    sentence = torch.Tensor(sentence).unsqueeze(0).type(torch.int).to(model.device)
    sentence_mask = create_padding_mask(sentence, sentence).to(model.device)
    enc_out = model.encode(sentence, sentence_mask)

    # Greedy-Decoding
    sos_token_idx = tkn.encode('<s>').ids[0]
    eos_token_idx = tkn.encode('</s>').ids[0]
    generations = torch.ones(1,1).fill_(sos_token_idx).type(torch.int).to(model.device)

    for i in range(max_length-1):
        dec_mask = create_padding_mask(generations, sentence)
        look_ahead_mask = create_lookahead_mask(generations, generations) & create_padding_mask(generations, generations)
        output = model.decode(generations, enc_out, dec_mask, look_ahead_mask) # (1, seq_len, vocab_size)
        pred = model.classifier(output[:, -1]) # 마지막 word의 logit, (1, vocab_size)
        next_word = torch.argmax(pred, dim=-1).item() # 마지막 word, (1, 1)
        generations = torch.concat([generations, torch.ones(1,1).fill_(next_word).type(torch.int).to(model.device)], dim=1) # (1, n+1)
        
        if next_word == eos_token_idx:
            break

    translation = tkn.decode(generations[0].detach().cpu().numpy())

    return translation

def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0
    n_steps = 0
    with torch.no_grad():
        for idx, (X_ko, X_en) in tqdm(enumerate(data_loader)):
            X_ko = X_ko.to(torch.int).to(model.device) # (batch, seq_len)
            X_en = X_en.to(torch.int).to(model.device) # (batch, seq_len)
            X_ko = remove_surplus(X_ko)
            X_en = remove_surplus(X_en)
            Y_en = X_en[:, 1:]
            X_en = X_en[:, :-1]

            output = model(X_ko, X_en) # (batch, seq_len, vocab_size)
            output = output.reshape(-1, output.shape[-1])
            Y_en = Y_en.reshape(-1).type(torch.long)
            loss = criterion(output, Y_en)

            epoch_loss += loss.item()
            n_steps += 1

        epoch_loss /= n_steps
        return epoch_loss


if __name__ == '__main__':
    set_seed(SEED)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    enc_embedder = Embedding(VOCAB_SIZE, D_MODEL)
    dec_embedder = Embedding(VOCAB_SIZE, D_MODEL)

    encoder_layer = EncoderLayer(D_MODEL, NUM_HEAD, D_FF, DROPOUT)
    encoder = Encoder(encoder_layer, N_LAYERS)

    decoder_layer = DecoderLayer(D_MODEL, NUM_HEAD, D_FF, DROPOUT)
    decoder = Decoder(decoder_layer, N_LAYERS)

    classifier = nn.Linear(D_MODEL, VOCAB_SIZE)

    model = Transformer(enc_embedder, dec_embedder, encoder, decoder, classifier)
    model.to(device)
    model.device = device
    
    logging.info('Model is Loaded')

    def initialize_weights(model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.kaiming_uniform_(model.weight)

    model.apply(initialize_weights)
    logging.info('He initialization is applied')

    
    koen_dataset = KoEnTranslation('./tokenizer/vocab.json', './tokenizer/merges.txt', MAX_SEQ_LEN)
    tokenizer = koen_dataset.tokenizer
    train_loader, val_loader, test_loader = koen_dataset.get_loaders(batch_size=BATCH_SIZE)
    logging.info('Datasets are Loaded')


    optimizer = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
    criterion = nn.CrossEntropyLoss(ignore_index=koen_dataset.pad_idx)
    logging.info('Optimizer and Scheduler are loaded')
    
    logging.info('Training Start')
    for epoch in range(EPOCHS):
        logging.info(f'Epoch: {epoch}')
        save = True if epoch % 2 == 0 else False
        train_loss = train(model, train_loader,optimizer, criterion, epoch, './checkpoint', save)
        logging.info(f'Train loss: {train_loss}')

        if (epoch % EVAL_EPOCH == 0):
            val_loss = evaluate(model, val_loader, criterion)
            logging.info(f'Validation loss: {val_loss}')
            if epoch > WARM_UP_STEP:
                scheduler.step(val_loss)

        ex_input = '한국원자력연구원은 국제원자력기구(IAEA)와 함께 대전 유성구 연구원에서 중성자 영상을 활용한 연구·산업적 적용 워크숍을 한다고 23일 밝혔다.'
        ex_output = 'The Korea Atomic Energy Research Institute announced on the 23rd that it will hold a research and industrial application workshop using neutron imaging at the Yuseong-gu Research Center in Daejeon together with the International Atomic Energy Agency (IAEA).'
        logging.info(f'입력 한국어: {ex_input}')
        ko_to_en = translate_from_model(model, ex_input, tokenizer)
        logging.info(f'모델 번역: {ko_to_en}')
        logging.info(f'정답: {ex_output}')

    test_loss = evaluate(model, test_loader, criterion)
    logging.info(f'Test loss: {test_loss}')