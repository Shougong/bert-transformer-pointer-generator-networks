import os
import re
import csv
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from models.PointerGeneratorTransformer import PointerGeneratorTransformer

from utils import *
from preprocess import *
from models.model_utils import padding_trg

class Trainer(object):
    def __init__(self, args, rank=0):
        super(Trainer, self).__init__()
        self.dataset_dir = args.dataset_dir
        self.max_len = args.max_len
        self.world_size = args.gpus
        self.rank = rank
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.epochs = args.epochs
        self.label_smooth = args.label_smooth

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.vocab = self.tokenizer.vocab
        self.inv_vocab = {v:k for k,v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        self.pad_id = self.vocab['[PAD]']
        self.cls_id = self.vocab['[CLS]']
        self.sep_id = self.vocab['[SEP]']
        self.unk_id = self.vocab['[UNK]']

        self.train_data = self.load_data(args.train_file, 'train.pt', is_test=False)
        self.dev_data = self.load_data(args.dev_file, 'dev.pt', is_test=False)
        
        self.test_data = self.load_test_data(args.test_file, 'test.pt')
        self.model = PointerGeneratorTransformer(
                        rank=self.rank, src_vocab_size=self.vocab_size, 
                        tgt_vocab_size=self.vocab_size, inv_vocab=self.inv_vocab,
                        pad_id=self.pad_id, max_len=self.max_len  
                    )

        # initialize model parameters
        self.init_parameters()
        
        self.logger = get_logger()
    
    def load_data(self, file_name, loader_name, is_test=False):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f'Loading data from {loader_file}')
            data = torch.load(loader_file)
        else:
            print(f'Construct data from {os.path.join(self.dataset_dir, file_name)}')
            src_data = []
            if not is_test:
                trg_data = []
                
            with open(os.path.join(self.dataset_dir, file_name), 'r', encoding='utf-8') as f:
                r = csv.reader(f, delimiter='\t')
                
                for line in tqdm(r):
                    src = line[0].strip()
                    # add preprocess progress
                    src = toSimpleChinese(src)
                    src_data.append(src)
                    if not is_test:
                        trg = line[1].strip()
                        trg = toSimpleChinese(trg)
                        trg_data.append(trg)
            
            encoded_dict = self.tokenizer.batch_encode_plus(src_data, add_special_tokens=True, max_length=self.max_len, padding='max_length',
                                                    return_attention_mask=True, truncation=True, return_tensors='pt')
            src_input_ids = encoded_dict['input_ids']
            src_attention_masks = encoded_dict['attention_mask']

            if not is_test:
                trg_input_ids, trg_ground_ids, trg_attention_masks = [], [], []
                for text in trg_data:
                    # encode text without trunction
                    encoded_text = self.tokenizer(text)
                    trg_ids, trg_attention_mask = encoded_text['input_ids'], encoded_text['attention_mask']
                    if len(trg_ids) > self.max_len:
                        trg_ids = trg_ids[:self.max_len-1] + [self.sep_id]
                        trg_attention_mask = trg_attention_mask[:self.max_len]

                    # add padding
                    trg_input, trg_ground, trg_mask = padding_trg(trg_ids[:-1], trg_ids[1:], trg_attention_mask[:-1], self.max_len)

                    trg_input_ids.append(trg_input)
                    trg_ground_ids.append(trg_ground)
                    trg_attention_masks.append(trg_mask)

                data = {
                    'src_input_ids': src_input_ids, 'src_attention_masks': src_attention_masks,
                    'trg_input_ids': torch.tensor(trg_input_ids), 'trg_ground_ids': torch.tensor(trg_ground_ids),
                    'trg_attention_masks': torch.tensor(trg_attention_masks)
                }
            else:
                data = {
                    'src_input_ids': src_input_ids, 'src_attention_masks': src_attention_masks
                }
        torch.save(data, loader_file)
        return data

    def load_test_data(self, file_name, loader_name):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f'Loading data from {loader_file}')
            data = torch.load(loader_file)
        else:
            print(f'Construct data from {os.path.join(self.dataset_dir, file_name)}')
            src_data = []
            with open(os.path.join(self.dataset_dir, file_name), 'r', encoding='utf-8') as f:
                data = f.readlines()
                for line in tqdm(data):
                    # add preprocess progress
                    src = toSimpleChinese(line.strip())
                    src_data.append(src)
            
            encoded_dict = self.tokenizer.batch_encode_plus(src_data, add_special_tokens=True, max_length=self.max_len, padding='max_length',
                                                    return_attention_mask=True, truncation=True, return_tensors='pt')
            src_input_ids = encoded_dict['input_ids']
            src_attention_masks = encoded_dict['attention_mask']
            data = {
                'src_input_ids': src_input_ids, 'src_attention_masks': src_attention_masks
            }
            torch.save(data, loader_file)
        return data

    # convert list of token ids to list of strings
    def decode(self, ids):
        strings = self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return strings

    # create dataset loader
    def make_dataloader(self, rank, data_dict, batch_size, shuffle=True):
        if "trg_input_ids" in data_dict:
            dataset = TensorDataset(data_dict["src_input_ids"], data_dict["src_attention_masks"],
                                    data_dict["trg_input_ids"], data_dict["trg_ground_ids"], data_dict["trg_attention_masks"])
        else:
            dataset = TensorDataset(data_dict["src_input_ids"], data_dict["src_attention_masks"])
        if shuffle:
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank)
            dataset_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False)
        else:
            dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataset_loader

    def cal_performance(self, logits, ground, smoothing=True):
        logits = logits.contiguous().view(-1, logits.size(-1))
        ground = ground.contiguous().view(-1)

        loss = self.cal_loss(logits, ground, smoothing=smoothing)

        pad_mask = ground.ne(self.pad_id)
        pred = logits.max(-1)[1]
        correct = pred.eq(ground)
        correct = correct.masked_select(pad_mask).sum().item()
        total_words = pad_mask.sum().item()
        return loss, correct, total_words

    def cal_loss(self, logits, ground, smoothing=True):
        def label_smoothing(logits, labels):
            eps = 0.1
            num_classes = logits.size(-1)

            # >>> z = torch.zeros(2, 4).scatter_(1, torch.tensor([[2], [3]]), 1.23)
            # >>> z
            # tensor([[ 0.0000,  0.0000,  1.2300,  0.0000],
            #        [ 0.0000,  0.0000,  0.0000,  1.2300]])
            one_hot = torch.zeros_like(logits).scatter(1, labels.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
            log_prb = F.log_softmax(logits, dim=1)
            non_pad_mask = ground.ne(self.pad_id)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).mean()
            return loss
        if smoothing:
            loss = label_smoothing(logits, ground)
        else:
            loss = F.cross_entropy(logits, ground, ignore_index=self.pad_id)
        
        return loss

    def init_parameters(self):
        for name, param in self.model.named_parameters():
            if 'encoder' not in name and 'tgt_embed' not in name and param.dim() > 1:
                xavier_uniform_(param)

    def train(self):
        train_loader = self.make_dataloader(0, self.train_data, self.train_batch_size)
        dev_loader = self.make_dataloader(0, self.dev_data, self.eval_batch_size)

        model = self.model.to(self.rank)

        total_steps = len(train_loader) * self.epochs

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': model.encoder.parameters(), 'lr': 1e-6, 'weight_decay': 0.01},
            {'params': model.tgt_embed.parameters(), 'lr': 1e-6, 'weight_decay': 0.01},
            {'params': model.decoder.parameters(), 'weight_decay': 0.01},
            {'params': model.p_vocab.parameters(), 'weight_decay': 0.01},
            {'params': model.p_gen.parameters(), 'weight_decay': 0.01}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-4, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=4000, num_training_steps=total_steps)

        is_best = False
        curr_valid_loss = 0
        best_valid_loss = float("inf")
        epochs_no_improve = 0

        total_steps = 0
        for epoch in range(self.epochs):
            print(f'Epoch / Total epochs: {epoch + 1} / {self.epochs}')
            running_loss = 0.0
            model.train()
            correct_words = 0
            total_words = 0
            for batch in tqdm(train_loader):
                src_input_ids, src_input_masks = batch[0].to(self.rank), batch[1].to(self.rank)
                trg_input_ids, trg_ground_ids, trg_input_masks = batch[2].to(self.rank), batch[3].to(self.rank), batch[4].to(self.rank)
                
                outputs = model(src_input_ids, src_input_masks, trg_input_ids, trg_input_masks)
                
                loss, n_correct, n_word = self.cal_performance(outputs, trg_ground_ids, smoothing=True)
              
                loss.backward()
                
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                running_loss += loss.item()

                correct_words += n_correct
                total_words += n_word
                total_steps += 1

            # print statistics
            if total_steps % 100 == 0 or epoch % 1 == 0:
                self.logger.info(f"Train Epoch: {epoch + 1}, avg loss: {running_loss / len(train_loader):.4f}, accuracy: {100 * correct_words / total_words:.2f}%")
            
            if epoch % 1 == 0:
                epochs_no_improve += 1
                curr_valid_loss = self.validation(model, epoch, dev_loader)
                # If best accuracy so far, save model as best and the accuracy
                if curr_valid_loss < best_valid_loss:
                    self.logger.info("New best loss, Model saved")
                    is_best = True
                    best_valid_loss = curr_valid_loss
                    best_valid_epoch = epoch
                    epochs_no_improve = 0
                    torch.save(model, './model_dict/model.pt')
            if epochs_no_improve > 3:
                self.logger.info("No best dev loss, stop training.")
                break

    def validation(self, model, epoch, dev_loader):
        """ Computes loss and accuracy over the validation set, using teacher forcing inputs """
        model = model.to(self.rank)
        model.eval()

        running_loss = 0
        correct_words = 0
        total_words = 0
        total_num = 0
        with torch.no_grad():
            for i, batch in enumerate(dev_loader):
                src_input_ids, src_input_masks = batch[0].to(self.rank), batch[1].to(self.rank)
                trg_input_ids, trg_ground_ids, trg_input_masks = batch[2].to(self.rank), batch[3].to(self.rank), batch[4].to(self.rank)
                
                # Compute output of model
                output = model(src_input_ids, src_input_masks, trg_input_ids, trg_input_masks)

                # Get model predictions
                predictions = output.topk(1)[1].squeeze()

                # Compute loss
                loss, n_correct, n_word = self.cal_performance(output, trg_ground_ids, smoothing=True)
                correct_words += n_correct
                total_words += n_word
                # -------------
                running_loss += loss.item()
                total_num += len(batch[0])
        # print statistics
        final_loss = running_loss / (i + 1)

        accuracy = float(100.0 * correct_words) / total_words
        self.logger.info(f"Validation. Epoch: {epoch + 1}, avg dev loss: {final_loss:.4f}, accuracy: {accuracy:.2f}%")

        # return accuracy
        return final_loss

    
    def test(self, out_max_len=60):
        test_loader = self.make_dataloader(0, self.test_data, 1, shuffle=False)

        model = torch.load('./model_dict/model.pt').to(self.rank)
        f = open(os.path.join(self.dataset_dir, 'results.csv'), 'a+', encoding='utf-8')

        idx = 20000
        for batch in tqdm(test_loader):
            src_input_ids, src_input_masks = batch[0].to(self.rank), batch[1].to(self.rank)

            memory = model.encode(src_input_ids, src_input_masks).transpose(0, 1)
            tgt_input_ids = torch.zeros(src_input_ids.shape[0], self.max_len, dtype=torch.long, device=self.rank)
            tgt_input_ids[:, 0] = self.cls_id   # bert sentence head
            for j in range(1, out_max_len):
                tgt_input_masks = torch.zeros(src_input_ids.shape[0], self.max_len, dtype=torch.long, device=self.rank)
                tgt_input_masks[:, :j] = 1

                src_attention_masks = ((1 - src_input_masks) > 0)
                tgt_attention_masks = ((1 - tgt_input_masks) > 0)

                output = model.decode(memory, tgt_input_ids[:, :j], src_input_ids, tgt_attention_masks[:, :j], src_attention_masks)
                _, ids = output.topk(1)
                ids = ids.squeeze(-1)

                tgt_input_ids[:, j] = ids[:, -1]

                if ids[:, -1] == self.sep_id:
                    break
            string = self.decode(tgt_input_ids)[0]
            if len(string) == 0:
                string = self.decode(src_input_ids)[0]
            string = re.sub(r"\s{1,}", "", string)
            f.write(str(idx) + '\t' + string + '\n')
            idx += 1
        f.close()
            

    def greedy_decode(self, model, src_seq, src_mask, out_max_len=60):
        model.eval()

        with torch.no_grad():
            memory = model.encode(src_seq, src_mask).transpose(0, 1)
            dec_seq = torch.full((src_seq.size(0), ), self.cls_id).unsqueeze(-1).type_as(src_seq)
            
            src_attention_masks = ((1 - src_mask) > 0)
            for i in range(out_max_len):
                dec_output = model.decode(memory, dec_seq, src_seq, None, src_attention_masks)
                dec_output = dec_output.max(-1)[1]
                dec_seq = torch.cat((dec_seq, dec_output[:, -1].unsqueeze(-1)), 1)
        return dec_seq

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits
    
    def generate(self, out_max_length=40, top_k=30, top_p=0.0, max_length=200):
        test_loader = self.make_dataloader(0, self.test_data, 1, shuffle=False)

        model = torch.load('./model_dict/model.pt').to(self.rank)
        f = open(os.path.join(self.dataset_dir, 'results.csv'), 'a+', encoding='utf-8')

        idx = 20000
        with torch.no_grad(): 
            for batch in tqdm(test_loader):
                src_input_ids, src_input_masks = batch[0].to(self.rank), batch[1].to(self.rank)

                memory = model.encode(src_input_ids, src_input_masks).transpose(0, 1)
                tgt_input_ids = torch.zeros(src_input_ids.shape[0], self.max_len, dtype=torch.long, device=self.rank)
                tgt_input_ids[:, 0] = self.cls_id   # bert sentence head
                output_ids = []
                for j in range(1, out_max_length):
                    tgt_input_masks = torch.zeros(src_input_ids.shape[0], self.max_len, dtype=torch.long, device=self.rank)
                    tgt_input_masks[:, :j] = 1

                    src_attention_masks = ((1 - src_input_masks) > 0)
                    tgt_attention_masks = ((1 - tgt_input_masks) > 0)

                    scores = model.decode(memory, tgt_input_ids[:, :j], src_input_ids, tgt_attention_masks[:, :j], src_attention_masks)
                    
                    logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
                    logit_score[self.unk_id] = -float('Inf')
                    
                    # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                    for id_ in set(output_ids):
                        logit_score[id_] /= 2.0                
                    
                    filtered_logits = self.top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    if self.sep_id == next_token.item():
                        break
                    tgt_input_ids[:, j] = next_token.item()
                    output_ids.append(next_token.item())
                string = self.tokenizer.decode(torch.tensor(output_ids))
                string = re.sub(r"\s{1,}", "", string)
                f.write(str(idx) + '\t' + string + '\n')
                idx += 1
            f.close()

    def eval(self):
        """ Computes loss and accuracy over the validation set, using teacher forcing inputs """
        model = torch.load('./model_dict/model.pt').to(self.rank)
        model.eval()

        data_loader = self.make_dataloader(0, self.train_data, 1, shuffle=False)

        f = open(os.path.join(self.dataset_dir, 'train_results.csv'), 'a+', encoding='utf-8')

        with torch.no_grad():
            for batch in tqdm(data_loader):
                src_input_ids, src_input_masks = batch[0].to(self.rank), batch[1].to(self.rank)
                trg_input_ids, trg_ground_ids, trg_input_masks = batch[2].to(self.rank), batch[3].to(self.rank), batch[4].to(self.rank)

                # Compute output of model
                output = model(src_input_ids, src_input_masks, trg_input_ids, trg_input_masks)

                # Get model predictions
                predictions = output.topk(1)[1].squeeze()
                predictions = predictions * trg_input_masks

                # Compute loss
                loss, n_correct, n_word = self.cal_performance(output, trg_ground_ids, smoothing=True)
                accuracy = float(100.0 * n_correct) / n_word

                if accuracy < 95.0:
                    src_string = self.decode(src_input_ids)[0]
                    src_string = re.sub(r"\s{1,}", "", src_string)
                    trg_string = self.decode(trg_input_ids)[0]
                    trg_string = re.sub(r"\s{1,}", "", trg_string)
                    pred_string = self.decode(predictions)[0]
                    pred_string = re.sub(r"\s{1,}", "", pred_string)
                    
                    f.write(src_string + '\t' + trg_string + '\t' + pred_string + '\n')
  
            f.close()