import os

import csv
from tqdm import tqdm

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from models.PointerGeneratorTransformer import PointerGeneratorTransformer

from utils import *

class Trainer(object):
    def __init__(self, args, rank=1):
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
        self.vocab_size = len(self.vocab)

        self.train_data = self.load_data(args.train_file, 'train.pt', is_test=False)
        self.dev_data = self.load_data(args.dev_file, 'dev.pt', is_test=False)
        self.test_data = self.load_data(args.test_file, 'test.pt', is_test=True)

        self.model = PointerGeneratorTransformer(
                        src_vocab_size=self.vocab_size, tgt_vocab_size=self.vocab_size,
                        max_len=self.max_len  
                    )
        
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
                tgt_data = []
                
            with open(os.path.join(self.dataset_dir, file_name), 'r', encoding='utf-8') as f:
                r = csv.reader(f, delimiter='\t')
                for line in tqdm(r):
                    src = line[0]
                    src_data.append(src)
                    if not is_test:
                        tgt = line[1]
                        tgt_data.append(tgt)
            encoded_dict = self.tokenizer.batch_encode_plus(src_data, add_special_tokens=True, max_length=self.max_len, padding='max_length',
                                                    return_attention_mask=True, truncation=True, return_tensors='pt')
            src_input_ids = encoded_dict['input_ids']
            src_token_type_ids = encoded_dict['token_type_ids']
            src_attention_masks = encoded_dict['attention_mask']

            if not is_test:
                encoded_dict = self.tokenizer.batch_encode_plus(tgt_data, add_special_tokens=True, max_length=self.max_len, padding='max_length',
                                                        return_attention_mask=True, truncation=True, return_tensors='pt')
                tgt_input_ids = encoded_dict['input_ids']
                tgt_token_type_ids = encoded_dict['token_type_ids']
                tgt_attention_masks = encoded_dict['attention_mask']
                data = {
                    'src_input_ids': src_input_ids, 'src_token_type_ids': src_token_type_ids, 'src_attention_masks': src_attention_masks,
                    'tgt_input_ids': tgt_input_ids, 'tgt_token_type_ids': tgt_token_type_ids, 'tgt_attention_masks': tgt_attention_masks
                }
            else:
                data = {
                    'src_input_ids': src_input_ids, 'src_token_type_ids': src_token_type_ids, 'src_attention_masks': src_attention_masks,
                }
        torch.save(data, loader_file)
        return data

    # convert list of token ids to list of strings
    def decode(self, ids):
        strings = self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return strings

    # create dataset loader
    def make_dataloader(self, rank, data_dict, batch_size, shuffle=True):
        if "tgt_input_ids" in data_dict:
            dataset = TensorDataset(data_dict["src_input_ids"], data_dict["src_token_type_ids"], data_dict["src_attention_masks"],
                                    data_dict["tgt_input_ids"], data_dict["tgt_token_type_ids"], data_dict["tgt_attention_masks"])
        else:
            dataset = TensorDataset(data_dict["src_input_ids"], data_dict["src_token_type_ids"], data_dict["src_attention_masks"])
        if shuffle:
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank)
            dataset_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False)
        else:
            dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataset_loader

    def get_loss(self, predict, target):
        """
        Compute loss
        :param predict: S x N x vocab_size
        :param target: S x N
        :return: loss
        """
        predict = predict.contiguous().view(-1, self.vocab_size)
        target = target.contiguous().view(-1, 1)
        non_pad_mask = target.ne(self.vocab['[PAD]'])
        nll_loss = -predict.gather(dim=-1, index=target)[non_pad_mask].mean()
        smooth_loss = -predict.sum(dim=-1, keepdim=True)[non_pad_mask].mean()
        smooth_loss = smooth_loss / self.vocab_size
        loss = (1. - self.label_smooth) * nll_loss + self.label_smooth * smooth_loss
        return loss

    def train(self):
        train_loader = self.make_dataloader(0, self.train_data, self.train_batch_size)
        dev_loader = self.make_dataloader(0, self.dev_data, self.eval_batch_size)

        model = self.model.to(self.rank)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        total_steps = len(train_loader) * self.epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

        is_best = False
        curr_valid_loss = 0
        best_valid_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            print(f'Epoch / Total epochs: {epoch + 1} / {self.epochs}')
            running_loss = 0.0
            model.train()
            for batch in tqdm(train_loader):
                src_input_ids, src_token_type_ids, src_input_masks = batch[0].to(self.rank), batch[1].to(self.rank), batch[2].to(self.rank)
                tgt_input_ids, tgt_token_type_ids, tgt_input_masks = batch[3].to(self.rank), batch[4].to(self.rank), batch[5].to(self.rank)
                
                outputs = model(src_input_ids, src_token_type_ids, src_input_masks, tgt_input_ids, tgt_token_type_ids, tgt_input_masks)
                
                loss = self.get_loss(outputs.transpose(0, 1), tgt_input_ids.transpose(0, 1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                running_loss += loss.item()

            # print statistics
            self.logger.info(f"Train Epoch: {epoch + 1}, avg loss: {running_loss / len(train_loader):.4f}")
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
                break

    def validation(self, model, epoch, dev_loader):
        """ Computes loss and accuracy over the validation set, using teacher forcing inputs """
        model = model.to(self.rank)
        model.eval()

        running_loss = 0
        correct_preds = 0

        total_num = 0
        with torch.no_grad():
            for i, batch in enumerate(dev_loader):
                src_input_ids, src_token_type_ids, src_input_masks = batch[0].to(self.rank), batch[1].to(self.rank), batch[2].to(self.rank)
                tgt_input_ids, tgt_token_type_ids, tgt_input_masks = batch[3].to(self.rank), batch[4].to(self.rank), batch[5].to(self.rank)
                
                # Compute output of model
                output = model(src_input_ids, src_token_type_ids, src_input_masks, tgt_input_ids, tgt_token_type_ids, tgt_input_masks)
                
                # Get model predictions
                predictions = output.topk(1)[1].squeeze()

                # Compute accuracy
                predictions = predictions * tgt_input_masks

                correct_preds += torch.all(torch.eq(predictions, tgt_input_ids), dim=-1).sum()
                
                # Compute loss
                loss = self.get_loss(output.transpose(0, 1), tgt_input_ids.transpose(0, 1))
                # -------------
                running_loss += loss.item()
                total_num += len(batch)
        # print statistics
        final_loss = running_loss / (i + 1)

        accuracy = float(100 * correct_preds) / total_num
        self.logger.info(f"Validation. Epoch: {epoch + 1}, avg dev loss: {final_loss:.4f}, accuracy: {accuracy:.2f}%")
        # return accuracy
        return final_loss
    
    def test(self, out_max_len=60):
        test_loader = self.make_dataloader(0, self.test_data, 1, shuffle=False)

        model = torch.load('./model_dict/model.pt').to(self.rank)
        model.eval()

        with torch.no_grad():
            for batch in tqdm(test_loader):
                src_input_ids, src_token_type_ids, src_input_masks = batch[0].to(self.rank), batch[1].to(self.rank), batch[2].to(self.rank)
                
                memory = model.encode(src_input_ids, src_token_type_ids, src_input_masks).transpose(0, 1)
                tgt_input_ids = torch.zeros(src_input_ids.shape[0], self.max_len, dtype=torch.long, device=self.rank)
                tgt_input_ids[:, 0] = self.vocab['[CLS]']   # bert sentence head
                for j in range(1, self.max_len):
                    tgt_input_masks = torch.zeros(src_input_ids.shape[0], self.max_len, dtype=torch.long, device=self.rank)
                    tgt_input_masks[:, :j] = 1

                    output = model(src_input_ids, src_token_type_ids, src_input_masks, tgt_input_ids[:, :j], None, tgt_input_masks[:, :j])
                    _, ids = output.topk(1)
                    ids = ids.squeeze(-1)

                    tgt_input_ids[:, j] = ids[:, -1]
                    
                    # print(tgt_input_ids[:, j])
                    # break
                string = self.decode(tgt_input_ids)
                print(string)
                break
                