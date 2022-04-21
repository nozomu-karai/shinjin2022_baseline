import argparse
import logging
import os
import random
import sys
from tqdm import tqdm
import numpy as np


from models import PosNegBERT
from dataset import PosNegDataset

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import BCELoss
from sklearn.metrics import f1_score



CLS = "[CLS]"
SEP = "[SEP]"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def setseed(seed):
    random.seed(seed)
    np.random.seed(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    n_gpu = torch.cuda.device_count()
    setseed(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model, do_lower_case=args.do_lower_case)

    # train set
    if args.do_train:
        train_dataset = PosNegDataset(args.train_data, tokenizer, args.max_seq_length)
        dev_dataset = PosNegDataset(args.valid_data, tokenizer, args.max_seq_length)

        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.train_batch_size, shuffle=False)

    # test set
    if args.do_eval:
        test_dataset = PosNegDataset(args.test_data, tokenizer, args.max_seq_length)

        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)


    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(sys.argv)
    logger.info(args)
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))

    model = PosNegBERT(args.pretrained_model)
    if args.saved_dir is not None:
        model.load_state_dict(torch.load(os.path.join(args.saved_dir, "pytorch_model.bin")), strict=False)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs

        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", len(train_dataloader))

        best_result = None
        
        lr = args.learning_rate

        model_params = list(model.named_parameters())
        params = [p for n, p in model_params]
        optimizer = AdamW(params, lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_optimization_steps * args.warmup_proportion), num_train_optimization_steps)

        for epoch in range(int(args.num_train_epochs)):
            model.train()
            logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
            train_bar = tqdm(train_dataloader)

            bce_loss = BCELoss()
            tr_loss = 0
            n_que, n_cor = 0, 0
            preds, labels = [], []
            for step, batch in enumerate(train_bar):
                batch = {key: value.to(device) for key, value in batch.items()}
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                token_type_ids = batch["token_type_ids"]
                domain_label = batch["label"]
                output_dict = model(input_ids, attention_mask, token_type_ids)
                logits = torch.squeeze(output_dict['logits'], dim=1)

                loss = bce_loss(logits, domain_label.float())

                if n_gpu > 1:
                    loss = loss.mean()
                
                loss.backward()

                batch_size = len(input_ids)
                tr_loss += loss.item() * batch_size

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                logits = logits.to('cpu').detach().numpy()
                domain_label = batch["label"].to('cpu').detach().numpy()
                _preds = np.where(logits > 0.5, 1, 0)
                preds.extend(_preds)
                labels.extend(domain_label)
                n_que += len(_preds)
                n_cor += sum(_preds == domain_label)

                train_bar.set_description(f"epoch : {epoch+1} train_loss : {round(tr_loss / (step + 1), 3):.4f} train_acc : {n_cor / n_que:.4f}")
            
            total_loss = 0
            n_que, n_cor = 0, 0
            preds, labels = [], []
            with torch.no_grad():
                dev_bar = tqdm(dev_dataloader)
                for step, batch in enumerate(dev_bar):
                    batch = {key: value.to(device) for key, value in batch.items()}
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    token_type_ids = batch["token_type_ids"]
                    domain_label = batch["label"]
                    output_dict = model(input_ids, attention_mask, token_type_ids)
                    logits = output_dict['logits']
                    logits = torch.squeeze(logits, dim=1)

                    loss = bce_loss(logits, domain_label.float())

                    if n_gpu > 1:
                        loss = loss.mean()
                    
                    batch_size = len(input_ids)
                    total_loss += loss.item() * batch_size

                    logits = logits.to('cpu').detach().numpy()
                    domain_label = batch["label"].to('cpu').detach().numpy()
                    _preds = np.where(logits > 0.5, 1, 0)
                    preds.extend(_preds)
                    labels.extend(domain_label)
                    n_que += len(_preds)
                    n_cor += sum(_preds == domain_label)

                    dev_bar.set_description(f"val_loss : {round(total_loss / (step + 1), 3):.4f} val_acc : {n_cor / n_que:.4f}")

            score = f1_score(labels, preds, average='macro')
            logger.info(f"validation accuracy: {n_cor / n_que:.4f}")
            logger.info(f"validation f1 score: {score:.4f}")

            if (best_result is None) or score > best_result:
                best_result = score
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, 'pytorch_model.bin'))

    if args.do_eval:
        bce_loss = BCELoss()
        total_loss = 0
        n_que, n_cor = 0, 0
        preds, labels = [], []

        with torch.no_grad():
            test_bar = tqdm(test_dataloader)
            for step, batch in enumerate(test_bar):
                batch = {key: value.to(device) for key, value in batch.items()}
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                token_type_ids = batch["token_type_ids"]
                domain_label = batch["label"]
                output_dict = model(input_ids, attention_mask, token_type_ids)
                logits = output_dict['logits']
                logits = torch.squeeze(logits, dim=1)

                loss = bce_loss(logits, domain_label.float())

                if n_gpu > 1:
                    loss = loss.mean()
                
                batch_size = len(input_ids)
                total_loss += loss.item() * batch_size

                logits = logits.to('cpu').detach().numpy()
                domain_label = batch["label"].to('cpu').detach().numpy()
                _preds = np.where(logits > 0.5, 1, 0)
                preds.extend(_preds)
                labels.extend(domain_label)
                n_que += len(_preds)
                n_cor += sum(_preds == domain_label)
           
                test_bar.set_description(f"test_loss : {round(total_loss / (step + 1), 3):.4f} test_acc : {n_cor / n_que:.4f}")

        score = f1_score(labels, preds, average='macro')
        logger.info(f"test accuracy: {n_cor / n_que:.4f}")
        logger.info(f"test f1 score: {score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default='/mnt/hinoki/ueda/shinjin2019/acp-2.0/train.txt')
    parser.add_argument("--valid_data", type=str, default='/mnt/hinoki/ueda/shinjin2019/acp-2.0/valid.txt')
    parser.add_argument("--test_data", type=str, default='/mnt/hinoki/ueda/shinjin2019/acp-2.0/test.txt')
    parser.add_argument("--pretrained_model", default="/home/karai/pretrained_model/NICT_BERT-base_JapaneseWikipedia_32K_BPE",
                        help="pretrained model path")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--saved_dir", default=None, type=str, 
                        help="The saved directory where the model predictions and checkpoints are written.")
    parser.add_argument("--max_seq_length", default=228, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=256, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument('--device', type=str, default='0,1,2,3,4,5,6,7',
                         help="number of GPU to use")

    args = parser.parse_args()
    main(args)
