import argparse
from Trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dataset_dir', default='./data',
                        help='directory for dataset')
    parser.add_argument('--train_file', default='train.csv',
                        help='name of train file')
    parser.add_argument('--dev_file', default='dev.csv',
                        help='name of dev file')
    parser.add_argument('--test_file', default='test.csv',
                        help='name of test file')    
    parser.add_argument('--vocab_path', default='./model_dict/vocab.txt',
                        help='vocab path for pre-trained model')
    parser.add_argument('--max_len', type=int, default=200, 
                        help='vocab path for pre-trained model')
    parser.add_argument('--train_batch_size', type=int, default=16, 
                        help='batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=2, 
                        help='batch size for evaluating')
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of epochs for training')
    parser.add_argument('--label_smooth', default=0.1, type=float,
                        help='label smoothing coeff')
    parser.add_argument('--gpus', default=1, type=int,
                        help='number of gpus to use')
    args = parser.parse_args()

    trainer = Trainer(args, rank=1)

    trainer.train()
    
    # trainer.test()
    # trainer.generate(out_max_length=60, top_k=5, top_p=0.95, max_length=200)
    # trainer.eval()

if __name__ == "__main__":
    main()