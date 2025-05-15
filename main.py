import random
import torch
import os
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from utils.utils import read_csv, split_dataset, param_num, log_print, GradualWarmupScheduler, read_csv_with_name
from utils.data import MyDataset
from model import Net
from train_validation import train, validate
import matplotlib.pyplot as plt
from utils.gen_bert_embedding import circRNABert

def fix_seed(seed):
    # If seed is None, generate a random seed between 1 and 10000
    if seed is None:
        seed = random.randint(1, 10000)

    torch.set_num_threads(1)

    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the PYTHONHASHSEED environment variable to ensure reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Set the seed for PyTorch's CPU random number generator
    torch.manual_seed(seed)

    # If using CUDA, also seed the GPU random number generator
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_embeddings(embeddings, structure, file_path):
    np.savez(file_path, bert_embeddings=embeddings, structure=structure)

def load_embeddings(file_path):
    data = np.load(file_path)
    return data['bert_embeddings'], data['structure']

def main(args):
   
    fix_seed(args.seed)  # fix seed for reproducibility

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(device)

    max_length = 125

    file_name = args.data_file
    data_path = args.data_path

    if args.train:
        embeddings_file = os.path.join(data_path, 'embeddings', file_name + '_train_embeddings.npz')
        names, sequences, structs, label = read_csv(os.path.join(data_path, file_name + "_train.tsv"))
        if os.path.exists(embeddings_file):
            bert_embedding, structure = load_embeddings(embeddings_file)
            print("OK")
        else:
            bert_model_path = args.BERT_model_path
            tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=False)
            model = BertModel.from_pretrained(bert_model_path)
            model = model.to(device)
            model = model.eval()
            bert_embedding = circRNABert(list(sequences), model, tokenizer, device, 3)  
            bert_embedding = bert_embedding.transpose([0, 2, 1])  

            structure = np.zeros((len(structs), 1, max_length))  
            for i in range(len(structs)):
                struct = structs[i].split(',')
                ti = [float(t) for t in struct]
                ti = np.array(ti).reshape(1, -1)
                structure[i] = np.concatenate([ti], axis=0)

            # 保存嵌入和结构数据
            save_embeddings(bert_embedding, structure, embeddings_file)
            
        hidden_size = bert_embedding.shape[1]
        
        [train_emb, train_struct, train_label], [val_emb, val_struct, val_label] = \
            split_dataset(bert_embedding, structure, label)  

        # 创建数据集对象，将数据集进行封装
        train_set = MyDataset(train_emb, train_struct, train_label)
        val_set = MyDataset(val_emb, val_struct, val_label)

        # 创建数据加载器对象，用于加载数据集
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32 * 8, shuffle=False)
 
        model = Net(hidden_size).to(device)
        # 定义损失函数
        criterion = nn.BCEWithLogitsLoss()

        # 定义优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)

        # 定义学习率调度器，用于在训练过程中动态调整学习率
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=8, total_epoch=float(200), after_scheduler=None)
        
        model_save_path = args.model_save_path

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        
        #表示在验证集上的性能在多少个训练周期内没有提升时，就停止训练
        early_stopping = args.early_stopping
        
        param_num(model)
        
        train_losses = []
        val_losses = []
        
        best_auc = 0 
        best_acc = 0
        best_epoch = 0
        best_aupr = 0
        best_loss = 100

        for epoch in range(1, 200):
            # 训练和验证
            train_metrics = train(model, device, train_loader, criterion, optimizer)
            val_metrics, _, _ = validate(model, device, val_loader, criterion)
            
            train_losses.append(train_metrics.other[0])
            val_losses.append(val_metrics.other[0])
            
            # 更新学习率
            scheduler.step()
            lr = scheduler.get_lr()[0]
            
            # 检查并更新最佳性能
            is_best = val_metrics.other[0] < best_loss
            if is_best:
                best_auc = val_metrics.auc
                best_acc = val_metrics.acc
                best_aupr = val_metrics.prc
                best_loss = val_metrics.other[0]
                best_epoch = epoch
                color_best = 'red'
                save_path = os.path.join(model_save_path, f"{file_name}.pth")
                torch.save(model.state_dict(), save_path)
            else:
                color_best = 'green'
            
            # 提前停止
            if epoch - best_epoch > early_stopping:
                print(f"Early stop at {epoch}, rG4PredNet!")
                break
            
            # 日志输出
            train_log = (f"{file_name} \t Train Epoch: {epoch}     avg.loss: {train_metrics.other[0]:.4f} "
                        f"Acc: {train_metrics.acc:.4f}, AUC: {train_metrics.auc:.4f}, AUPR: {train_metrics.prc:.4f} lr: {lr:.6f}")
            log_print(train_log, color='green', attrs=['bold'])

            val_log = (f"{file_name} \t Validate  Epoch: {epoch}     avg.loss: {val_metrics.other[0]:.4f} "
                    f"Acc: {val_metrics.acc:.4f}, AUC: {val_metrics.auc:.4f}, AUPR: {val_metrics.prc:.4f} ({best_auc:.4f}) {best_epoch}")
            log_print(val_log, color=color_best, attrs=['bold'])
            print("------------------------------------------------------------------------------------------------------------------")

        print(f"{file_name} AUC: {best_auc:.4f} Acc: {best_acc:.4f} AUPR: {best_aupr:.4f}")
            
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss over Epochs')
        plt.legend()
        plt.savefig(os.path.join(model_save_path, file_name + '_loss_plot.png'))
        plt.show()
        
    if args.validate:
        # 加载嵌入和结构数据
        names, sequences, structs, label = read_csv(os.path.join(data_path, file_name+'_test.tsv'))
        [train_seq, train_struc, train_label], [test_seq, test_struc, test_label] = \
            split_dataset(sequences, structs, label, test_frac=1.0) 

        bert_model_path = args.BERT_model_path
        tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=False)
        model = BertModel.from_pretrained(bert_model_path)
        model = model.to(device)
        model = model.eval()
        bert_embedding = circRNABert(list(test_seq), model, tokenizer, device, 3)  
        test_emb = bert_embedding.transpose([0, 2, 1])  

        structure = np.zeros((len(test_struc), 1, max_length))
        for i in range(len(test_struc)):
            struct = test_struc[i].split(',')
            ti = [float(t) for t in struct]
            ti = np.array(ti).reshape(1, -1)
            structure[i] = np.concatenate([ti], axis=0)

        test_set = MyDataset(test_emb, structure, test_label)
        test_loader = DataLoader(test_set, batch_size=32 * 8, shuffle=False)

        model = Net().to(device)
        model_file = os.path.join(args.model_save_path, file_name+'.pth')
        if not os.path.exists(model_file):
            print('Model file does not exitsts! Please train first and save the model')
            exit()
        model.load_state_dict(torch.load(model_file))

        criterion = nn.BCEWithLogitsLoss()

        met, y_all, p_all = validate(model, device, test_loader, criterion)
        best_auc = met.auc
        best_acc = met.acc
        best_aupr = met.prc
        print("{}_test AUC: {:.4f} Acc: {:.4f} AUPR: {:.4f}".format(file_name, best_auc, best_acc, best_aupr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Welcome to rG4PredNet!')
    parser.add_argument('--data_file', default='rG4', type=str, help='Dataset to train or validate')
    parser.add_argument('--data_path', default='./data', type=str, help='The data path')
    parser.add_argument('--BERT_model_path', default='./BERT_Model', type=str, help='RNAErnie model path, in case you have another model')
    parser.add_argument('--model_save_path', default='./models', type=str, help='Save the trained model for prediction')

    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--validate', default=False, action='store_true')

    parser.add_argument('--seed', default=1024, type=int, help='The random seed')
    parser.add_argument('--early_stopping', type=int, default=5)

    args = parser.parse_args()
    main(args)