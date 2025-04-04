import os
import sys
import torch
import torch_geometric.datasets as GeoData
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn.data_parallel import DataParallel
import random
import numpy as np

from opt import *
from ACA_GCN import ACA_GCN
from GraphSAGE import ACA_GraphSAGE
import time
from utils.metrics import accuracy, auc_2, prf_2
from dataloader import dataloader, dataloader_adni, dataloader_odir

epoch_times = []

if __name__ == '__main__':
    set_seed(123)
    opt = OptInit().initialize()

    print('  Loading dataset ...')
    if opt.dataset == 'ABIDE':
        dl = dataloader()
    if opt.dataset == 'ADNI':
        dl = dataloader_adni()
    if opt.dataset == 'ODIR':
        dl = dataloader_odir()

    raw_features, y, nonimg = dl.load_data()

    n_folds = 10
    cv_splits = dl.data_split(n_folds)


    corrects = np.zeros(n_folds, dtype=np.int32) 
    accs = np.zeros(n_folds, dtype=np.float32) 
    aucs = np.zeros(n_folds, dtype=np.float32)
    prfs = np.zeros([n_folds, 3], dtype=np.float32)

    for fold in range(n_folds):
        print("\r\n========================== Fold {} ==========================".format(fold)) 
        train_ind = cv_splits[fold][0] 
        test_ind = cv_splits[fold][1] 

        print('  Constructing graph data...')
        # extract node features  
        node_ftr = dl.get_node_features(train_ind)  #The centrality of the processed node features (image features X feature matrix) was not utilized.

        # get PAE inputs
        edge_index, edgenet_input, pd_ftr_dim, nonimg, n = dl.get_PAE_inputs(nonimg, opt) #1对边（节点对）进行删选：返回值有3个加上, edge_index_copy

        # normalization for PAE
        edgenet_input = (edgenet_input - edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)

        # build network architecture  
        model = ACA_GCN(node_ftr.shape[1], opt.num_classes, opt.dropout, edge_dropout=opt.edropout, hgc=opt.hgc,
                       lg=opt.lg, edgenet_input_dim=2*nonimg.shape[1],).to(opt.device)


        model = model.to(opt.device)

        # build loss, optimizer, metric 
        loss_fn =torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        features_cuda = torch.tensor(node_ftr, dtype=torch.float32).to(opt.device)

        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(opt.device)

        labels = torch.tensor(y, dtype=torch.long).to(opt.device)
        nonimg = torch.tensor(nonimg, dtype=torch.float32).to(opt.device)
        fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)

        def train(flag=None): 
            print("  Number of training samples %d" % len(train_ind))
            print("  Number of testing samples %d" % len(test_ind))
            print("  Start training...\r\n")
            acc = 0
            for epoch in range(opt.num_iter):
                start = time.perf_counter()
                model.train()
                optimizer.zero_grad()


                with torch.set_grad_enabled(True):

                    _, node_logits = model(features_cuda, edge_index, edgenet_input, pd_ftr_dim, nonimg, n, flag=0)

                    z1, _ = model(features_cuda, edge_index, edgenet_input, pd_ftr_dim, nonimg, n, flag=1)

                    z2, _ = model(features_cuda, edge_index, edgenet_input, pd_ftr_dim, nonimg, n, flag=2)

                    #（1）classification loss
                    loss1 = loss_fn(node_logits[train_ind], labels[train_ind]) ##
                    #（2）feature loss
                    loss2 = model.loss(z1, z2)
                    loss = loss1 + (0.8 * loss2)
                    # loss = loss1

                    loss.backward(retain_graph=True)
                    optimizer.step()
                # correct_train, acc_train = accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])
                epoch_time = time.perf_counter() - start
                epoch_times.append(epoch_time)
                model.eval()

                with torch.set_grad_enabled(False):
                    _, node_logits = model(features_cuda, edge_index, edgenet_input, pd_ftr_dim, nonimg, n, flag=0)

                logits_test = node_logits[test_ind].detach().cpu().numpy()
                correct_test, acc_test = accuracy(logits_test, y[test_ind])
                auc_test = auc_2(logits_test, y[test_ind], opt.num_classes)
                # auc_all = auc_m(logits_test, y[test_ind], opt.num_classes)
                prf_test = prf_2(logits_test, y[test_ind], opt.num_classes)
                # 转换为张量
                print("Epoch: {},\tce loss: {:.5f}".format(epoch, loss.item()))

                if acc_test > acc and epoch > 9:
                    acc = acc_test
                    correct = correct_test
                    aucs[fold] = auc_test


                    prfs[fold] = prf_test

                    if opt.ckpt_path !='':
                        if not os.path.exists(opt.ckpt_path):
                            os.makedirs(opt.ckpt_path)
                        torch.save(model.state_dict(), fold_model_path)

            accs[fold] = acc 
            corrects[fold] = correct
            print("\r\n => Fold {} test accuacry {:.5f}".format(fold, acc))
            ff = open("acc.txt", "a+")
            ff.write("%.5f  " % acc)
            ff.write("\n")
            ff.close()

        def evaluate():
            print("  Number of testing samples %d" % len(test_ind))
            print('  Start testing...')
            model.load_state_dict(torch.load(fold_model_path))
            model.eval()
           
            node_logits, _ = model(features_cuda, edge_index, edgenet_input, pd_ftr_dim, nonimg, n, flag=0)
    
            inference_time = end_time - start_time
            print(f"Inference time: {inference_time:.6f} seconds")

            logits_test = node_logits[test_ind].detach().cpu().numpy()

            corrects[fold], accs[fold] = accuracy(logits_test, y[test_ind])
            aucs[fold] = auc_2(logits_test, y[test_ind], opt.num_classes)
            prfs[fold] = prf_2(logits_test, y[test_ind], opt.num_classes)
            # AUC.append(aucs[fold])

            print("  Fold {} test accuracy {:.5f}, AUC {:.5f}".format(fold, accs[fold], aucs[fold]))

        if opt.train == 1:
            train()
        elif opt.train == 0:
            evaluate()

    print("\r\n========================== Finish ==========================") 
    n_samples = raw_features.shape[0]
    acc_nfold = np.sum(corrects)/n_samples
    print("=> Average test accuracy in {}-fold CV: {:.5f}".format(n_folds, acc_nfold))
    print("=> Average test AUC in {}-fold CV: {:.5f}".format(n_folds, np.mean(aucs)))

    se, sp, f1 = np.mean(prfs, axis=0)
    print("=> Average test sensitivity {:.4f}, specificity {:.4f}, F1-score {:.4f}".format(se, sp, f1))
    print(f"Mean epoch time: {np.mean(epoch_times):.2f}±{np.std(epoch_times):.2f} s")
    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3  # 转换为GB
    print(f"Peak GPU memory: {peak_mem:.2f} GB")

    f = open("result_acc_auc_sen_spe_f1.txt", "a+")
    f.write("%.5f  %.5f  %.5f  %.5f  %.5f" % (acc_nfold, np.mean(aucs), se, sp, f1))
    f.write("\r\n\n")
    f.close()

