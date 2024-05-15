import os
import sys
import torch
import torch_geometric.datasets as GeoData
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn.data_parallel import DataParallel
import random
import numpy as np
import networkx as nx

from opt import * 
from AA_GCN import AA_GCN

from utils.metrics import accuracy, auc_2, prf_2
from dataloader import dataloader, dataloader_adni, dataloader_odir

from edge_net import weight

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

    raw_features, y, nonimg = dl.load_data() #  nonimg非图像特征

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
        node_ftr = dl.get_node_features(train_ind)  #node_ftr处理后的（图像特征X  特征矩阵）度中心性没用到
        # print(node_ftr.shape)

        # get PAE inputs
        edge_index, edgenet_input, pd_ftr_dim, nonimg, n = dl.get_PAE_inputs(nonimg, opt) #1对边（节点对）进行删选：返回值有3个加上, edge_index_copy

        #拓扑级别--随机的边丢失
        # def drop_edges(edge_index, p):
        #     num_edges = edge_index.shape[1]
        #     num_keep = int(p * num_edges)
        #     mask = [True] * num_keep + [False] * (num_edges - num_keep)
        #     random.shuffle(mask)
        #     mask = torch.tensor(mask, dtype=torch.bool)
        #     new_edge_index = edge_index[:, mask]
        #     return new_edge_index
        # p = 0.7  # 保留70%的边
        # edge_index1 = drop_edges(edge_index, p)
        # edgenet_input1 = np.zeros([len(edge_index1[0]), 2 * pd_ftr_dim], dtype=np.float32)
        # i = 0
        # j = 0
        # flatten_ind1 = 0
        # num = []
        # for x in range(0, n):
        #     num.append(x)
        # for edge_index1[0][i] in num:
        #     for edge_index1[1][j] in num:
        #         edgenet_input1[flatten_ind1] = np.concatenate((nonimg[edge_index1[0][i]], nonimg[edge_index1[1][j]]))
        #         flatten_ind1 += 1
        #         i += 1
        #         j += 1
        #         continue
        #     break
        #
        # 节点级别,掩码
        # def random_select_and_zero_out(node_ftr, p, dim_to_zero):
        #     num_nodes, num_dims = node_ftr.shape
        #     num_selected = int(p * num_nodes)
        #     selected_indices = random.sample(range(num_nodes), num_selected)
        #     new_node_ftr = np.copy(node_ftr)
        #     for index in selected_indices:
        #         new_node_ftr[index, dim_to_zero] = 0
        #     return new_node_ftr
        # p = 0.3
        # dim_to_zero = random.sample(range(2000), 500)
        # node_ftr1 = random_select_and_zero_out(node_ftr, p, dim_to_zero)

        # if opt.type == 'Degree':
        #     edge_index1, edgenet_input1 = dl.edge_index1(edge_index, pd_ftr_dim, nonimg, n, opt)
        #     edge_index2, edgenet_input2 = dl.edge_index2(edge_index, pd_ftr_dim, nonimg, n, opt)
        # if opt.type == 'PageRank':
        #     edge_index1, edgenet_input1 = dl.edge_index1(edge_index, pd_ftr_dim, nonimg, n, opt)
        #     edge_index2, edgenet_input2 = dl.edge_index2(edge_index, pd_ftr_dim, nonimg, n, opt)
        # if opt.type == 'Eigenvector':
        #     edge_index1, edgenet_input1 = dl.edge_index1(edge_index, pd_ftr_dim, nonimg, n, opt)
        #     edge_index2, edgenet_input2 = dl.edge_index2(edge_index, pd_ftr_dim, nonimg, n, opt)
        #丢节点
        # def remove_random_nodes_and_edges(edge_index,node_ftr, drop_percent):
        #     # print("特征矩阵形状：", node_ftr.shape)
        #
        #     # 获取节点数量
        #     num_nodes = node_ftr.shape[0]
        #     # print("节点数量：", num_nodes)
        #
        #     # 计算要删除的节点数量
        #     drop_num = int(num_nodes * drop_percent)
        #
        #     # 随机选择要删除的节点索引
        #     drop_node_indices = np.random.choice(num_nodes, size=drop_num, replace=False)
        #     # print("节点索引：", drop_node_indices)
        #
        #     # 将特征矩阵中对应的节点特征置为零
        #     node_ftr[drop_node_indices] = 0
        #
        #     # print("删除后特征矩阵形状：", node_ftr.shape)
        #
        #     return edge_index, node_ftr
        #
        #
        # # 删除节点和相应的边的百分比
        # drop_percent = 0.45
        # # 删除节点和相应的边
        # edge_index_node, node_ftr = remove_random_nodes_and_edges(edge_index, node_ftr, drop_percent)
        # edgenet_input_node = np.zeros([len(edge_index[0]), 2 * pd_ftr_dim], dtype=np.float32)


        # normalization for PAE
        edgenet_input = (edgenet_input - edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)
        # edgenet_input1 = (edgenet_input1 - edgenet_input1.mean(axis=0)) / edgenet_input1.std(axis=0)
        # edgenet_input_node = (edgenet_input - edgenet_input_node.mean(axis=0)) / edgenet_input.std(axis=0)
        # edgenet_input2 = (edgenet_input2 - edgenet_input2.mean(axis=0)) / edgenet_input2.std(axis=0)

        # build network architecture  
        model = AA_GCN(node_ftr.shape[1], opt.num_classes, opt.dropout, edge_dropout=opt.edropout, hgc=opt.hgc,
                       lg=opt.lg).to(opt.device)
        model = model.to(opt.device)

        # build loss, optimizer, metric 
        loss_fn =torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        features_cuda = torch.tensor(node_ftr, dtype=torch.float32).to(opt.device)
        # features_cuda1 = torch.tensor(node_ftr1, dtype=torch.float32).to(opt.device)

        # print(edge_index_node)




        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(opt.device)

        # print("形状")
        # print(edge_index_node.shape)

        # edge_index_node = torch.tensor(edge_index_node, dtype=torch.long).to(opt.device)
        # edgenet_input_node = torch.tensor(edgenet_input, dtype=torch.float32).to(opt.device)

        # edge_index1 = torch.tensor(edge_index1, dtype=torch.long).to(opt.device)
        # edgenet_input1 = torch.tensor(edgenet_input1, dtype=torch.float32).to(opt.device)

        # edge_index2 = torch.tensor(edge_index2, dtype=torch.long).to(opt.device)
        # edgenet_input2 = torch.tensor(edgenet_input2, dtype=torch.float32).to(opt.device)

        labels = torch.tensor(y, dtype=torch.long).to(opt.device)
        fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)

        #PAE计算边
        pae_weight = weight(opt.dropout, edge_dropout=opt.edropout, edgenet_input_dim=2 * nonimg.shape[1])
        edge_weight3, edge_index3 = pae_weight.weight_(edge_index, edgenet_input, flag=0)
        edge_weight1, edge_index1 = pae_weight.weight_(edge_index, edgenet_input, flag=1)
        edge_weight2, edge_index2 = pae_weight.weight_(edge_index, edgenet_input, flag=2)
        # edge_weight2, edge_index2 = pae_weight.weight_(edge_index, edgenet_input, flag=3)

        # edge_weight11, edge_index11 = pae_weight.weight_(edge_index1, edgenet_input1, flag=1)  # 边置换
        # edge_weight22, edge_index22 = pae_weight.weight_(edge_index1, edgenet_input1, flag=2)  # 边丢失


        def train(): 
            print("  Number of training samples %d" % len(train_ind))
            print("  Start training...\r\n")
            acc = 0
            for epoch in range(opt.num_iter):
                model.train()  
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    _, node_logits = model(features_cuda, edge_index3, edge_weight3)
                    z1, _ = model(features_cuda, edge_index1, edge_weight1)
                    z2, _ = model(features_cuda, edge_index2, edge_weight2)

                    #（1）分类loss
                    loss1 = loss_fn(node_logits[train_ind], labels[train_ind]) ##
                    #（2）特征loss
                    loss2 = model.loss(z1, z2)
                    loss = loss1 + loss2

                    loss.backward(retain_graph=True)
                    optimizer.step()
                # correct_train, acc_train = accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])
                model.eval()

                with torch.set_grad_enabled(False):
                    _, node_logits = model(features_cuda, edge_index3, edge_weight3)
                logits_test = node_logits[test_ind].detach().cpu().numpy()
                correct_test, acc_test = accuracy(logits_test, y[test_ind])
                auc_test = auc_2(logits_test, y[test_ind], opt.num_classes)
                prf_test = prf_2(logits_test, y[test_ind], opt.num_classes)

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
            node_logits, _ = model(features_cuda, edge_index, edgenet_input)

            logits_test = node_logits[test_ind].detach().cpu().numpy()
            corrects[fold], accs[fold] = accuracy(logits_test, y[test_ind])

            aucs[fold] = auc_2(logits_test, y[test_ind], opt.num_classes)
            prfs[fold] = prf_2(logits_test, y[test_ind], opt.num_classes)

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

    f = open("result_acc_auc_sen_spe_f1.txt", "a+")
    f.write("%.5f  %.5f  %.5f  %.5f  %.5f" % (acc_nfold, np.mean(aucs), se, sp, f1))
    f.write("\r\n\n")
    f.close()

