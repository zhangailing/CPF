from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import scipy.sparse as sp
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

import dgl
from models.GCN import GCN
from models.GAT import GAT
from models.GraphSAGE import GraphSAGE
from models.APPNP import APPNP
from models.MoNet import MoNet
from models.GCNII import GCNII
from dgl.nn.pytorch.conv import SGConv
from models.utils import get_training_config

from data.utils import load_tensor_data, load_ogb_data, check_writable
from data.get_dataset import get_experiment_config

from utils.logger import get_logger
from utils.metrics import accuracy

# 解析命令行参数，并将这些参数添加到给定的解析器对象中
def arg_parse(parser):
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument('--teacher', type=str, default='GCN', help='Teacher Model')
    parser.add_argument('--device', type=int, default=2, help='CUDA Device')
    parser.add_argument('--labelrate', type=int, default=20, help='Label rate')
    return parser.parse_args()

# 根据给定的配置创建输出目录和级联目录
def choose_path(conf):
    output_dir = Path.cwd().joinpath('outputs', conf['dataset'], conf['teacher'],
                                     'cascade_random_' + str(conf['division_seed']) + '_' + str(args.labelrate))
    check_writable(output_dir)
    cascade_dir = output_dir.joinpath('cascade')
    check_writable(cascade_dir)
    return output_dir, cascade_dir

# 根据配置选择并初始化图神经网络模型
def choose_model(conf):
    # 设置输入特征维度、隐藏层大小、分类数量、层数、激活函数和 dropout 概率
    if conf['model_name'] == 'GCN':
        model = GCN(
            g=G,
            in_feats=features.shape[1],
            n_hidden=conf['hidden'],
            n_classes=labels.max().item() + 1,
            n_layers=1,
            activation=F.relu,
            dropout=conf['dropout']).to(conf['device'])
    # 使用 GAT 或 SGAT 模型，设置注意力头的数量、层数、输入维度、隐藏层大小、分类数量、激活函数、dropout 概率、负斜率和是否使用残差连接
    elif conf['model_name'] in ['GAT', 'SGAT']:
        if conf['model_name'] == 'GAT':
            num_heads = 8
        else:
            num_heads = 1
        num_layers = 1
        num_out_heads = 1
        heads = ([num_heads] * num_layers) + [num_out_heads]
        model = GAT(g=G,
                    num_layers=num_layers,
                    in_dim=features.shape[1],
                    num_hidden=8,
                    num_classes=labels.max().item() + 1,
                    heads=heads,
                    activation=F.relu,
                    feat_drop=0.6,
                    attn_drop=0.6,
                    negative_slope=0.2,     # negative slope of leaky relu
                    residual=False).to(conf['device'])
    # 使用 GraphSAGE 模型，设置输入维度、嵌入维度、分类数量、层数、激活函数、dropout 概率和聚合器类型
    elif conf['model_name'] == 'GraphSAGE':
        model = GraphSAGE(in_feats=features.shape[1],
                          n_hidden=conf['embed_dim'],
                          n_classes=labels.max().item() + 1,
                          n_layers=2,
                          activation=F.relu,
                          dropout=0.5,
                          aggregator_type=conf['agg_type']).to(conf['device'])
    # 设置输入维度、隐藏层大小、分类数量、激活函数、特征 dropout 概率、边 dropout 概率、alpha 参数和传播步数
    elif conf['model_name'] == 'APPNP':
        model = APPNP(g=G,
                      in_feats=features.shape[1],
                      hiddens=[64],
                      n_classes=labels.max().item() + 1,
                      activation=F.relu,
                      feat_drop=0.5,
                      edge_drop=0.5,
                      alpha=0.1,
                      k=10).to(conf['device'])
    # 使用 MoNet 模型，设置输入维度、隐藏层大小、输出维度、层数、伪坐标维度、混合模型核的数量和 dropout 概率
    elif conf['model_name'] == 'MoNet':
        model = MoNet(g=G,
                      in_feats=features.shape[1],
                      n_hidden=64,
                      out_feats=labels.max().item() + 1,
                      n_layers=1,
                      dim=2,
                      n_kernels=3,
                      dropout=0.7).to(conf['device'])
    # 使用 SGC 模型，设置输入维度、输出维度、卷积步数、缓存标志和是否使用偏置
    elif conf['model_name'] == 'SGC':
        model = SGConv(in_feats=features.shape[1],
                       out_feats=labels.max().item() + 1,
                       k=2,
                       cached=True,
                       bias=False).to(conf['device'])
    # 使用 GCNII 模型，设置输入维度、层数、隐藏层大小、分类数量、dropout 概率、lambda 参数和 alpha 参数
    elif conf['model_name'] == 'GCNII':
        if conf['dataset'] == 'citeseer':
            conf['layer'] = 32
            conf['hidden'] = 256
            conf['lamda'] = 0.6
            conf['dropout'] = 0.7
        elif conf['dataset'] == 'pubmed':
            conf['hidden'] = 256
            conf['lamda'] = 0.4
            conf['dropout'] = 0.5
        model = GCNII(nfeat=features.shape[1],
                      nlayers=conf['layer'],
                      nhidden=conf['hidden'],
                      nclass=labels.max().item() + 1,
                      dropout=conf['dropout'],
                      lamda=conf['lamda'],
                      alpha=conf['alpha'],
                      variant=False).to(conf['device'])
    return model

# 在一个 epoch 内训练图神经网络模型
def train(all_logits, dur, epoch):
    # 初始化和准备
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    # 根据模型类型进行前向传播
    if conf['model_name'] in ['GCN', 'APPNP']:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] in ['GAT', 'SGAT']:
        logits, _ = model(G.ndata['feat'])
    elif conf['model_name'] in ['GraphSAGE', 'SGC']:
        logits = model(G, G.ndata['feat'])
    elif conf['model_name'] == 'MoNet':
        us, vs = G.edges(order='eid')
        udeg, vdeg = 1 / torch.sqrt(G.in_degrees(us).float()), 1 / torch.sqrt(G.in_degrees(vs).float())
        pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)
        logits = model(G.ndata['feat'], pseudo)
    elif conf['model_name'] == 'GCNII':
        logits = model(features, adj)
    else:
        raise ValueError(f'Undefined Model')
    # 计算训练损失和准确率
    logp = F.log_softmax(logits, dim=1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[idx_train], labels[idx_train])
    acc_train = accuracy(logp[idx_train], labels[idx_train])
    # 反向传播和优化
    loss.backward()
    optimizer.step()
    # 记录训练时间
    dur.append(time.time() - t0)
    # 在验证集上进行评估
    model.eval()
    if conf['model_name'] in ['GCN', 'APPNP']:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] in ['GAT', 'SGAT']:
        logits, _ = model(G.ndata['feat'])
    elif conf['model_name'] in ['GraphSAGE', 'SGC']:
        logits = model(G, G.ndata['feat'])
    elif conf['model_name'] == 'MoNet':
        us, vs = G.edges(order='eid')
        udeg, vdeg = 1 / torch.sqrt(G.in_degrees(us).float()), 1 / torch.sqrt(G.in_degrees(vs).float())
        pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)
        logits = model(G.ndata['feat'], pseudo)
    elif conf['model_name'] == 'GCNII':
        logits = model(features, adj)
    else:
        raise ValueError(f'Undefined Model')
    logp = F.log_softmax(logits, dim=1)
    # we save the logits for visualization later
    all_logits.append(logp.cpu().detach().numpy())
    loss_val = F.nll_loss(logp[idx_val], labels[idx_val])
    acc_val = accuracy(logp[idx_val], labels[idx_val])
    acc_test = accuracy(logp[idx_test], labels[idx_test])
    # 打印和返回结果
    print('Epoch %d | Loss: %.4f | loss_val: %.4f | acc_train: %.4f | acc_val: %.4f | acc_test: %.4f | Time(s) %.4f' % (
        epoch, loss.item(), loss_val.item(), acc_train.item(), acc_val.item(), acc_test.item(), dur[-1]))
    return acc_val, loss_val

# 训练图神经网络模型
def model_train(conf, model, optimizer, all_logits):
    # 初始化变量
    dur = []
    best = 0
    cnt = 0
    epoch = 1
    # 训练循环
    while epoch < conf['max_epoch']:
        acc_val, loss_val = train(all_logits, dur, epoch)
        epoch += 1
        # 保存最佳模型
        if acc_val >= best:
            best = acc_val
            state = dict([('model', copy.deepcopy(model.state_dict())),
                          ('optim', copy.deepcopy(optimizer.state_dict()))])
            cnt = 0
        else:
            cnt += 1
        if cnt == conf['patience'] or epoch == conf['max_epoch']:
            print("Stop!!!")
            # print("Saving cascade info...")
            # all_logits = all_logits[: -cnt]
            # if cascade_dir is not None:
            #     for i in range(len(all_logits)):
            #         np.savetxt(cascade_dir.joinpath(str(i) + '.txt'),
            #                    np.exp(all_logits[i]),
            #                    fmt='%.4f', delimiter='\t')
            break
    # 加载最佳模型
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optim'])
    # 打印优化信息
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(np.sum(dur)))

# 用于测试图神经网络模型的性能
def test(conf):
    # 模型评估模式
    model.eval()
    # 根据模型类型进行前向传播
    if conf['model_name'] in ['GCN', 'APPNP']:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] in ['GAT', 'SGAT']:
        logits, G.edata['a'] = model(G.ndata['feat'])
    elif conf['model_name'] in ['GraphSAGE', 'SGC']:
        logits = model(G, G.ndata['feat'])
    elif conf['model_name'] == 'MoNet':
        us, vs = G.edges(order='eid')
        udeg, vdeg = 1 / torch.sqrt(G.in_degrees(us).float()), 1 / torch.sqrt(G.in_degrees(vs).float())
        pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)
        logits = model(G.ndata['feat'], pseudo)
    elif conf['model_name'] == 'GCNII':
        logits = model(features, adj)
    else:
        raise ValueError(f'Undefined Model')
    # 计算预测概率
    logp = F.log_softmax(logits, dim=1)
    # 计算测试集的损失和准确率
    loss_test = F.nll_loss(logp[idx_test], labels[idx_test])
    acc_test = accuracy(logp[idx_test], labels[idx_test])
    # 打印测试结果
    print("Test set results: loss= {:.4f} acc_test= {:.4f}".format(
        loss_test.item(), acc_test.item()))
    # 返回测试准确率和对数概率
    return acc_test, logp

# 配置、训练和评估图神经网络模型，并将结果保存到文件中
if __name__ == '__main__':
    # 解析命令行参数
    args = arg_parse(argparse.ArgumentParser())
    # 加载训练配置
    config_path = Path.cwd().joinpath('models', 'train.conf.yaml')
    conf = get_training_config(config_path, model_name=args.teacher)
    # 加载数据配置
    config_data_path = Path.cwd().joinpath('data', 'dataset.conf.yaml')
    conf['division_seed'] = get_experiment_config(config_data_path)['seed']
    # 设置设备
    if args.device > 0:
        conf['device'] = torch.device("cuda:" + str(args.device))
    else:
        conf['device'] = torch.device("cpu")
    conf = dict(conf, **args.__dict__)
    print(conf)
    # 选择路径和日志
    output_dir, cascade_dir = choose_path(conf)
    logger = get_logger(output_dir.joinpath('log'))
    print(output_dir)
    print(cascade_dir)
    # 设置随机种子
    np.random.seed(conf['seed'])
    torch.manual_seed(conf['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 加载图数据
    if conf['dataset'] in ['arxiv']:
        G, features, labels, idx_train, idx_val, idx_test = load_ogb_data(conf['dataset'], conf['device'])
    else:
        adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test = \
            load_tensor_data(conf['model_name'], conf['dataset'], args.labelrate, conf['device'])
        G = dgl.graph((adj_sp.row, adj_sp.col)).to(conf['device'])
        G.ndata['feat'] = features
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    # The first layer transforms input features of size of 5 to a hidden size of 5.
    # The second layer transforms the hidden layer and produces output features of size 2, corresponding to the two groups of the karate club.
    # 选择模型并设置优化器
    model = choose_model(conf)
    if conf['model_name'] == 'GCNII':
        if conf['dataset'] == 'pubmed':
            conf['wd1'] = 0.0005
        optimizer = optim.Adam([
            {'params': model.params1, 'weight_decay': conf['wd1']},
            {'params': model.params2, 'weight_decay': conf['wd2']},
        ], lr=conf['learning_rate'])
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=conf['learning_rate'],
                               weight_decay=conf['weight_decay'])
    # 训练模型
    all_logits = []
    model_train(conf, model, optimizer, all_logits)
    # 测试模型并获取结果
    acc_test, logp = test(conf)
    preds = logp.max(1)[1].type_as(labels).cpu().numpy()
    labels = labels.cpu().numpy()
    output = np.exp(logp.cpu().detach().numpy())
    acc_test = acc_test.cpu().item()
    # 保存结果
    np.savetxt(output_dir.joinpath('preds.txt'), preds, fmt='%d', delimiter='\t')
    np.savetxt(output_dir.joinpath('labels.txt'), labels, fmt='%d', delimiter='\t')
    np.savetxt(output_dir.joinpath('output.txt'), output, fmt='%.4f', delimiter='\t')
    np.savetxt(output_dir.joinpath('test_acc.txt'), np.array([acc_test]), fmt='%.4f', delimiter='\t')
    # 保存注意力权重
    if 'a' in G.edata:
        print('Saving Attention...')
        edge = torch.stack((G.edges()[0], G.edges()[1]),0)
        sp_att = sp.coo_matrix((G.edata['a'].cpu().detach().numpy(), edge.cpu()), shape=adj.cpu().size())
        sp.save_npz(output_dir.joinpath('attention_weight.npz'), sp_att, compressed=True)
