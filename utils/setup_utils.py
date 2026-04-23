import argparse
import torch
import random
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import pickle

'''命令行参数'''
def get_args():
    parser = argparse.ArgumentParser(description='程序运行命令行参数设置')
    parser.add_argument('--seed', type=int, default=1234, help='')
    parser.add_argument('--mode', type=str, default='train', help='')
    parser.add_argument('--total_epochs', type=int, default=1000, help='')
    parser.add_argument('--machine', type=str, default='none', help='')
    parser.add_argument('--loss_type', type=str, default='l2', choices=['l1', 'l2', 'Euclid'], help='')
    parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear', 'cosine'], help='')
    parser.add_argument('--dim', type=int, default=2, help='', choices=[1, 2, 3])
    parser.add_argument('--dataset', type=str, default='JH_CHSD',choices=['JH_CHSD'], help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--timesteps', type=int, default=100, help='')
    parser.add_argument('--samplingsteps', type=int, default=100, help='')
    parser.add_argument('--objective', type=str, default='pred_noise', help='')
    parser.add_argument('--cuda_id', type=str, default='0', help='')
    parser.add_argument('--model_dim', type=int, default=256, help='')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args

'''学习率预热'''
def LR_warmup(lr, epoch_num, epoch_current):
    return lr * (epoch_current+1) / epoch_num

def setup_init(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

'''设置日志'''
def set_log(opt,tag = "_HG"):  # 设置log存储tag
    logdir = "./logs/{}_timesteps_{}".format(opt.dataset + tag, opt.timesteps)  # 日志保存路径
    model_path = './ModelSave_HG/dataset_{}_timesteps_{}/'.format(opt.dataset + tag, opt.timesteps)  # 模型保存路径
    if not os.path.exists('./ModelSave_HG'): os.mkdir('./ModelSave_HG')
    if not os.path.exists('./logs'): os.mkdir('./logs')
    if 'train' in opt.mode and not os.path.exists(model_path): os.mkdir(model_path)
    writer = SummaryWriter(log_dir=logdir, flush_secs=5)
    # 持久化文件
    train_result_list = {}
    test_result_list = {}
    val_result_list = {}
    train_result_list["loss_epoch"], train_result_list["NLL_epoch"], train_result_list["NLL_temporal_epoch"], \
    train_result_list["NLL_spatial_epoch"] = [], [], [], []
    test_result_list["loss_test"], test_result_list["NLL_test"], test_result_list["NLL_temporal_test"], \
    test_result_list["NLL_spatial_test"] = [], [], [], []
    val_result_list["loss_val"], val_result_list["NLL_val"], val_result_list["NLL_temporal_val"], val_result_list[
        "NLL_spatial_val"], \
    val_result_list["mae_temporal_val"], val_result_list["rmse_temporal_val"], val_result_list["distance_spatial_val"] \
        = [], [], [], [], [], [], []
    print("******日志设置完成******")
    return writer,logdir,model_path,train_result_list,test_result_list,val_result_list,tag

'''读取超图数据'''
def read_HG():
    # 读取基础超边
    with open("../dataset/Jinghu_HG/basic_event_sequence_list.pkl", "rb") as f:
        basic_event_sequence_list = pickle.load(f)
    print(f"基础超边共有{len(basic_event_sequence_list)}个，元素示例：{basic_event_sequence_list[0]}")
    # 读取有向邻接矩阵
    EventDirectAdjacencyMatrix = np.load("../dataset/Jinghu_HG/EventDirectAdjacencyMatrix.npy")
    print(f"事件图有向邻接矩阵维度：{EventDirectAdjacencyMatrix.shape}, 数据类型：{EventDirectAdjacencyMatrix.dtype},存在为1，不存在为0")
    # 读取特征矩阵
    with open("../dataset/Jinghu_HG/F_EventFeatureMatrix.pkl", "rb") as f:
        F_EventFeatureMatrix = pickle.load(f)
    # 替换F_EventFeatureMatrix（ndarray） 这里面有nan值 车没开
    F_EventFeatureMatrix = np.nan_to_num(F_EventFeatureMatrix, nan=0)
    print(f"特征矩阵维度为{F_EventFeatureMatrix.shape},元素示例[0,0,:]：{list(F_EventFeatureMatrix[0, 0, :])}")
    with open("../dataset/Jinghu_HG/str_int_map_dict.json", "r", encoding="utf-8") as f:
        str_int_map_dict = json.load(f)
    print(
        f"字符串int映射字典长度为：{len(str_int_map_dict)},元素实例：{list(str_int_map_dict.keys())[0]}:{list(str_int_map_dict.values())[0]}")
    print("******超图数据读取完成******")


    '''
    基础超边元素示例：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    事件图有向邻接矩阵维度：(637, 637), 数据类型：float64,存在为1，不存在为0
    特征矩阵元素示例[0,0,:]：[6.6, 116.37907, 39.864464, 1.0, 2.0, 6.6, 6.6, 0.0, 6.6, 0.0, 10.0, 9.0, 3.0, 4.0, 22.0, 0.0]
    字符串int映射字典元素实例：Beijingnan Railway Station:1
    '''

    return basic_event_sequence_list,EventDirectAdjacencyMatrix,F_EventFeatureMatrix,str_int_map_dict

if __name__ == '__main__':
    read_HG()
