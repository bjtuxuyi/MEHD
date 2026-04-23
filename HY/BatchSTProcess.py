import torch
import numpy as np

def generate_index_list(no_operation_train_index,eventnum_in_trains):
    count = 0
    index_list = []
    for i, eventnum_in_train in enumerate(eventnum_in_trains):
        if i not in no_operation_train_index:
            sub_index_list = []
            for k,j in enumerate(range(eventnum_in_train)):
                if k != eventnum_in_train-1:
                    sub_index_list.append(count)
                count = count + 1
            index_list.append(sub_index_list)
        else:
            count = count + eventnum_in_train
    a = [index for indexs in index_list for index in indexs]
    return a


def pad_time(insts):
    # print(len(insts)) # batch
    max_len = max(len(inst) for inst in insts)
    batch_seq = np.array([
        inst + [0] * (max_len - len(inst))
        for inst in insts])
    return torch.tensor(batch_seq, dtype=torch.float32)
def pad_feature(insts):
    # print(len(insts)) # batch
    max_len = max(inst.shape[1] for inst in insts)
    padding_tensor = [torch.nn.functional.pad(inst, (0, 0, 0, max_len-inst.shape[1])).squeeze() for inst in insts]
    return torch.stack(padding_tensor, dim=0)

def get_max_min_for_interval(train_dataloader,basic_event_sequence_list):
    max_interval,min_interval = [],[]
    for batch in train_dataloader:
        event_time_batch = batch[:, :, 0]
        event_time = []
        for basic_event_sequence in basic_event_sequence_list:
            event_time.append(event_time_batch[:, list(basic_event_sequence)].squeeze().numpy().tolist())
        event_interval = [[event - seq[index - 1] if index > 0 else event for index, event in enumerate(seq)] for seq in event_time]
        max_interval_batch = list(sorted(set(sorted([interval for u in event_interval for interval in u]))))[-1]
        min_interval_batch = list(sorted(set(sorted([interval for u in event_interval for interval in u]))))[0]
        # print(max_interval_batch)
        if min_interval_batch == 0:
            min_interval_batch = list(sorted(set(sorted([interval for u in event_interval for interval in u]))))[1]
        max_interval.append(max_interval_batch)
        min_interval.append(min_interval_batch)

    return sorted(max_interval)[-1],sorted(min_interval)[1]



def Batch2toModel(batch,basic_event_sequence_list,device,transformer,strembedding,fusion_B_R,max_interval,min_interval):
    '''1.提取事件特征'''
    event_time_batch = batch[:,:,0]
    event_lng_batch = batch[:,:,1]
    event_lat_batch = batch[:,:,2]
    event_feature_batch = batch[:,:,3:batch.shape[2]-1]
    '''2.提取出每个事件序列'''
    event_time,event_lng,event_lat,event_feature = [],[],[],[]
    for basic_event_sequence in basic_event_sequence_list:
        event_time.append(event_time_batch[:,list(basic_event_sequence)].squeeze().numpy().tolist())
        event_lng.append(event_lng_batch[:,list(basic_event_sequence)].squeeze().numpy().tolist())
        event_lat.append(event_lat_batch[:,list(basic_event_sequence)].squeeze().numpy().tolist())
        event_feature.append(event_feature_batch[:,list(basic_event_sequence),:])

    # 提取事件间隔  # 这个没有归一化
    event_interval = [[event - seq[index - 1] if index > 0 else event for index, event in enumerate(seq)] for seq in event_time]
    '''3.填充事件序列至max长度'''
    event_time = pad_time(event_time)# [69, 14]
    event_interval = pad_time(event_interval)

    event_interval = (event_interval - min_interval) / (max_interval - min_interval)

    event_lng = pad_time(event_lng)
    event_lat = pad_time(event_lat)
    event_loc = torch.stack([event_lng,event_lat],dim=-1)
    event_feature = pad_feature(event_feature)# [69, 14,12]  # 这里需要修改map的0索引，填充的0和原始的0不能是一回事
    event_operation_feature, event_external_feature = event_feature[:,:,[0,1,2,3,4,5,6]],event_feature[:,:,[7,8,9,10,11]]
    assert event_time.shape == event_lng.shape == event_lat.shape == event_feature[:,:,0].squeeze().shape
    event_time,event_interval, event_loc,event_operation_feature, event_external_feature = event_time.to(device),event_interval.to(device), event_loc.to(device),event_operation_feature.to(device), event_external_feature.to(device)
    # print("event_time:",event_time.shape)
    # print("event_interval:",event_interval.shape)
    # print("event_loc:",event_loc.shape)
    # print("event_operation_feature:",event_operation_feature.shape)
    # print("event_external_feature:",event_external_feature.shape)
    '''4.事件编码'''
    B_enc_out, R_enc_out, mask,lengths = transformer(batch,event_time, event_loc,event_operation_feature, event_external_feature,strembedding,device)
    '''5.获取基础超边非掩码部分的编码输出、时间信息、位置信息'''
    enc_out_non_mask = []
    event_interval_non_mask = []
    event_loc_non_mask = []
    length_list = []
    for index in range(mask.shape[0]): # 遍历批次
        length = int(sum(mask[index]).item()) # 获取当前批次的掩码，计算非填充元素的参数
        length_list.append(length)
        if length > 1: # 避免处理空序列
            enc_out_non_mask += [i.unsqueeze(dim=0) for i in B_enc_out[index][:length - 1]]  # 存的是每一个编码/时间/位置的编码
            event_interval_non_mask += [i.unsqueeze(dim=0) for i in event_interval[index][1:length]]
            event_loc_non_mask += [i.unsqueeze(dim=0) for i in event_loc[index][1:length]]
    '''6.整理基础超边'''
    enc_out_non_mask = torch.cat(enc_out_non_mask, dim=0)  # [sum(length_list-1),192]
    event_interval_non_mask = torch.cat(event_interval_non_mask, dim=0)  # [sum(length_list-1)]
    event_loc_non_mask = torch.cat(event_loc_non_mask, dim=0)  # [sum(length_list-1),2]
    event_interval_non_mask = event_interval_non_mask.reshape(-1, 1, 1)  # [sum(length_list-1), 1, 1]
    event_loc_non_mask = event_loc_non_mask.reshape(-1, 1, 2)  # [sum(length_list-1), 1, 2]
    enc_out_non_mask = enc_out_non_mask.reshape(event_interval_non_mask.shape[0], 1, -1)  # [sum(length_list-1), 1, 192]
    '''7.整理相关超边'''
    no_operation_train_index = (lengths == 0).nonzero().squeeze().tolist()
    eventnum_in_trains = [len(i) for i in basic_event_sequence_list]
    event_index = generate_index_list(no_operation_train_index, eventnum_in_trains)
    R_enc_non_mask = R_enc_out[event_index,:].reshape(event_interval_non_mask.shape[0],1,-1)
    '''8.事件节点特征融合'''
    enc_out = fusion_B_R(enc_out_non_mask,R_enc_non_mask) # *1

    return event_interval_non_mask, event_loc_non_mask, enc_out # *1
    # return event_interval_non_mask, event_loc_non_mask, enc_out_non_mask



