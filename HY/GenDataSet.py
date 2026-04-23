import torch
from torch.utils.data  import Dataset, DataLoader
from torch.utils.data import Subset
from utils.setup_utils import read_HG

# a = ["actual_arrival_time"0, "lng"1, "lat"2, "station_name"3, "train_number"4, "scheduled_arrival_time"5,
#  "scheduled_departure_time"6,
#  "arrival_delay"7, "actual_departure_time"8, "departure_delay"9, "month"10, "day"11, "wind"12, "weather"13,
#  "temperature"14, "major_holiday"15]  event_feature[:,:,[0,1,2,3,4,5,6]],event_feature[:,:,[7,8,9,10,11]]

def custom_nanmin(tensor, dim):
    masked = tensor.clone()
    masked[torch.isnan(masked)] = float('inf')  # 用极大值填充nan
    min_val, _ = torch.min(masked, dim=dim)  # 取最小值
    min_val[min_val == float('inf')] = 0  # 处理全nan列
    return min_val


def custom_nanmax(tensor, dim):
    masked = tensor.clone()
    masked[torch.isnan(masked)] = -float('inf')  # 用极小值填充nan
    max_val, _ = torch.max(masked, dim=dim)
    max_val[max_val == -float('inf')] = 0
    return max_val

class TimeSeriesDataset(Dataset):
    def __init__(self, feature_matrix, selected_features):
        # 转换为张量并保持原始数据副本
        self.raw_data = torch.as_tensor(feature_matrix, dtype=torch.float32)

        # 特征选择逻辑增强：当未指定时默认选择全部特征
        if selected_features is None:
            self.selected_features = list(range(self.raw_data.shape[-1]))
        else:
            self.selected_features = selected_features

            # 初始化归一化参数
        self.data_min = None
        self.data_max = None
        self.data_range = None
        self._compute_normalization_params()

    def _compute_normalization_params(self):
        # 提取目标特征维度数据（形状：[days, nodes, selected_features]）
        features_to_normalize = self.raw_data[..., self.selected_features]
        # 展平前两维进行全局统计（形状变为：[days*nodes, selected_features]）
        flattened = features_to_normalize.view(-1, len(self.selected_features))

        flattened_non_zero = flattened.clone()
        flattened_non_zero[flattened == 0] = float('nan')

        # 计算全局最小值和最大值
        self.data_min = custom_nanmin(flattened_non_zero, dim=0).unsqueeze(0)
        self.data_max = custom_nanmax(flattened_non_zero, dim=0).unsqueeze(0)

        # 处理全零特征列的特殊情况
        nan_mask = torch.isnan(self.data_min)
        self.data_min[nan_mask] = 0
        self.data_max[nan_mask] = 0

        # 计算极差并添加极小值防止除零
        self.data_range = (self.data_max - self.data_min) + 1e-8

    def _apply_normalization(self, data):
        # 提取目标特征数据
        selected_data = data[..., self.selected_features]
        zero_mask = (selected_data == 0)
        # 执行归一化计算（自动广播最小值/极差）
        normalized = (selected_data - self.data_min) / self.data_range
        normalized[zero_mask] = 0
        # 重构完整特征矩阵
        data[..., self.selected_features] = normalized
        return data

    def __len__(self):
        return self.raw_data.shape[0]

    def __getitem__(self, idx):
        daily_data = self.raw_data[idx].clone()  # 创建副本避免污染原始数据
        return self._apply_normalization(daily_data)

    def get_original_data(self, idx):
        return self.raw_data[idx]

    def get_max_min(self):
        return self.data_max[0,[0,1]].tolist(), self.data_min[0,[0,1]].tolist()

def split_train_test_val_dataset(dataset):
    # 按时间顺序划分索引
    train_idx = range(0, 25)  # 前25天
    val_idx = range(25, 28)  # 中间3天
    test_idx = range(28, 32)  # 最后4天
    # 创建子数据集
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)
    # 训练集（启用打乱）
    train_loader = DataLoader(train_set,batch_size=1,shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    return train_loader,test_loader,val_loader

def HG_data_loader():
    selected_features = [1,2,10,11,14] # 选择需要归一化的列
    basic_event_sequence_list, EventDirectAdjacencyMatrix, F_EventFeatureMatrix, str_int_map_dict = read_HG()
    event_count = 0
    for basic_event_sequence in basic_event_sequence_list: event_count = event_count + len(basic_event_sequence)
    assert event_count == F_EventFeatureMatrix.shape[1]
    dataset = TimeSeriesDataset(F_EventFeatureMatrix,selected_features=selected_features)  #需要原始数据前三列和interval的max，min
    max_for_loc,min_for_loc = dataset.get_max_min()
    train_loader,test_loader,val_loader = split_train_test_val_dataset(dataset)
    return basic_event_sequence_list, EventDirectAdjacencyMatrix,str_int_map_dict,train_loader,test_loader,val_loader,max_for_loc,min_for_loc






