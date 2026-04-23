PAD = 0
str_embedding_dim = 8
operation_embedding_dim = str_embedding_dim * 2 + 5
# 一共7个特征，station_name和train_number需要nn.embedding："station_name"3, "train_number"4, "scheduled_arrival_time"5,"scheduled_departure_time"6,"arrival_delay"7, "actual_departure_time"8, "departure_delay"9
external_embedding_dim = str_embedding_dim * 2 + 3
# 一共5个特征，wind，weather需要nn.embedding："month"10, "day"11, "wind"12, "weather"13,"temperature"14
hy_embedding_dim = str_embedding_dim * 2 + 8