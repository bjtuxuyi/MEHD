import json
from HY import GaussianDiffusion_ST,ST_Diffusion,Model_all,Transformer_ST,StrFeature_Embedding,WeightedFusion
from torch.optim import AdamW
import setproctitle
from utils.setup_utils import get_args,setup_init,set_log,LR_warmup
from GenDataSet import HG_data_loader
from BatchSTProcess import Batch2toModel,get_max_min_for_interval
import Constants
import torch
import numpy as np

if __name__ == '__main__':
    '''基础设置'''
    opt = get_args()
    device = torch.device("cuda:{}".format(opt.cuda_id) if opt.cuda else "cpu")
    setup_init(opt)
    setproctitle.setproctitle("Model-Training")
    print('dataset:{}'.format(opt.dataset))
    '''日志设置'''
    writer,logdir,model_path,train_result_list,test_result_list,val_result_list,tag = set_log(opt,tag = "_test")
    '''加载数据'''
    basic_event_sequence_list, EventDirectAdjacencyMatrix,str_int_map_dict,trainloader,testloader,valloader,max_for_loc,min_for_loc = HG_data_loader()
    '''模型设置'''
    model = ST_Diffusion(n_steps=opt.timesteps, dim=1 + opt.dim, condition=True, cond_dim=opt.model_dim).to(device)
    diffusion = GaussianDiffusion_ST(model, loss_type=opt.loss_type, seq_length=1 + opt.dim, timesteps=opt.timesteps,sampling_timesteps=opt.samplingsteps, objective=opt.objective,beta_schedule=opt.beta_schedule).to(device)
    transformer = Transformer_ST(adj=EventDirectAdjacencyMatrix,d_model=opt.model_dim, d_rnn=128, d_inner=128, n_layers=4, n_head=4, d_k=16, d_v=16,dropout=0.1, device=device, loc_dim=opt.dim, CosSin=True).to(device)
    strembedding = StrFeature_Embedding(num_embeddings=max(str_int_map_dict.values())+1, embedding_dim=Constants.str_embedding_dim).to(device)
    fusion_B_R = WeightedFusion(feat_dim = 3*opt.model_dim).to(device)
    Model = Model_all(transformer, diffusion,strembedding,fusion_B_R)
    '''训练设置'''
    optimizer = AdamW(Model.parameters(), lr=0.001, betas=(0.9, 0.99))
    step, early_stop, min_loss_test, warmup_steps = 0, 0, 1e20, 5
    '''获取时间维度和位置维度的最大最小值'''
    max_interval,min_interval = get_max_min_for_interval(trainloader,basic_event_sequence_list)
    MAX = [1] + [max_interval] + max_for_loc
    MIN = [0] + [0] + min_for_loc

    gen_result_list = []
    real_result_list = []
    for itr in range(opt.total_epochs):
        print('epoch:{}'.format(itr))
        '''1.模型验证和测试 10个epoch验证一次'''
        if itr % 1==0:
            print('Evaluate!')
            with torch.no_grad():
                Model.eval()
                loss_test_all, vb_test_all, vb_test_temporal_all, vb_test_spatial_all = 0.0, 0.0, 0.0, 0.0
                mae_temporal, rmse_temporal, mae_spatial, mae_lng, mae_lat, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                count = 0
                for batch in valloader:
                    event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch,basic_event_sequence_list,device,Model.transformer,Model.strembedding,Model.fusion_B_R,max_interval,min_interval)
                    sampled_seq = Model.diffusion.sample(batch_size = event_time_non_mask.shape[0],cond=enc_out_non_mask)
                    sampled_seq_temporal_all, sampled_seq_spatial_all = [], []
                    loss = Model.diffusion(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)
                    vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)
                    vb_test_all += vb
                    vb_test_temporal_all += vb_temporal
                    vb_test_spatial_all += vb_spatial
                    loss_test_all += loss.item() * event_time_non_mask.shape[0]
                    real = (event_time_non_mask[:,0,:].detach().cpu() + MIN[1]) * (MAX[1]-MIN[1])
                    gen = (sampled_seq[:,0,:1].detach().cpu() + MIN[1]) * (MAX[1]-MIN[1])

                    '''多列车到站事件拟合实验
                    if count == 0 :
                        academic_1d_kde(real.squeeze().numpy(),tag="True")
                        academic_kde_plot(real.squeeze().numpy(), tag="True")
                        academic_1d_kde(gen.squeeze().numpy(), tag="pred")
                        academic_kde_plot(gen.squeeze().numpy(), tag="pred")
                    count = 1
                    '''
                    # gen_result_list.append(gen.squeeze().numpy())
                    # real_result_list.append(real.squeeze().numpy())
                    # with open('real_result_list.pkl', 'wb') as f:
                    #     pickle.dump(real_result_list, f)


                    assert real.shape==gen.shape
                    # assert real.shape == sampled_seq_temporal_all.shape
                    mae_temporal += torch.abs(real-gen).sum().item()
                    rmse_temporal += ((real-gen)**2).sum().item()
                    # rmse_temporal_mean += ((real-sampled_seq_temporal_all)**2).sum().item()
                    real = event_loc_non_mask[:,0,:].detach().cpu()
                    assert real.shape[1:] == torch.tensor(MIN[2:]).shape
                    real = (real + torch.tensor([MIN[2:]])) * (torch.tensor([MAX[2:]])-torch.tensor([MIN[2:]]))
                    gen = sampled_seq[:,0,-opt.dim:].detach().cpu()
                    gen = (gen + torch.tensor([MIN[2:]])) * (torch.tensor([MAX[2:]])-torch.tensor([MIN[2:]]))
                    assert real.shape==gen.shape
                    mae_spatial += torch.sqrt(torch.sum((real-gen)**2,dim=-1)).sum().item()
                    # mae_spatial_mean += torch.sqrt(torch.sum((real-sampled_seq_spatial_all)**2,dim=-1)).sum().item()
                    total_num += gen.shape[0]
                    assert gen.shape[0] == event_time_non_mask.shape[0]
                # 早停
                if loss_test_all > min_loss_test:
                    early_stop += 1
                    if early_stop >= 100:
                        break
                else:
                    early_stop = 0
                torch.save(Model.state_dict(), model_path+'model_{}.pkl'.format(itr))
                min_loss_test = min(min_loss_test, loss_test_all)
                print(f"loss_val:{loss_test_all/total_num},NLL_temporal_val:{vb_test_temporal_all/total_num}，mae_temporal_val:{mae_temporal/total_num}")
                writer.add_scalar(tag='Evaluation/loss_val',scalar_value=loss_test_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_val',scalar_value=vb_test_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_temporal_val',scalar_value=vb_test_temporal_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_spatial_val',scalar_value=vb_test_spatial_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/mae_temporal_val',scalar_value=mae_temporal/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/rmse_temporal_val',scalar_value=np.sqrt(rmse_temporal/total_num),global_step=itr)
                # writer.add_scalar(tag='Evaluation/rmse_temporal_mean_val',scalar_value=np.sqrt(rmse_temporal_mean/total_num),global_step=itr)
                writer.add_scalar(tag='Evaluation/distance_spatial_val',scalar_value=mae_spatial/total_num,global_step=itr)
                # writer.add_scalar(tag='Evaluation/distance_spatial_mean_val',scalar_value=mae_spatial_mean/total_num,global_step=itr)
                # 持久化保存
                val_result_list["loss_val"].append(loss_test_all / total_num)
                val_result_list["NLL_val"].append(vb_test_all / total_num)
                val_result_list["NLL_temporal_val"].append(vb_test_temporal_all / total_num)
                val_result_list["NLL_spatial_val"].append(vb_test_spatial_all / total_num)
                val_result_list["mae_temporal_val"].append(mae_temporal / total_num)
                val_result_list["rmse_temporal_val"].append(np.sqrt(rmse_temporal / total_num))
                val_result_list["distance_spatial_val"].append(mae_spatial / total_num)


                '''2.模型测试 10个epoch验证一次'''
                loss_test_all, vb_test_all, vb_test_temporal_all, vb_test_spatial_all = 0.0, 0.0, 0.0, 0.0
                mae_temporal, rmse_temporal, mae_spatial, mae_lng, mae_lat, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for batch in testloader:
                    event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch,basic_event_sequence_list,device,Model.transformer,Model.strembedding,Model.fusion_B_R,max_interval,min_interval)
                    sampled_seq = Model.diffusion.sample(batch_size = event_time_non_mask.shape[0],cond=enc_out_non_mask)
                    loss = Model.diffusion(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)
                    vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)
                    vb_test_all += vb
                    vb_test_temporal_all += vb_temporal
                    vb_test_spatial_all += vb_spatial
                    loss_test_all += loss.item() * event_time_non_mask.shape[0]
                    total_num += gen.shape[0]
                print(f"Test: loss_test:{loss_test_all / total_num}; NLL_test:{vb_test_all / total_num}; NLL_temporal_test:{vb_test_temporal_all / total_num}; NLL_spatial_test:{vb_test_spatial_all / total_num}")
                writer.add_scalar(tag='Evaluation/loss_test',scalar_value=loss_test_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_test',scalar_value=vb_test_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_temporal_test',scalar_value=vb_test_temporal_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_spatial_test',scalar_value=vb_test_spatial_all/total_num,global_step=itr)
                # 持久化保存
                test_result_list["loss_test"].append(loss_test_all / total_num)
                test_result_list["NLL_test"].append(vb_test_all / total_num)
                test_result_list["NLL_temporal_test"].append(vb_test_temporal_all / total_num)
                test_result_list["NLL_spatial_test"].append(vb_test_spatial_all / total_num)
        # 学习率预热
        if itr < warmup_steps:
            for param_group in optimizer.param_groups:
                lr = LR_warmup(1e-3, warmup_steps, itr)
                param_group["lr"] = lr
        else:
            for param_group in optimizer.param_groups:
                lr = 1e-3- (1e-3 - 5e-5)*(itr-warmup_steps)/opt.total_epochs
                param_group["lr"] = lr
        writer.add_scalar(tag='Statistics/lr',scalar_value=lr,global_step=itr)

        '''3.模型训练'''
        Model.train()
        loss_all, vb_all, vb_temporal_all, vb_spatial_all, total_num = 0.0, 0.0, 0.0, 0.0, 0.0
        for batch in trainloader:
            event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch,basic_event_sequence_list,device,Model.transformer,Model.strembedding,Model.fusion_B_R,max_interval,min_interval)
            loss = Model.diffusion(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1),enc_out_non_mask)
            optimizer.zero_grad()
            loss.backward()
            loss_all += loss.item() * event_time_non_mask.shape[0]
            vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)
            vb_all += vb
            vb_temporal_all += vb_temporal
            vb_spatial_all += vb_spatial
            writer.add_scalar(tag='Training/loss_step',scalar_value=loss.item(),global_step=step)
            torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.)
            optimizer.step()
            step += 1
            total_num += event_time_non_mask.shape[0]
        # with torch.cuda.device("cuda:{}".format(opt.cuda_id)):
        #     torch.cuda.empty_cache()
        print(f"Training: loss_epoch:{loss_all / total_num}; NLL_epoch:{vb_all / total_num}; NLL_temporal_epoch:{vb_temporal_all / total_num}; NLL_spatial_epoch:{vb_spatial_all / total_num}")
        writer.add_scalar(tag='Training/loss_epoch',scalar_value=loss_all/total_num,global_step=itr)
        writer.add_scalar(tag='Training/NLL_epoch',scalar_value=vb_all/total_num,global_step=itr)
        writer.add_scalar(tag='Training/NLL_temporal_epoch',scalar_value=vb_temporal_all/total_num,global_step=itr)
        writer.add_scalar(tag='Training/NLL_spatial_epoch',scalar_value=vb_spatial_all/total_num,global_step=itr)

        train_result_list["loss_epoch"].append(loss_all / total_num)
        train_result_list["NLL_epoch"].append(vb_all / total_num)
        train_result_list["NLL_temporal_epoch"].append(vb_temporal_all / total_num)
        train_result_list["NLL_spatial_epoch"].append(vb_spatial_all / total_num)
        if itr % 20 == 0 and itr!=1:
            with open('train_result{}.json'.format(tag), 'w') as json_file:
                json.dump(train_result_list, json_file, indent=4)
            with open('test_result{}.json'.format(tag), 'w') as json_file:
                json.dump(test_result_list, json_file, indent=4)
            with open('val_result{}.json'.format(tag), 'w') as json_file:
                json.dump(val_result_list, json_file, indent=4)

