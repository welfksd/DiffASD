import torch
import os
# from train import build_model
from dataset import DCASE2022Dataset
from sample import *
import joblib
from tqdm import tqdm
import sklearn
from scipy.stats import hmean
from unet import *
import re

def build_model(config):
    unet = UNetModel(config.data.image_size, 64, dropout=0.0, n_heads=4,in_channels=config.data.imput_channel)
    return unet

def gwrp(data, decay, dim=1):
    data = np.sort(data, axis=dim)[:, ::-1]
    gwrp_w = decay ** np.arange(data.shape[dim])
    #gwrp_w[gwrp_w < 0.1] = 0.1
    sum_gwrp_w = np.sum(gwrp_w)
    data = data * gwrp_w
    out = np.sum(data, axis=dim)
    out = out / sum_gwrp_w
    # print(out.shape)
    return out  

def train_gmm(config, t_sne_data):
    model = build_model(config)
    model = torch.nn.DataParallel(model)
    ckpt_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir)
    ckpt = torch.load(os.path.join(ckpt_save_dir, str(config.model.load_chp)))
    
    model.load_state_dict(ckpt['model'])
    model.to(config.model.device)
    model.eval()
    
    visual = config.model.visual
    
    with torch.no_grad():

        for train_dir in config.train_dirs:
            machine = train_dir.split('/')[-2]
            
            t_sne_data[machine] = {} # for plot
            train_dir = [train_dir]  # 
            train_dataset = DCASE2022Dataset(train_dir, ckpt_save_dir, is_train=False, need_convert_mel=False, use_gmm=True)
            trainloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.data.batch_size,
                shuffle=True,
                num_workers=config.model.num_workers,
                drop_last=True,
                pin_memory=True,
            )
            section_id_list = [re.findall(f'section_[0-9][0-9]', meta)[0] for meta in train_dataset.meta2label.keys() if machine in meta]
            
            train_features = [[] for section in section_id_list]
            scale_params_dir = os.path.join(train_dataset.model_save_dir, f'scale_params/{machine}_mean_std_max_min.npy')
            mean_std_max_min = torch.load(scale_params_dir)
            for data, class_labels in tqdm(trainloader, desc=f'generating gmm features for {machine}'):
                data = data.to(config.model.device)
                
                test_time_steps = torch.Tensor([config.model.test_time_steps]).type(torch.int64).to(config.model.device)  # 从x_t开始降噪
                at = compute_alpha(test_time_steps.long(),config)
                noisy_spec = at.sqrt() * data + (1- at).sqrt() * torch.randn_like(data).to('cuda')
                seq = range(0 , config.model.test_time_steps, config.model.skip)
                reconstructed = ReconstructionWithTime(data, noisy_spec, seq, model, config, config.model.w, class_labels)  # 200 除以 skip = 扩散步数，reconstrcted中包含了所有的生成的结果
                # reconstructed = ReconstructionWithCondition(data, noisy_spec, seq, model, config, config.model.w, class_labels)  # 200 除以 skip = 扩散步数，reconstrcted中包含了所有的生成的结果
                data_reconstructed = reconstructed[-1].to(config.model.device)  # [bs, 1, 256, 256]，现在只拿最后的作为结果
                

                
                for i in range(data.shape[0]):
                    section_str = train_dataset.label2meta[int(class_labels[i])].split('-')[-1]  # 把section_00转换成数字
                    mean, std, max, min = mean_std_max_min[section_str]
                    section = int(section_str[-1][-1])
                    # submel = ((torch.squeeze(data[i] - data_reconstructed[i].to(config.model.device), dim=0) + 1) / 2 * (max - min) + min) * std + mean
                    submel = torch.squeeze(data[i] - data_reconstructed[i].to(config.model.device), dim=0)
                    feature = gwrp(submel.cpu().numpy(), config.gwrp_decays.twfr_gmm[machine]).reshape(1, -1)  # 这是用的twfr-gmm
                    # feature = torch.mean(submel).cpu().numpy().reshape(1, -1)
                    # feature = submel.mean(dim=-1).cpu().numpy().reshape(1, -1)
                    
                    train_features[section].append(feature)
                    ########## GMM-submel ##########
                    
                    ########## GMM ##########
                    # feature_ori = gwrp(data[i].cpu().numpy(), 1.00).reshape(1, -1)  # 这是用的twfr-gmm
                    # feature_rec = gwrp(data_reconstructed[i].cpu().numpy(), 1.00).reshape(1, -1)
                    # section = int(train_dataset.label2meta[int(class_labels[i])].split('-')[-1][-1])  # 把section_00转换成数字
                    # train_features[section].append(feature_ori)
                    # train_features[section].append(feature_rec)
                    ########## GMM ##########
                                   
                    
            for i in range(len(section_id_list)):
                train_features[i] = np.concatenate(train_features[i], axis=0)
                
            # Kmeans
            # from sklearn.cluster import KMeans
            # gmms = [KMeans(n_clusters=1) for i in range(len(section_id_list))]
            # for i in range(len(section_id_list)):
            #     gmms[i].fit(train_features[i])
                
            # GMM
            from sklearn.mixture import GaussianMixture
            gmms = [GaussianMixture(n_components=config.gmm_ns.twfr_gmm[machine], covariance_type='full',
                            # means_init=means_init, precisions_init=precisions_init,
                            tol=1e-6, reg_covar=1e-6, verbose=2) for i in range(len(section_id_list))]
            for i in range(len(section_id_list)):
                gmms[i].fit(train_features[i])
                
            if visual:
                for i, sec in enumerate(section_id_list):
                    t_sne_data[machine][sec] = {}
                    t_sne_data[machine][sec]['train_features'] = train_features[i]
                    t_sne_data[machine][sec]['gmm'] = gmms[i]   
            
            gmm_save_dir = os.path.join(ckpt_save_dir, 'gmm')
            if not os.path.exists(gmm_save_dir):
                os.mkdir(gmm_save_dir)
            joblib.dump(gmms, os.path.join(gmm_save_dir, f'{machine}-gmm'))  


def test(config):
    model = build_model(config)
    model = torch.nn.DataParallel(model)  
    ckpt_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir)
    ckpt = torch.load(os.path.join(ckpt_save_dir, str(config.model.load_chp)))
    
    model.load_state_dict(ckpt['model'])
    model.to(config.model.device)
    model.eval()
    
    

    cal_metric(model, config, ckpt_save_dir)
    

def cal_metric(model, config, ckpt_save_dir):
    visual = config.model.visual
    # train_gmm(config, t_sne_data)
    
    
    metric_all = {}  # {'ToyTrain': [auc_s, auc_t, pauc, final_metric], }
    with torch.no_grad():
        for test_dir in config.test_dirs:  
            
            machine = test_dir.split('/')[-2]
            
            test_dir = [test_dir]
            test_dataset = test_dataset = DCASE2022Dataset(test_dir, ckpt_save_dir, is_train=False, need_convert_mel=True, use_gmm=False)
            testloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=config.data.batch_size,
                shuffle=True,
                num_workers=config.model.num_workers,
                drop_last=True,
                pin_memory=True,
            )
            section_id_list = [re.findall(f'section_[0-9][0-9]', meta)[0] for meta in test_dataset.meta2label.keys() if machine in meta]
            

            metric_section = {section:[] for section in section_id_list}  # {'section':[auc_s, auc_t, pauc]}
            y_true_s = {section:[] for section in section_id_list}  # {'section': []} 
            y_true_t = {section:[] for section in section_id_list}  # {'section': []} 
            y_true_all = {section:[] for section in section_id_list}
            y_pred_s = {section:[] for section in section_id_list}  # {'section': []} 
            y_pred_t = {section:[] for section in section_id_list}  # {'section': []} 
            y_pred_all = {section:[] for section in section_id_list}  # 
            # y_pred_s = {section:[[],[]] for section in section_id_list}  # {'section': []} 
            # y_pred_t = {section:[[],[]] for section in section_id_list}  # {'section': []} 
            # y_pred_all = {section:[[],[]] for section in section_id_list}  
            
            gmms = joblib.load(os.path.join(ckpt_save_dir, 'gmm', f'{machine}-gmm'))
            
            scale_params_dir = os.path.join(test_dataset.model_save_dir, f'scale_params/{machine}_mean_std_max_min.npy')
            mean_std_max_min = torch.load(scale_params_dir)
            test_features = [[] for section in section_id_list]
            normal_score = []
            abnomal_score = []
            for data, class_labels, labels, domains in tqdm(testloader, desc=f'test {machine}'):
                data = data.to(config.model.device)
                    
                test_time_steps = torch.Tensor([config.model.test_time_steps]).type(torch.int64).to(config.model.device)
                at = compute_alpha( test_time_steps.long(),config)
                noisy_image = at.sqrt() * data + (1- at).sqrt() * torch.randn_like(data).to('cuda')
                seq = range(0 , config.model.test_time_steps, config.model.skip)
                reconstructed = ReconstructionWithTime(data, noisy_image, seq, model, config, config.model.w, class_labels)
                # reconstructed = ReconstructionWithCondition(data, noisy_image, seq, model, config, config.model.w, class_labels)
                data_reconstructed = reconstructed[-1]
                
                for i in range(data.shape[0]):
                    section = test_dataset.label2meta[int(class_labels[i])].split('-')[-1]
                    section_id = int(section[-1])
                    mean, std, max, min = mean_std_max_min[section]
                    ########## GMM-submel ##########
                    # # submel = ((torch.squeeze(data[i] - data_reconstructed[i].to(config.model.device), dim=0) + 1) / 2 * (max - min) + min) * std + mean
                    # submel = torch.squeeze(data[i] - data_reconstructed[i].to(config.model.device), dim=0)
                    # # paint.mel_paint(submel.cpu().numpy(), f'./sub-ToyCar.png')
                    # # feature = torch.mean(submel).cpu().numpy().reshape(1, -1)
                    # feature = gwrp(submel.cpu().numpy(), config.gwrp_decays.twfr_gmm[machine]).reshape(1, -1)   # 这是用的twfr-gmm
                    # # feature = submel.mean(dim=-1).cpu().numpy().reshape(1, -1)  # min-gmm
                    # annmscore_gmm = - np.max(gmms[int(section[-1])]._estimate_log_prob(feature))
                    ########## GMM ##########
                    
                    

                    
                    ########## KMeans ##########
                    # submel = torch.squeeze(data[i] - data_reconstructed[i].to(config.model.device), dim=0)
                    # # feature = gwrp(submel.cpu().numpy(), 1.00).reshape(1, -1)   # 这是用的twfr-gmm
                    # # feature = submel.mean(dim=-1).cpu().numpy().reshape(1, -1)  # 这是用的min-gmm
                    # feature = torch.mean(submel).cpu().numpy().reshape(1, -1)
                    # annmscore_gmm = gmms[int(section[-1])].transform(feature)[0]
                    ########## KMeans ##########
                    
                    ########## GMM ##########
                    # feature_ori = gwrp(data[i].cpu().numpy(), 1.00).reshape(1, -1)  # 这是用的twfr-gmm
                    # feature_rec = gwrp(data_reconstructed[i].cpu().numpy(), 1.00).reshape(1, -1)
                    # annmscore_gmm_ori = np.max(gmms[int(section[-1])]._estimate_log_prob(feature_ori))
                    # annmscore_gmm_rec = np.max(gmms[int(section[-1])]._estimate_log_prob(feature_rec))
                    ########## GMM ##########
                    
                    # annmscore_sub = torch.abs(torch.mean(submel)).cpu().numpy()
                    
                    
                    annmscore_l1 = torch.sum(torch.abs(data_reconstructed[i].to(config.model.device) - data[i])).cpu().numpy()
                    # annmscore_mse = torch.nn.functional.mse_loss(data_reconstructed[i].to(config.model.device), data[i]).cpu().numpy()
                    
                    
                    ########## feature ##########
                    # feature_ori = gwrp(data[i].cpu().numpy(), 1.00).reshape(1, -1)
                    # feature_rec = gwrp(data_reconstructed[i].cpu().numpy(), 1.00).reshape(1, -1)
                    # # 计算余弦相似度
                    # from sklearn.metrics.pairwise import cosine_similarity
                    # annmscore_cos = cosine_similarity(feature_ori, feature_rec)
                    ########## feature ##########
                    # test_features[section_id].append(feature)
                    annmscore = annmscore_l1
                    

                    if section_id == 2:
                        if labels[i] == 0:  
                            normal_score.append(annmscore)
                        else:
                            abnomal_score.append(annmscore)
                    
                    if domains[i] == 0:  # source
                        y_true_s[section].append(labels[i].cpu().numpy())
                        y_pred_s[section].append(annmscore)
                        # y_pred_s[section][0].append(annmscore)
                        # y_pred_s[section][1].append(annmscore_mse)
                    else:
                        y_true_t[section].append(labels[i].cpu().numpy())
                        y_pred_t[section].append(annmscore)
                        # y_pred_t[section][0].append(annmscore)
                        # y_pred_t[section][1].append(annmscore)
                    y_true_all[section].append(labels[i].cpu().numpy())
                    y_pred_all[section].append(annmscore)
                    # y_pred_all[section][0].append(annmscore_gmm_ori)
                    # y_pred_all[section][1].append(annmscore_mse)
            # if visual:
            #     for i, section in enumerate(section_id_list):
            #         temp = np.concatenate(test_features[i], axis=0)
            #         t_sne_data[machine][section]['test_features'] = temp
                

            # tsne_data_path = os.path.join('./t-SNE/twfr-gmm/GMM-Mix', 'tsne_data.db')
            # joblib.dump(t_sne_data, tsne_data_path)

            max_fpr = 0.1
            hmean_auc_s = []
            hmean_auc_t = []
            hmean_pauc = []
            
            for section in y_true_s.keys():
                # max_mse = np.max(y_pred_all[section][1])
                # max_gmm = np.max(y_pred_all[section][0])
                # y_pred_s[section][0] = y_pred_s[section][0] / max_gmm * max_mse
                # y_pred_s[section] = 100 * y_pred_s[section][0] + 1 * y_pred_s[section][1]
                
                # y_pred_t[section][0] = y_pred_t[section][0] / max_gmm * max_mse
                # y_pred_t[section] = 100 * y_pred_t[section][0] + 1 * y_pred_t[section][1]
                
                # y_pred_all[section][0] = y_pred_all[section][0] / max_gmm * max_mse
                # y_pred_all[section] = 100 * y_pred_all[section][0] + 1 * y_pred_all[section][1]
                
                auc_s = sklearn.metrics.roc_auc_score(y_true_s[section], y_pred_s[section])
                auc_t = sklearn.metrics.roc_auc_score(y_true_t[section], y_pred_t[section])
                p_auc = sklearn.metrics.roc_auc_score(y_true_all[section], y_pred_all[section], max_fpr=max_fpr)    
                         
                metric_section[section] = [auc_s, auc_t, p_auc]
                hmean_auc_s.append(auc_s)
                hmean_auc_t.append(auc_t)
                hmean_pauc.append(p_auc)
                
                plt.figure(figsize=(8, 6))
                import seaborn as sns
                sns.distplot(normal_score, bins=50, hist=True, kde=True, norm_hist=False,
                            rug=False, vertical=False, label='normal wave',
                            axlabel='score', 
                            # rug_kws={'label': 'RUG', 'color': 'b'},
                            kde_kws={'label': 'KDE', 'color': 'g', 'linestyle': '--'},
                            hist_kws={'color': 'g'})
                sns.distplot(abnomal_score, bins=50, hist=True, kde=True, norm_hist=False,
                            rug=False, vertical=False, label='anomaly wave',
                            axlabel='score', 
                            # rug_kws={'label': 'RUG', 'color': 'k'},
                            kde_kws={'label': 'KDE', 'color': 'k', 'linestyle': '--'},
                            hist_kws={'color': 'k'})
                # plt.axvline(threshold, color='r', label='threshold:' + str(threshold))
                plt.legend()
                plt.savefig('./result_diffasd_l1_0903.png')
                
                print(f'{machine}-{section}-- auc_s: {auc_s} | auc_t: {auc_t} | pauc: {p_auc}')
                
            hmean_auc_s = hmean(hmean_auc_s)
            hmean_auc_t = hmean(hmean_auc_t)
            hmean_pauc = hmean(hmean_pauc)
            
            final_metric = hmean([hmean_auc_s, hmean_auc_t, hmean_pauc])
            metric_all[machine] = [hmean_auc_s, hmean_auc_t, hmean_pauc, final_metric]
    print(metric_all)

    hmean_machine = {}
    for key in metric_all.keys():
        hmean_machine[key] = hmean(metric_all[key])
    print(hmean_machine)
    
    hmean_all = []
    for key in hmean_machine:
        hmean_all.append(hmean_machine[key])
        
    result_all = hmean(hmean_all)
    print(result_all)
    
    return result_all
    
            
    
    