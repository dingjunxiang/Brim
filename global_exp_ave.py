from __future__ import print_function
import os
import numpy as np
import torch
import heapq 
from models.model_porpoise_captum import PorpoiseMMF
from utils.utils import *
from utils.coattn_train_utils import *
from utils.cluster_train_utils import *
import numpy as np
from os import path
import matplotlib.pyplot as plt
import torch,gc
import pandas as pd
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
import numpy as np
import numpy as np
import pandas as pd
from datasets.dataset_survival import  Generic_MIL_Survival_Dataset
from utils.utils import get_custom_exp_code
import torch.nn.functional as F
from sksurv.metrics import concordance_index_censored
import random
from matplotlib import font_manager
# import shap
# import ast
# import seaborn as sb
# from collections import Counter

# 将_data映射到不同的色彩区间中
def convert_data(_data,_max,_min):
    _range = _max - _min
    if _data <= _min:
        return [0, 0, 1]
    if _data >= _max:
    	return [1, 0, 0]
    r = (_data - _min) / _range
    step = (_range / 4)
    idx = int(r * 4)
    h = (idx + 1) * step + _min
    m = idx * step + _min
    
    local_r = (_data - m) / (h - m)
    

    if idx == 0:
        return [0/255, int(local_r * 255)/255, 255/255]
    if idx == 1:
        return [0/255, 255/255, int((1 - local_r) * 255)/255]
    if idx == 2:
        return [int(local_r * 255)/255, 255/255, 0/255]
    if idx == 3:
        return [255/255, int((1 - local_r) * 255)/255, 0/255]

def train_load_save_model(model_obj, model_path):
    if path.isfile(model_path):
        # load model
        print('Loading pre-trained model from: {}'.format(model_path))
        model_obj.load_state_dict(torch.load(model_path),strict=False)
    # else:    
    #     # train model
    #     train(model_obj)
    #     print('Finished training the model. Saving the model to the path: {}'.format(model_path))
    #     torch.save(model_obj.state_dict(), model_path)
    
def find_topk_importances(feature_names, ig_attr_test_norm_sum, k):
        ig_attr_test_norm_sum_temp = ig_attr_test_norm_sum
        ig_attr_test_norm_sum=np.absolute(ig_attr_test_norm_sum)
        index = heapq.nlargest(k, range(len(ig_attr_test_norm_sum)), ig_attr_test_norm_sum.take)
        feature_names_new = []
        ig_attr_test_norm_sum_new = []
        for i in range(len(index)):
            feature_names_new.append(feature_names[index[i]])
            print("featurename:",feature_names[index[i]])
            if ig_attr_test_norm_sum_temp[index[i]] < 0:
                ig_attr_test_norm_sum[index[i]] = ig_attr_test_norm_sum_temp[index[i]]
            ig_attr_test_norm_sum_new.append(ig_attr_test_norm_sum[index[i]])
        # feature_names_new = np.array(feature_names_new)
        # ig_attr_test_norm_sum_new = np.array(ig_attr_test_norm_sum_new)
        print(type(ig_attr_test_norm_sum_new))
        print(feature_names_new)
        print(len(feature_names_new),len(ig_attr_test_norm_sum_new))
        return feature_names_new,ig_attr_test_norm_sum_new
    
def visualize_importances(feature_names, importances, title="Average Feature Importances", cohort= "tcga_blca", seed=1, plot=True, axis_title="Features",slide_ids="ave"):
    print(title)
    importances = np.absolute(importances)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    
    
    importances = np.absolute(importances)
    x_pos = (np.arange(len(feature_names)))
    x_pos = list(x_pos)
    x_pos.reverse()
    if plot:
        plt.figure(figsize=(12,6))
        # font_manager.findfont('sans-serif')
        # plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Arial'
        jco_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']

        # plt.bar(x_pos, importances, align='center',color=color_list_new)
        plt.barh(y= x_pos,width= importances, height=0.35, align='center',color=jco_colors) #[21/255,151/255,165/255],[255/255,190/255,122/255],[250/255,127/255,111/255] 0,
        # plt.xticks(x_pos, feature_names, wrap=True)
        # plt.xlabel(axis_title)
        plt.yticks(x_pos, feature_names, wrap=True)
        # plt.ylabel(axis_title)
        # plt.xlabel("IG value")
        
        # ax = plt.gca()
        # ax.set_xlim([-max(np.absolute(importances)), max(np.absolute(importances))])
        # plt.title(title)
        plt.savefig("/data/home/dingjunxiang/transmil_MMF/IG_result/bridge/{}/{}_local_IG_seed{}_cnv.svg".format(cohort,cohort,seed),format='svg', transparent=True)
        plt.close()

def set_seed(seed):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_local(seed=1,k=10,ig_attr_test_norm_sum=None,feature_names=None,cohort="coadread",slide_ids="ave"):
    ig_attr_test_norm_sum = np.array(ig_attr_test_norm_sum)
    # print(ig_attr_test_norm_sum.shape)
    feature_names, ig_attr_test_norm_sum = find_topk_importances(feature_names, ig_attr_test_norm_sum, k)
    # print("find k success")
    visualize_importances(feature_names=feature_names,importances=ig_attr_test_norm_sum,cohort=cohort,seed=seed,slide_ids=slide_ids)
    return feature_names,ig_attr_test_norm_sum
    print("plot success")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(1)
# cohort = "zsly_argo445"
# cohort = "tcga_coadread"
cohort  = "tcga_blca"
cancer = "blca"
### get feature name
if os.path.exists(r"/data/home/dingjunxiang/transmil_MMF/IG_result/{}_feature_names.csv".format(cohort)):
    genomic_features = pd.read_csv(r"/data/home/dingjunxiang/transmil_MMF/IG_result/{}_feature_names.csv".format(cohort))
    feature_names = list(genomic_features.columns)
else:
    metadata = ['case_id','slide_id','cohort','age','censorship','survival_months','tnm.stage','sex']
    # metadata = ['case_id','slide_id','oncotree_code','age','sex','grade','tnm.stage','TNM.T','TNM.N','TNM.M',
    #                 'histopathology','lymph_node','location_original','mmr','chemotherapy','drug','drug_response','cimp.status',
    #                 'cin.status','tp53.mutation','kras.mutation','braf.mutation','cms','rfs.event','rfs.delay','censorship','survival_months',
    #                 'dfs.event','dfs.delay','location'] #ARGO
    df_slide_data = pd.read_csv(r"/data/home/dingjunxiang/transmil_MMF/datasets_csv/{}_all_clean.csv.zip".format(cohort), low_memory=False)

    genomic_features = df_slide_data.drop(metadata, axis=1)
    feature_names = list(genomic_features.columns)
    df_feature_names = pd.DataFrame(columns=feature_names)
    df_feature_names.to_csv('/data/home/dingjunxiang/transmil_MMF/IG_result/{}_feature_names.csv'.format(cohort),sep=',',index=False)
print("feature_success")
###
rnaseq_index = []
cnv_index = []
mut_index = []
i = 0
for col in genomic_features.columns:
    if "rnaseq" in col:
        rnaseq_index.append(i)
    elif "cnv" in col:
        cnv_index.append(i)
    elif "mut" in col:
        mut_index.append(i)
    i = i + 1
print("rnaseq:",rnaseq_index)
print("cnv:",cnv_index)
# print("mut:",mut_index)


dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all_clean.csv.zip' % ("datasets_csv", cohort),
										   mode = "pathomic",
										   apply_sig = False,
										   data_dir=os.path.join("/mnt/minio/node77/dingjunxiang/TCGA_BLCA/clam_feature/"),#os.path.join("/cache1/ding_junxiang/CLAM/datasets/tcga_coadread_features/"),   # 
										   shuffle = False, 
										   seed = 1, 
										   print_info = True,
										   patient_strat= False,
										   n_bins=4,
										   label_col = 'survival_months',
										   ignore=[])

print("dataset_success")

train_dataset, valid_dataset = dataset.return_splits(from_id=False, 
				csv_path='{}/splits_{}.csv'.format("splits/head1_bridge/{}".format(cohort), 3))
train_dataset.set_split_id(split_id=3)
valid_dataset.set_split_id(split_id=3)
print(type(train_dataset))
print("valid_data:",valid_dataset)
val_loader = get_split_loader(valid_dataset, testing = False, mode= 'pathomic', batch_size=1)
# X_test_path = [item[0].detach().numpy() for item in val_loader]
train_loader = get_split_loader(train_dataset, testing = False, mode= 'pathomic', batch_size=1)
print('training: {}, validation: {}'.format(len(train_dataset), len(valid_dataset)))
print("split_success")

omic_input_dim = valid_dataset.genomic_features.shape[1]
model = PorpoiseMMF(**{'omic_input_dim': omic_input_dim, 'n_classes': 4})
SAVED_MODEL_PATH = 'r/mnt/minio/node77/dingjunxiang/Results/head1_bridge/2_PorpoiseMMF_nll_surv_a0.0_lr2e-03_pathomicreg1e-05_head1_gc32_bilinear/{}_PorpoiseMMF_nll_surv_a0.0_lr2e-03_pathomicreg1e-05_head1_gc32_bilinear_s1/s_3_minloss_checkpoint.pt'.format(cohort)
# SAVED_MODEL_PATH = r"/mnt/minio/node77/dingjunxiang/Results/s_3_minloss_checkpoint.pt"
train_load_save_model(model, SAVED_MODEL_PATH)
for name, param in model.named_parameters():
    if "layer" in name:
        param.requires_grad = False
model.relocate()
model.eval()



print("model_success")

ig = IntegratedGradients(model)
# ig = DeepLift(model)
# ig = shap.Explainer(model)
print("IG success")

def cnv_convert(data):
    # print("int",data)
    if int(data) == -1:
        return [0,0,1]
    if int(data) == 1:
        return [1,0,0]
    if int(data) == 0:
        return [0,1,0]
def mut_convert(data):
    if int(data) == 1:
        return [1,0,0]
    if int(data) == 0:
        return [0,0,1]
    

slide_ids = val_loader.dataset.slide_data['slide_id']


ig_attr_test_norm_sum_list = []
for batch_idx, (data_WSI, data_omic, y_disc, event_time, censor) in enumerate(val_loader):
    gc.collect()
    data_WSI = data_WSI.to(device).unsqueeze(1)
    data_WSI = torch.transpose(data_WSI,1,0)
    data_omic = data_omic.to(device)
    data_WSI = data_WSI.requires_grad_(False)
    # print("raw_genomic:",data_omic)
    
    # print("data_WSI:",data_WSI.requires_grad)
    # print("path_omic_shape:",data_WSI.size(),data_omic.size())
    # ig_attr_test = ig((data_WSI,data_omic))[1] #8
    # print("shap_values:",ig_attr_test)
    ig_attr_test = ig.attribute((data_WSI,data_omic),internal_batch_size = 1)[1] #8
    print("Attribution success")
    data_omic.detach()
    data_WSI.detach()
    ig_attr_test_sum = ig_attr_test.cpu().detach().numpy().sum(0)
    print(ig_attr_test.device)
    ig_attr_test_norm_sum_1 = (ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)).copy()
    print("normalization success")
    ig_attr_test_norm_sum_list.append(ig_attr_test_norm_sum_1)
    
    # dic.update({slide_ids[batch_idx]:gene_names})
    gc.collect()
    torch.cuda.empty_cache()
for batch_idx, (data_WSI, data_omic, y_disc, event_time, censor) in enumerate(train_loader):
    gc.collect()
    data_WSI = data_WSI.to(device).unsqueeze(1)
    data_WSI = torch.transpose(data_WSI,1,0)
    data_omic = data_omic.to(device)
    data_WSI = data_WSI.requires_grad_(False)
    # print("raw_genomic:",data_omic)
    # data_omic = data_omic.half()
    # data_WSI = data_WSI.half()
    # print("data_WSI:",data_WSI.requires_grad)
    # print("path_omic_shape:",data_WSI.size(),data_omic.size())
    # ig_attr_test = ig((data_WSI,data_omic))[1] #8
    # print("shap_values:",ig_attr_test)
    ig_attr_test = ig.attribute((data_WSI,data_omic),internal_batch_size = 1)[1] #8
    print("Attribution success")
    data_omic.detach()
    data_WSI.detach()
    ig_attr_test_sum = ig_attr_test.cpu().detach().numpy().sum(0)
    print(ig_attr_test.device)
    ig_attr_test_norm_sum_1 = (ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)).copy()
    print("normalization success")
    ig_attr_test_norm_sum_list.append(ig_attr_test_norm_sum_1)
    # dic.update({slide_ids[batch_idx]:gene_names})
    gc.collect()
    torch.cuda.empty_cache()

cnv_list = []
mut_list = []
rnaseq_list = []
feature_names_new_cnv = []
feature_names_new_mut = []
feature_names_new_rnaseq = []
ig_attr_test_norm_sum_list = np.array(ig_attr_test_norm_sum_list)
mean_attr = np.mean(ig_attr_test_norm_sum_list,axis=0)
for  i  in cnv_index:
    cnv_list.append(mean_attr[i])
    feature_names_new_cnv.append(feature_names[i])
for  i  in mut_index:
    mut_list.append(mean_attr[i])
    feature_names_new_mut.append(feature_names[i])
for  i  in rnaseq_index:
    rnaseq_list.append(mean_attr[i])
    feature_names_new_rnaseq.append(feature_names[i])
    
# gene_names,index = return_global_feature(seed=1,k=10,ig_attr_test_norm_sum_list=ig_attr_test_norm_sum_list,cohort=cohort,feature_names=feature_names)
mean_attr_cnv = np.array(cnv_list)
mean_attr_mut = np.array(mut_list)
mean_attr_rnaseq = np.array(rnaseq_list)
feature_names_new_cnv = [s.rstrip('_cnv') for s in feature_names_new_cnv]
feature_names_new_mut = [s.rstrip('_mut') for s in feature_names_new_mut]
feature_names_new_rnaseq = [s.rstrip('_rnaseq') for s in feature_names_new_rnaseq]
feature_cnv,attr_cnv = plot_local(seed=1,k=10,ig_attr_test_norm_sum=mean_attr_cnv,feature_names=feature_names_new_cnv,cohort="blca")
feature_mut,attr_mut = plot_local(seed=1,k=10,ig_attr_test_norm_sum=mean_attr_mut,feature_names=feature_names_new_mut,cohort="blca")
feature_rnaseq,attr_rnaseq = plot_local(seed=1,k=10,ig_attr_test_norm_sum=mean_attr_rnaseq,feature_names=feature_names_new_rnaseq,cohort="blca")

df_fea_cnv = pd.DataFrame(zip(feature_cnv,attr_cnv),columns=["gene","attr"])
df_fea_cnv.to_csv(r"/data/home/dingjunxiang/transmil_MMF/IG_result/bridge/{}/{}_attr_cnv.csv".format(cancer),index=False)
df_fea_mut = pd.DataFrame(zip(feature_mut,attr_mut),columns=["gene","attr"])
df_fea_mut.to_csv(r"/data/home/dingjunxiang/transmil_MMF/IG_result/bridge/{}/{}_attr_mut.csv".format(cancer),index=False)
df_fea_rnaseq = pd.DataFrame(zip(feature_rnaseq,attr_rnaseq),columns=["gene","attr"])
df_fea_rnaseq.to_csv(r"/data/home/dingjunxiang/transmil_MMF/IG_result/bridge/{}/{}_attr_rnaseq.csv".format(cancer),index=False)

# ### plot global 

    

    