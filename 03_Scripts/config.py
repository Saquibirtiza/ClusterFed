######### Shared config #########
config_dict = {

    #### Data ####
    'dataset': 'UNSW_IoT', # choices: CIFAR10, CIFAR100, ACI_IoT, UNSW_IoT
    'input_features': 42, # choices: UNSW:42, ACI:80
    'seed': 1, # set random seed
    'num_clients': 10, # choices: Int
    'alpha': 1e-1,  # choices= 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5
    'data_dir': '../04_Output/data/', # where data is located
    'save_dir': '../04_Output/', # where model directory is to be created
    'force_restart_training': False, # set False if you want to restart federated training from the last global server model
    'force_restart_hparam': False, # set False if you want to restart unsupervised hyperparameter tuning from the last global server model

    #### GPU and virtualization ####
    'main_device': 'cuda:0', # choices: 0--7
    # 'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7', # choices: 0--7
    'CUDA_VISIBLE_DEVICES': '0,1', # choices: 0--7
    'virtualize': True, # choices: True, False
    'client_vram': 3000, # vram for client in MB

    #### Training config ####
    # Train/eval technique
    'train_mode': 'simclr', # choices: sup, simclr, simsiam, byol, specloss, rotpred, orchestra, custom
    'da_method': 'simclr', # does not matter in our case # choices: sup, simclr, simsiam, byol, specloss, rotpred, orchestra
    'div_aware_update': False, # use divergence aware update for SimSiam/BYOL's predictor (used in cross-silo setting by prior works--not used with Orchestra)
    'stateful_client': False, # create a stateful client that does not share target model every round (used in cross-silo setting by prior works--not used with Orchestra)

    # Model
    'model_class': 'mlp', # choices: VGG, res18, res34, res56, mlp
    'block': 'BasicBlock', # choices: BasicBlock, Bottleneck

    # Fit / Eval fractions
    'fraction_fit': 0.8, # choices: Float <= 1
    'fraction_eval': 1., # choices: Float <= 1 (this only matters for supervised training; can be ignored for SSL)

    # Hyperparameters
    'num_rounds': 20, # number of communication rounds
    'local_bsize': 16, # batch size for client training
    'local_epochs':15, # number of local epochs for client training
    'local_lr': 0.001, # Learning rate for client training

    # Temperature
    'main_T': 0.1, # Online model's temperature (used in SimCLR and Orchestra)

    # EMA
    'ema_value': 0.996, # Exponential Moving Average (EMA) for target model

    # Clustering arguments for Orchestra and SpecLoss 
    'num_global_clusters': 128, # number of global clusters (used in Orchestra and SpecLoss)
    'num_local_clusters': 16, # number of local clusters (used in Orchestra only)
    'cluster_m_size': 128, # Memory size per client (used in Orchestra only)
}

# "/home/saquib/Saquib/Federated Learning/Unsupervised/orchestra/saved_models/model_ACI_IoT_0.001_alpha_custom_10_clients_16_bsize_2_lepochs_0.1_fit_128_gclusters_16_lclusters_1_seed.pth"

eval_dict = {
    'main_device': 'cuda:0', # choices: 0--7
    'pretrained_loc': "../04_Output/saved_models/model_ACI_IoT_0.001_alpha_custom_10_clients_16_bsize_15_lepochs_0.8_fit_128_gclusters_16_lclusters_1_seed.pth", # set this to location where the global model is saved
    'forced_path': "1212312312", # nuisace variable; ignore
    'batch_size': 256, # batch size for linear eval
    'warmup_epochs': 0, # warmup epochs (some papers use this; we don't)
    'warmup_lr': 0, # warmup learning rate for warmup training
    'num_epochs': 10, # number of epochs for linear eval
    'base_lr': 30, # initial learning rate for linear eval training
    'final_lr': 0, # final learning rate that the scheduler decays to
    'model_class': config_dict['model_class'], # choices: VGG, res18, res34, res56
    'dataset': config_dict['dataset'], # choices: CIFAR10, CIFAR100, ACI_IoT
    'use_multi_gpu': False, # choices: True, False
    'subset_proportion': 1, # choices: float <= 1; denotes amount of data used for semi-supervised training
    'subset_force_class_balanced': False, # use class-balanced dataset for semi-supervised training 
    'subset_seed': 0, # just another seed
}

def get_config_dict():
    return config_dict

def get_eval_dict():
    return eval_dict


