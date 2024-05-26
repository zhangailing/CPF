import yaml
import torch

# 获取训练配置并检查设备设置

# 读取指定路径的配置文件，获取全局和特定模型的配置参数
def get_training_config(config_path, model_name):
    with open(config_path, 'r') as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    specific_config = dict(full_config['global'], **full_config[model_name])
    specific_config['model_name'] = model_name
    return specific_config

# 根据配置参数确定模型训练时使用的设备（CPU或GPU），并设置随机种子以保证实验的可重复性
def check_device(conf):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(conf['device'])
    if conf['model_name'] in ['DeepWalk', 'GraphSAGE']:
        is_cuda = False
    else:
        is_cuda = not conf['no_cuda'] and torch.cuda.is_available()
    if is_cuda:
        torch.cuda.manual_seed(conf['seed'])
        torch.cuda.manual_seed_all(conf['seed'])  # if you are using multi-GPU.
    device = torch.device("cuda:" + str(conf['device'])
                          ) if is_cuda else torch.device("cpu")
    return device
