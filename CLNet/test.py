import torch
from config import Config
import os
from dataset.dataset import CorrespondencesDataset, collate_fn
from torch.utils.data import DataLoader
from models.clnet import CLNet
from utils.train_eval_utils import evaluate

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    conf = Config()

    # Load data
    test_dataset = CorrespondencesDataset(conf.data_te, conf)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=False,
                             num_workers=conf.num_workers,
                             collate_fn=collate_fn)

    print('Using {} dataloader workers every process'.format(conf.num_workers))

    # Create model
    model = CLNet(conf).cuda()

    weights_dict = torch.load(os.path.join(conf.best_model_path, 'model_best.pth'), map_location="cuda")
    best_acc = weights_dict['best_acc']
    # 参考：https://blog.csdn.net/hustwayne/article/details/120324639
    load_weights_dict = {k.replace('module.', ''): v for k, v in weights_dict['state_dict'].items()}  # DDP训练的模型加载
    model.load_state_dict(load_weights_dict, strict=False)

    result_path = conf.logs + '/' + conf.dataset
    if not os.path.isdir(result_path + '/test'):
        os.makedirs(result_path + '/test')
    if conf.res_path == '':
        conf.res_path = result_path + '/test'

    aucs5, aucs10, aucs20 = evaluate(model, test_loader, conf)
    va_res = [aucs5, aucs10, aucs20]

    output = ''
    name = ['AUC@5', 'AUC@10', 'AUC@20']
    for i, j in enumerate(va_res):
        output += name[i] + ": " + str(j) + "\n"

    print(output)