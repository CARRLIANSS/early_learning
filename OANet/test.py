import torch
from config import Config
import os
from dataset.dataset import CorrespondencesDataset, collate_fn
from torch.utils.data import DataLoader
from models.oanet import OANet
from utils.train_eval_utils import evaluate

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    conf = Config()

    # Load data
    test_dataset = CorrespondencesDataset(conf.data_te, conf)

    print('Using {} dataloader workers every process'.format(conf.num_workers))

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=False,
                             num_workers=conf.num_workers,
                             collate_fn=collate_fn)

    # Create model
    model = OANet(conf).cuda()

    weights_dict = torch.load(os.path.join(conf.best_model_path, 'model_best.pth'), map_location="cuda")
    best_acc = weights_dict['best_acc']
    load_weights_dict = {k: v for k, v in weights_dict['state_dict'].items()
                         if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(load_weights_dict, strict=False)

    result_path = conf.logs + '/' + conf.dataset
    if not os.path.isdir(result_path + '/test'):
        os.makedirs(result_path + '/test')
    if conf.res_path == '':
        conf.res_path = result_path + '/test'

    va_res = evaluate(model, test_loader, conf, 0, 'test')

    output = ''
    name = ['qt_auc20', 'geo_losses', 'cla_losses', 'l2_losses', 'precisions', 'recalls', 'f_scores']
    for i, j in enumerate(va_res):
        output += name[i] + ": " + str(j) + "\n"

    print(output)