

class Config(object):

    def __init__(self):
        self.mode = 'train'
        self.batch_size = 1
        self.scribble_dilation = 0
        self.bbox = False
        self.scribble = False
        self.key = 'align'
        self.notrain = False
        self.n_runs = 5 # number of test runs

        self.seed = 1234

        self.input_size = [417, 417]
        self.cuda_visable = '0,1,2,3,4,5'
        self.gpu_id = '5'
        self.dataset = 'VOC'
        self.n_steps = 30000 # train: 30000 test: 1000
        self.label_sets = 0 # fold
        self.lr_milestones = [10000, 20000, 30000]
        self.align_loss_scaler = 1
        self.ignore_label = 255
        self.print_interval = 100
        self.save_pred_every = 10000
        self.model = {'align': True}
        self.task = {'n_ways': 1, 'n_shots': 1, 'n_queries': 1}
        self.optim = {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0005}
        self.exp_str = 'VOC_align_sets_0_1way_1shot_[train]'
        self.path = {'init_path': './pretrained_model/vgg16-397923af.pth',
                     'VOC': {'data_dir': './data/Pascal/VOCdevkit/VOC2012/', 'data_split': 'trainaug'},
                     'COCO': {'data_dir': './data/COCO/', 'data_split': 'train'}}