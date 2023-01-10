from torchvision import transforms

class Config(object):

    def __init__(self):
        # network
        self.backbone = "resnet50"
        self.use_bottleneck = True
        self.loss_type = "coral"

        # deepda loss related
        self.deepda_loss_weight = 10.0

        # Optimizer related
        self.lr = 3e-3
        self.weight_decay = 5e-4
        self.lr_scheduler = True
        self.lr_gamma = 0.0003
        self.lr_decay = 0.75
        self.momentum = 0.9

        # Training related
        self.batch_size = 32
        self.epochs = 20

        # data related
        self.folder_src = "./data/office31/amazon/images/"
        self.folder_tgt = "./data/office31/webcam/images/"
        self.transform = {
            'train': transforms.Compose(
                [transforms.Resize([256, 256]),
                 transforms.RandomCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])]),
            'test': transforms.Compose(
                [transforms.Resize([224, 224]),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])
        }
        self.n_class = 31
        self.num_workers = 8

        # multi gpu info
        self.device = 'cuda'
        self.rank = 0
        self.world_size = 1
        self.gpu = 5
        self.distributed = False
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'