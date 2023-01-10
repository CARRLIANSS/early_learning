

class Config(object):

    def __init__(self):
        # pre_processing
        self.pre_processing = 'sift-2000'

        # data related
        self.dataset = 'yfcc'
        self.data_te = './data_dump/yfcc-sift-2000-test.hdf5' # yfcc-sift-2000-test.hdf5 or sun3d-sift-2000-test.hdf5
        self.data_tr = './data_dump/yfcc-sift-2000-train.hdf5' # yfcc-sift-2000-train.hdf5 or sun3d-sift-2000-train.hdf5
        self.data_va = './data_dump/yfcc-sift-2000-val.hdf5' # yfcc-sift-2000-val.hdf5 or or sun3d-sift-2000-val.hdf5

        # network related
        self.net_channels = 128 # number of channels in a layer, PointCN embedding size
        self.net_depth = 12 # total number of layers, including multiple iterative networks
        self.clusters = 500  # cluster number in OANet
        self.share = False # share the parameter in iterative network
        self.use_fundamental = False  # train fundamental matrix estimation
        self.use_ratio = 0  # use ratio test. 0: not use, 1: use before network, 2: use as side information
        self.use_mutual = 0  # use mutual nearest neighbor check. 0: not use, 1: use before network, 2: use as side information
        self.ratio_test_th = 0.8  # ratio test threshold
        self.iter_num = 1  # The number of network iterations

        # loss related
        self.geo_loss_margin = 0.1 # clamping margin in geometry loss
        self.loss_classif = 1.0  # weight of the classification loss
        self.loss_essential = 0.5 # weight of the essential loss
        self.loss_essential_init_iter = 20000 # initial iterations to run only the classification loss
        self.weight_decay = 0 # l2 decay
        self.momentum = 0.9

        # objective related
        self.obj_geod_th = 1e-4  # theshold for the good geodesic distance
        self.obj_geod_type = 'episym' # type of geodesic distance
        self.obj_num_kp = 2000 # number of keypoints per image
        self.obj_top_k = -1 # number of keypoints above the threshold to use for essential matrix estimation, put -1 to use all

        # training related
        self.num_workers = 8
        self.train_batch_size = 32
        self.train_lr = 1e-3 # learning rate
        self.lrf = 0.1
        self.epochs = 30 # training iterations to perform
        self.logs = './logs'  # save directory name inside results
        self.checkpoint_path = './checkpoint/'
        self.pretrained = './pretrained/yfcc/essential/sift-2000/model_best.pth' # yfcc essential and fundamental, sun3d, gl3d
        if self.use_fundamental:
            self.best_model_path = './best_model/' + self.dataset + '/fundamental/' + self.pre_processing + '/'
        else:
            self.best_model_path = './best_model/' + self.dataset + '/essential/' + self.pre_processing + '/'

        # testing related
        self.use_ransac = False # use ransac when testing?
        self.model_path = self.best_model_path  # saved best model path for test
        self.res_path = ''  # path for saving results, valid and test's result

        # multi gpu info
        self.device = 'cuda'
        self.rank = 0
        self.world_size = 1
        self.gpu = 5
        self.distributed = False
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'