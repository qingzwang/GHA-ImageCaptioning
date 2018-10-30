import torch
import torchvision as tv
from PIL import Image
from models.model import CNNPlusCNN2
from models.model import NewCNNPlusCNNHierAtt


class Config(object):
    def __init__(self):
        self.max_epoch = 50
        self.batch_size = 1
        self.encoder_name = 'resnet101'
        self.kernel_size = 3
        self.num_layers = 6
        self.channels = 300
        self.prediction_dim = 1024
        # ak_token: 9489
        self.voc_size = 9489
        self.attention_tracker = False
        self.width = 224
        self.height = 224
        self.is_gpu = True
        self.is_train = True
        self.shuffle = True
        self.num_workers = 1
        self.transformer = tv.transforms.Compose(
            [
                tv.transforms.Resize((self.width + 32, self.height + 32)),
                tv.transforms.RandomCrop(self.width, self.height),
                tv.transforms.RandomHorizontalFlip()
             ]
        )
        self.is_dotatt = True
        self.image_dir = ''  ## dir of images
        self.train_ann_file = 'data/files/captions_train2014.json'
        self.val_ann_file = 'data/files/captions_val2014.json'
        self.split_file = 'data/files/caption_id.pkl'
        self.vocab_file = 'data/files/vocab.pkl'
        self.f = torch.nn.GLU(1)
        self.keep_prob = 0.5
        if self.is_dotatt:
            string = ('%s_numl%s_kz%s_dotatt_visualization%s') % \
                     (self.encoder_name, self.num_layers, self.kernel_size, self.keep_prob)
            self.trained_model = 'trained_models/' + string + '.pth'
        else:
            string = ('%s_numl%s_kz%s_nnatt_keepprob%s')%\
                     (self.encoder_name, self.num_layers, self.kernel_size, self.keep_prob)
            self.trained_model = 'trained_models/' + string + '.pth'
        self.log_file = 'logs/' + self.trained_model.split('/')[1] + '.txt'
        self.result_file = 'results/captions_val2014_' + string + '_results.json'
        self.annfile = None


class HierConfig(object):
    def __init__(self):
        self.max_epoch = 50
        self.batch_size = 1
        self.encoder_name = 'resnet101'
        self.kernel_size = 3
        self.num_layers = 6
        self.channels = 300
        self.hier_att_hidden_size = 512
        self.hier_att_lang_hidden_size = 512
        self.prediction_dim = 1024
        self.voc_size = 9489
        self.width = 224
        self.height = 224
        self.is_gpu = True
        self.is_train = True
        self.shuffle = True
        self.num_workers = 1
        self.transformer = tv.transforms.Compose(
            [
                tv.transforms.Resize((self.width + 32, self.height + 32)),
                tv.transforms.RandomCrop(self.width, self.height),
                tv.transforms.RandomHorizontalFlip()
            ]
        )
        self.is_dotatt = True
        self.image_dir = '' ##dir of images
        self.train_ann_file = 'data/files/captions_train2014.json'
        self.val_ann_file = 'data/files/captions_val2014.json'
        self.split_file = 'data/files/caption_id.pkl'
        self.vocab_file = 'data/files/vocab.pkl'
        self.keep_prob = 0.5
        if self.is_dotatt:
            string = ('%s_numl%s_kz%s_hierdotatt_visualization%s') % \
                     (self.encoder_name, self.num_layers, self.kernel_size, self.keep_prob)
            self.trained_model = 'trained_models/' + string + '.pth'
        else:
            string = ('%s_numl%s_kz%s_hiernnatt_keepprob%s') % \
                     (self.encoder_name, self.num_layers, self.kernel_size, self.keep_prob)
            self.trained_model = 'trained_models/' + string + '.pth'
        self.log_file = 'logs/' + self.trained_model.split('/')[1] + '.txt'
        self.result_file = 'results/captions_val2014_' + string + '_results.json'
        self.annfile = None
        self.weight_decay = 0.0


HIERATT = True

if HIERATT:
    config = HierConfig()
    model = NewCNNPlusCNNHierAtt(config)
    model.inference(is_loading_model=True, bw=3)
else:
    config = Config()
    model = CNNPlusCNN2(config)
    model.inference(is_loading_model=True, bw=3)