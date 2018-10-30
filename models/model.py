import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from data.ak_data_loader import get_data_loader
from data.ak_data_loader import get_data_loader as ak_get_data_loader
from torch.autograd import Variable
import pickle as pkl
import json
import numpy as np
import os
from coco_eval.demo import scores
from PIL import Image
import os
import cv2


class FullVgg162(nn.Module):
    def __init__(self):
        super(FullVgg162, self).__init__()
        self.block0 = nn.Sequential(*list(list(vgg16.children())[0])[:24])
        self.block1 = nn.Sequential(*list(list(vgg162.children())[0]))

    def forward(self, x):
        y1 = self.block0(x)  # 14 x 14
        y2 = self.block1(x)  # 7 x 7
        return y1, y2


vgg16 = tv.models.vgg16(pretrained=True)
vgg162 = tv.models.vgg16(pretrained=True)
vgg16_features = nn.Sequential(*list(list(vgg16.children())[0]))
resnet101 = tv.models.resnet101(pretrained=True)
resnet101_features = nn.Sequential(*list(resnet101.children())[:-2])
resnet101_conv4x = nn.Sequential(*list(resnet101.children())[:7])
resnet101_conv3x = nn.Sequential(*list(resnet101.children())[:6])
resnet101_conv3x.add_module(name='lastpool', module=nn.AvgPool2d(2))
resnet152 = tv.models.resnet152(pretrained=True)
resnet152_features = nn.Sequential(*list(resnet152.children())[:-2])


def get_image_path(image_dir):
    image_dict_list = []
    image_list = os.listdir(image_dir)
    for image in image_list:
        image_dict = {}
        image_id = int(image.split('_')[-1].split('.')[0])
        image_dict['image_id'] = image_id
        image_dict['image_path'] = os.path.join(image_dir, image)
        image_dict_list.append(image_dict)

    print('number of images: %d')%(len(image_dict_list))
    return image_dict_list


class FullVgg16(nn.Module):
    def __init__(self):
        super(FullVgg16, self).__init__()
        self.block0 = nn.Sequential(*list(list(vgg16.children())[0])[:24])
        self.block1 = nn.Sequential(*list(list(vgg16.children())[0])[24:])

    def forward(self, x):
        y1 = self.block0(x)  # 14 x 14
        y2 = self.block1(y1)  # 7 x 7
        return y1, y2


class FullResNet101(nn.Module):
    def __init__(self, is_transfer=False):
        super(FullResNet101, self).__init__()
        self.block0 = nn.Sequential(*list(resnet101.children())[0:6])  # 28x28
        self.block1 = nn.Sequential(*list(resnet101.children())[6])  # 14x14
        self.block2 = nn.Sequential(*list(resnet101.children())[7])  # 7x7
        self.avg_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self._is_transfer = is_transfer
        if is_transfer:
            self.conv1 = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1)
            self.conv2 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1)

    def forward(self, x):
        y1 = self.block0(x)
        y1_out = self.avg_pooling(y1)  # 14 x 14
        y2 = self.block1(y1)
        y3 = self.block2(y2)
        if self._is_transfer:
            y1_out = self.conv1(y1_out)
            y2 = self.conv2(y2)
        return y1_out, y2, y3


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=0,
                                           dilation=dilation,
                                           groups=groups,
                                           bias=bias)
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, input):
        x = F.pad(input.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        y = super(CausalConv1d, self).forward(x)
        return y


class Embedding(nn.Embedding):
    def __init__(self, num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2,
                 scale_grad_by_freq=False,
                 sparse=False,
                 _weight=None):
        super(Embedding, self).__init__(num_embeddings=num_embeddings,
                                        embedding_dim=embedding_dim,
                                        padding_idx=padding_idx,
                                        max_norm=max_norm,
                                        norm_type=norm_type,
                                        scale_grad_by_freq=scale_grad_by_freq,
                                        sparse=sparse)

    def forward(self, input):
        y = super(Embedding, self).forward(input)  # (b, l, c)
        new_y = y.transpose(1, 2)  # (b, c, l)
        return new_y


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, inside_f=True, kernel_size=3):
        super(Bottleneck, self).__init__()
        self.glu = nn.GLU(1)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=2*in_channels, kernel_size=1, bias=False)
        self.conv2 = CausalConv1d(in_channels=in_channels, out_channels=2*in_channels,
                                  kernel_size=kernel_size, bias=False)
        self.inside_f = inside_f
        if self.inside_f:
            self.conv3 = nn.Conv1d(in_channels=in_channels, out_channels=2*out_channels, kernel_size=1, bias=False)
            self.downsample = nn.Conv1d(in_channels=in_channels, out_channels=2*out_channels, kernel_size=1, bias=False)
        else:
            self.conv3 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.glu(self.conv1(x))
        out = self.glu(self.conv2.forward(out))
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.inside_f:
            out = self.glu(out)
        return out  # (b, l, out_channel)


# probably have some problems 08/04/18
class Attention(nn.Module):
    # one hidden layer nn
    # takes conception and visual representation as inputs
    def __init__(self, visual_channel, conception_channel, num_hidden=1024):
        super(Attention, self).__init__()
        self.vis_conv = nn.Conv2d(in_channels=visual_channel, out_channels=num_hidden, kernel_size=1)
        self.lang_conv = nn.Conv2d(in_channels=conception_channel, out_channels=num_hidden, kernel_size=1)
        self.score_conv = nn.Conv2d(in_channels=num_hidden, out_channels=1, kernel_size=1)
        self.f = nn.ELU()
        self.softmax = nn.Softmax(dim=-1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, vis_map, con_map, return_weights=False):
        """
        :param vis_map: (batch_size, vis_channel, n, n)
        :param con_map: (batch_size, con_channel, l)
        :return:
        """
        batch_size, vis_channel, n, _ = vis_map.shape
        _, con_channel, l = con_map.shape
        vis_map = vis_map.view(batch_size, vis_channel, n*n)
        vis_map = vis_map.unsqueeze(2).repeat(1, 1, l, 1)  # (b, cv, l, n*n)
        con_map = con_map.unsqueeze(3).repeat(1, 1, 1, n*n)  # (b, cw, l, n*n)

        h = self.vis_conv(vis_map) + self.lang_conv(con_map)
        h = self.f(h)
        scores = self.score_conv(h)  # (b, 1, l, n*n)
        scores = self.softmax(scores)
        final_score = scores
        scores = scores.repeat(1, vis_channel, 1, 1)
        att_map = torch.mul(vis_map, scores)  # (b, c, l, n*n)
        att_map = att_map.sum(dim=-1)  # (b, c, l)
        if return_weights:
            return att_map, final_score.unsqueeze(1).contiguous().view(batch_size, l, n, n)
        else:
            return att_map


class AttentionDotProduct(nn.Module):
    # scaled dot-product attention
    def __init__(self, visual_channel, concept_channel):
        super(AttentionDotProduct, self).__init__()
        self.weight = nn.Parameter()
        self.weight.data = nn.init.orthogonal(torch.FloatTensor(visual_channel, concept_channel))
        self.weight.requires_grad=True
        self._softmax = nn.Softmax(dim=-1)

    def forward(self, vis_map, concepts, return_weights=False):
        """
        :param vis_map: (b, c1, l)
        :param concepts: (b, c2, n, n)
        :param return_weights
        :return:
        """
        concepts = concepts.transpose(1, 2)  # (b, l, c1)
        b, l, c1 = concepts.shape
        _, c2, n, m = vis_map.shape
        concepts = concepts.contiguous().view(-1, c1)  # (b*l, c1)
        vis_map_new = vis_map.transpose(1,2).transpose(2,3).contiguous().view(-1, c2)  # (b*n*n, c2)
        vis_map_new = torch.matmul(vis_map_new, self.weight)  # (b*n*n, c1)
        scores = torch.matmul(concepts, vis_map_new.transpose(0, 1))  # (b*l, b*n*n)
        scores = scores.contiguous().view(b, l, b, n*n)
        final_score = []
        for i in range(b):
            final_score.append(scores[i:(i+1), :, i:(i+1), :])  # (1, l, 1, n*n)
        if return_weights:
            return_scores = -torch.cat(final_score, dim=0)
            return_scores = self._softmax(return_scores)
        final_score = torch.cat(final_score, dim=0) / np.sqrt(c1)  # (b, l, 1, n*n)
        final_score = self._softmax(final_score)  # (b, l, 1, n*n)
        vis_map = vis_map.contiguous().view(b, c2, -1).unsqueeze(1)  # (b, 1, c2, n*n)
        att_map = torch.mul(vis_map.repeat(1, l, 1, 1), final_score.repeat(1, 1, c2, 1))  # (b, l, c2, n*n)
        att_map = att_map.sum(-1).transpose(1, 2)  # (b, c2, l)
        if return_weights:
            return att_map, return_scores.unsqueeze(2).contiguous().view(b, l, n, n)
        else:
            return att_map


class AttentionDotProduct2(nn.Module):
    # scaled dot-product attention
    def __init__(self, visual_channel, concept_channel):
        super(AttentionDotProduct2, self).__init__()
        self.conv = nn.Conv1d(in_channels=concept_channel, out_channels=visual_channel, kernel_size=1)
        self._softmax = nn.Softmax(dim=-1)

    def forward(self, vis_map, concepts, return_weights=False):
        """
        :param vis_map: (b, c1, l)
        :param concepts: (b, c2, n, n)
        :param return_weights
        :return:
        """
        b, c1, l = concepts.shape
        concepts = self.conv(concepts)  # (b, c2, l)
        _, c2, n, m = vis_map.shape
        concepts = concepts.transpose(1, 2).contiguous().view(-1, c2)  # (b*l, c2)
        vis_map_new = vis_map.transpose(1,2).transpose(2,3).contiguous().view(-1, c2)  # (b*n*n, c2)
        scores = torch.matmul(concepts, vis_map_new.transpose(0, 1))  # (b*l, b*n*n)
        scores = scores.contiguous().view(b, l, b, n*n)
        final_score = []
        for i in range(b):
            final_score.append(scores[i:(i+1), :, i:(i+1), :])  # (1, l, 1, n*n)
        if return_weights:
            return_score = torch.cat(final_score, dim=0)
            return_score = self._softmax(return_score)
        final_score = torch.cat(final_score, dim=0) / np.sqrt(c1)  # (b, l, 1, n*n)
        final_score = self._softmax(final_score)  # (b, l, 1, n*n)
        vis_map = vis_map.contiguous().view(b, c2, -1).unsqueeze(1)  # (b, 1, c2, n*n)
        att_map = torch.mul(vis_map.repeat(1, l, 1, 1), final_score.repeat(1, 1, c2, 1))  # (b, l, c2, n*n)
        att_map = att_map.sum(-1).transpose(1, 2)  # (b, c2, l)
        if return_weights:
            return att_map, return_score.unsqueeze(2).contiguous().view(b, l, n, n)
        else:
            return att_map


class Prediction(nn.Module):
    def __init__(self, vis_channel, con_channel, voc_size, keep_prob=0.5, num_hidden=1024):
        super(Prediction, self).__init__()
        self.vis_conv = nn.Conv1d(in_channels=vis_channel, out_channels=num_hidden, kernel_size=1)
        self.con_conv = nn.Conv1d(in_channels=con_channel, out_channels=num_hidden, kernel_size=1)
        self.conv_h = nn.Sequential(
            nn.ELU(),
            nn.Dropout(1-keep_prob),
            nn.Conv1d(in_channels=num_hidden, out_channels=num_hidden, kernel_size=1),
            nn.ELU(),
            nn.Dropout(1-keep_prob),
            nn.Conv1d(in_channels=num_hidden, out_channels=voc_size, kernel_size=1)
        )
        self.xe_f = nn.CrossEntropyLoss(reduce=False)

        self._init_weights()

        # self.emb = Embedding(num_embeddings=voc_size, embedding_dim=voc_size)
        # self.emb.weight.data = torch.eye(voc_size)
        # self.emb.weight.requires_grad = False

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()

    def loss_stable(self, logits, targets, mask):
        """
        :param logits: (b, voc_size, l)
        :param targets: (b, l)
        :param mask: (b, l)
        :return:
        """
        b, voc_size, l = logits.shape
        logits = logits.transpose(1,2).contiguous()  # (b, l, voc_size)
        logits = logits.contiguous().view(-1, voc_size)
        targets = targets.contiguous().view(-1)
        # mask_new = mask.view(-1)
        xe = self.xe_f(logits, targets).view(b, l)
        xe = torch.sum(torch.mul(xe, mask).sum(1) / mask.sum(1)) / b

        return xe

    def forward(self, vision, conception):
        h = self.vis_conv(vision) + self.con_conv(conception)
        logits = self.conv_h(h)
        return logits


class Prediction2(nn.Module):
    def __init__(self, vis_channel, con_channel, voc_size, num_hidden=1024):
        super(Prediction2, self).__init__()
        self.vis_conv = nn.Conv1d(in_channels=vis_channel, out_channels=num_hidden, kernel_size=1)
        self.con_conv = nn.Conv1d(in_channels=con_channel, out_channels=num_hidden, kernel_size=1)
        self.conv_h = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(in_channels=num_hidden, out_channels=voc_size, kernel_size=1)
        )
        self.xe_f = nn.CrossEntropyLoss(reduce=False)

        self._init_weights()

        # self.emb = Embedding(num_embeddings=voc_size, embedding_dim=voc_size)
        # self.emb.weight.data = torch.eye(voc_size)
        # self.emb.weight.requires_grad = False

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()

    def loss_stable(self, logits, targets, mask):
        """
        :param logits: (b, voc_size, l)
        :param targets: (b, l)
        :param mask: (b, l)
        :return:
        """
        b, voc_size, l = logits.shape
        logits = logits.transpose(1,2).contiguous()  # (b, l, voc_size)
        logits = logits.contiguous().view(-1, voc_size)
        targets = targets.contiguous().view(-1)
        # mask_new = mask.view(-1)
        xe = self.xe_f(logits, targets).view(b, l)
        xe = torch.sum(torch.mul(xe, mask).sum(1) / mask.sum(1)) / b

        return xe

    def forward(self, vision, conception):
        h = self.vis_conv(vision) + self.con_conv(conception)
        logits = self.conv_h(h)
        return logits


class CNNPlusCNN(nn.Module):
    def __init__(self, config):
        super(CNNPlusCNN, self).__init__()
        self._max_epoch = config.max_epoch
        self._batch_size = config.batch_size
        self._encoder_name = config.encoder_name
        self._kernel_size = config.kernel_size
        self._num_layers = config.num_layers
        self._channels = config.channels
        self._prediction_dim = config.prediction_dim
        self._voc_size = config.voc_size
        self._image_dir = config.image_dir
        self._train_ann_file = config.train_ann_file
        self._val_ann_file = config.val_ann_file
        self._split_file = config.split_file
        self._vocab_file = config.vocab_file
        self._trained_model = config.trained_model
        self._log_file = config.log_file
        self._image_width = config.width
        self._image_height = config.height
        self._is_gpu = config.is_gpu
        self._is_train = config.is_train
        self._shuffle = config.shuffle
        self._num_workers = config.num_workers
        self._transformer = config.transformer
        self._is_dotatt = config.is_dotatt
        self._result_file = config.result_file
        self._annfile = config.annfile
        self._keep_prob = config.keep_prob

        self._nonlinear_f = config.f
        self._blocks = nn.Sequential()
        self._modules_list = nn.ModuleList()

        self._build()

    def _build(self):
        # encoder
        if self._encoder_name == 'vgg16':
            encoder = vgg16_features
            if self._is_dotatt:
                att = AttentionDotProduct(visual_channel=512, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=512, conception_channel=self._channels)
            pred = Prediction(vis_channel=512, con_channel=self._channels, keep_prob=self._keep_prob,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim)
        elif self._encoder_name == 'resnet152':
            encoder = resnet152_features
            if self._is_dotatt:
                att = AttentionDotProduct(visual_channel=2048, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=2048, conception_channel=self._channels)
            pred = Prediction(vis_channel=2048, con_channel=self._channels, keep_prob=self._keep_prob,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim)
        else:
            encoder = resnet101_features
            if self._is_dotatt:
                att = AttentionDotProduct(visual_channel=2048, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=2048, conception_channel=self._channels)
            pred = Prediction(vis_channel=2048, con_channel=self._channels, keep_prob=self._keep_prob,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim)

        self._modules_list.append(encoder)
        self._modules_list.append(att)
        self._modules_list.append(pred)
        # word embedding layer
        word_embedding = Embedding(num_embeddings=self._voc_size,
                                   embedding_dim=self._channels)
        self._blocks.add_module(name='word_embedding', module=word_embedding)
        # conv layers
        for i in range(self._num_layers):
            key = 'layer' + str(i+1)
            if isinstance(self._nonlinear_f, nn.GLU):
                conv = CausalConv1d(in_channels=self._channels,
                                    out_channels=2*self._channels,
                                    kernel_size=self._kernel_size)
            else:
                conv = CausalConv1d(in_channels=self._channels,
                                    out_channels=self._channels,
                                    kernel_size=self._kernel_size)
            self._blocks.add_module(name=key, module=conv)
            self._blocks.add_module(name=key + '_f', module=self._nonlinear_f)
        if self._is_gpu:
            self._blocks = self._blocks.cuda()
            self._modules_list = self._modules_list.cuda()

    def data_produceer(self,
                       image_dir,
                       train_ann_file,
                       val_ann_file,
                       split_file,
                       vocab_file,
                       transformer,
                       batch_size,
                       width,
                       height,
                       num_workers,
                       shuffle,
                       split_key):
        data_loader = ak_get_data_loader(
            image_dir=image_dir,
            split_file=split_file,
            split_key=split_key,
            transformer=transformer,
            width=width,
            height=height,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )
        return data_loader

    def forward(self, images, input_seq, target_seq, mask):
        image_feature_map = self._modules_list[0](images)
        conceptions = self._blocks(input_seq)
        att_map = self._modules_list[1](image_feature_map, conceptions)
        logits = self._modules_list[2](att_map, conceptions)
        if self._is_train:
            loss = self._modules_list[2].loss_stable(logits, targets=target_seq, mask=mask)
            return loss
        else:
            return logits

    def save_load_model(self, mode='save', is_debug=False):
        if mode == 'save':
            print "Saving model..."
            torch.save(self.state_dict(), self._trained_model)
        elif mode == 'load':
            print "Loading model..."
            if is_debug:
                self.load_state_dict(torch.load(self._trained_model, map_location=lambda storage, loc: storage))
            else:
                self.load_state_dict(torch.load(self._trained_model))
        else:
            raise IOError

    def _cpu_debug_load_model(self):
        state_dict = torch.load(self._trained_model, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)

    def _get_para_group(self, encoder_lr, decoder_lr):
        encoder_paras = {
            'params': [p[1] for p in self. named_parameters() if '_modules_list.0' in p[0] and p[1].requires_grad],
            'lr': encoder_lr
        }
        decoder_paras = {
            'params':[p[1] for p in self. named_parameters() if '_modules_list.0' not in p[0] and p[1].requires_grad],
            'lr': decoder_lr
        }
        print 'number of encoder parameters is: %d\n' % (len(encoder_paras['params']))
        print 'number of decoder parameters is: %d\n' % (len(decoder_paras['params']))
        return [encoder_paras, decoder_paras]

    def trainer(self):
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key=['train', 'val'],
            transformer=self._transformer,
            batch_size=self._batch_size,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=self._shuffle
        )
        try:
            self.save_load_model('load')
        except:
            print 'The model is randomly initialized......'
            pass
        param_group = self._get_para_group(encoder_lr=1e-5, decoder_lr=3e-4)
        optimizer = torch.optim.Adam(param_group)
        hostname = os.popen('hostname').read()
        fo = open(self._log_file, 'a+')
        fo.write(hostname)
        fo.close()
        if not os.path.exists(self._result_file):
            self.inference(result_file=self._result_file,
                           bw=3)
        eval_scores = scores(res_file=self._result_file, ann_file=self._annfile)
        buf = 'cider=%.5f\n' % (eval_scores['CIDEr'])
        fo = open(self._log_file, 'a+')
        fo.write(buf)
        fo.close()
        cider_scores = [eval_scores['CIDEr']]

        for epoch in range(1, self._max_epoch + 1):
            self.train()
            for i, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
                optimizer.zero_grad()
                if self._is_gpu:
                    loss = self.forward(Variable(images).cuda(),
                                        Variable(inputs).cuda(),
                                        Variable(targets).cuda(),
                                        Variable(mask).cuda())
                else:
                    loss = self.forward(Variable(images),
                                        Variable(inputs),
                                        Variable(targets),
                                        Variable(mask))
                loss.backward()
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, self.parameters()), max_norm=5.0)
                optimizer.step()
                if i % 1000 == 0:
                    buf = 'epoch=%d, batch#=%d, loss=%.3f\n' % (epoch, i, loss.data.cpu().numpy()[0])
                    fo = open(self._log_file, 'a+')
                    fo.write(buf)
                    fo.close()

            previous_cider = max(cider_scores)
            self.inference(result_file=self._result_file,
                           bw=3)
            eval_scores = scores(res_file=self._result_file, ann_file=self._annfile)
            buf = 'cider=%.5f\n' % (eval_scores['CIDEr'])
            fo = open(self._log_file, 'a+')
            fo.write(buf)
            fo.close()
            cider_scores.append(eval_scores['CIDEr'])

            if eval_scores['CIDEr'] > previous_cider:
                self.save_load_model(mode='save')

    def inference(self, result_file='results.json', bw=3, max_len=20, is_debug=False, is_loading_model=False):
        if is_loading_model:
            self.save_load_model(mode='load', is_debug=is_debug)
        self.eval()
        self._is_train = False
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key='test',
            transformer=None,
            batch_size=1,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=False)
        voc = pkl.load(open(self._vocab_file))  # {'token2id':{}, 'id2token': {}}
        id2token = voc['id2token']
        token2id = voc['token2id']
        im_ids = []
        captions = []
        for num, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
            image_id = image_ids.numpy()[0, 0]
            if image_id not in im_ids:
                im_ids.append(image_id)
                cap = ' '
                cap_dict = {}
                s, sp = self.beam_search(image=images, voc=voc, is_debug=is_debug, bw=bw)
                b, l = s[0].shape
                w_list = []
                for i in range(l):
                    w = id2token[s[0][0, i].data.cpu().numpy()[0]]
                    if w != '<start>' and w != '<end>':
                        w_list.append(w)
                cap = cap.join(w_list)
                if bw > 2:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\nsec_prob=%.8f\nthird_prob=%.8f\n')%\
                          (num, image_id, cap, sp[0], sp[1], sp[2])
                else:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\n') % (num, image_id, cap, sp[0])
                cap_dict['caption'] = cap
                cap_dict['image_id'] = int(image_id)
                captions.append(cap_dict)
            else:
                continue
        with open(result_file, 'w') as f:
            json.dump(captions, f)
        self._is_train = True

    def multiple_inference(self, result_file='results.json', bw=3, max_len=20, is_debug=False, is_loading_model=False):
        if is_loading_model:
            self.save_load_model(mode='load')
        self.eval()
        self._is_train = False
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key='test',
            transformer=self._transformer,
            batch_size=1,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=False)
        voc = pkl.load(open(self._vocab_file))  # {'token2id':{}, 'id2token': {}}
        id2token = voc['id2token']
        token2id = voc['token2id']
        im_ids = []
        captions = []
        for num, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
            image_id = image_ids.numpy()[0, 0]
            if image_id not in im_ids:
                im_ids.append(image_id)
            cap = ' '
            cap_dict = {}
            s, sp = self.beam_search(image=images, voc=voc, is_debug=is_debug, bw=bw)
            b, l = s[0].shape
            w_list = []
            for i in range(l):
                w = id2token[s[0][0, i].data.cpu().numpy()[0]]
                if w != '<start>' and w != '<end>':
                    w_list.append(w)
            cap = cap.join(w_list)
            if bw > 2:
                print ('num=%d  image_id=%d  caption=%s  probs=%.8f\nsec_prob=%.8f\nthird_prob=%.8f\n')%\
                        (num, image_id, cap, sp[0], sp[1], sp[2])
            else:
                print ('num=%d  image_id=%d  caption=%s  probs=%.8f\n') % (num, image_id, cap, sp[0])
            cap_dict['caption'] = cap
            cap_dict['image_id'] = int(image_id)
            captions.append(cap_dict)

        with open(result_file, 'w') as f:
            json.dump(captions, f)
        self._is_train = True

    def visualization_forward(self, images, input_seq):
        image_feature_map = self._modules_list[0](images)
        conceptions = self._blocks(input_seq)
        v, weights = self._modules_list[1](image_feature_map, conceptions, True)
        return weights

    def visualization_inference(self, result_file='results.json', dir='visualization', bw=3, max_len=20, is_debug=False,
                                is_loading_model=False):
        if is_loading_model:
            self.save_load_model(mode='load', is_debug=is_debug)
        self.eval()
        self._is_train = False
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key='test',
            transformer=None,
            batch_size=1,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=False)
        voc = pkl.load(open(self._vocab_file))  # {'token2id':{}, 'id2token': {}}
        id2token = voc['id2token']
        token2id = voc['token2id']
        im_ids = []
        captions = []
        for num, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
            image_id = image_ids.numpy()[0, 0]
            if (image_id not in im_ids) and (len(im_ids) < 100):
                im_ids.append(image_id)
                cap = ' '
                cap_dict = {}
                s, sp = self.beam_search(image=images, voc=voc, is_debug=is_debug, bw=bw)
                b, l = s[0].shape
                w_list = []
                for i in range(l):
                    w = id2token[s[0][0, i].data.cpu().numpy()[0]]
                    if w != '<start>' and w != '<end>':
                        w_list.append(w)
                cap = cap.join(w_list)
                # visualization
                if self._is_gpu:
                    att_weights = self.visualization_forward(Variable(images).cuda(), s[0][:, :-1])
                    weights = att_weights.data.cpu().numpy()
                else:
                    att_weights = self.visualization_forward(Variable(images), s[0][:, :-1])
                    weights = att_weights.data.cpu().numpy()
                if not os.path.exists(os.path.join(dir, 'CNNPlusCNN', str(image_id))):
                    os.mkdir(os.path.join(dir, 'CNNPlusCNN', str(image_id)))
                for j in range(l-1):
                    img_str = '%012d' % image_id
                    img_file = 'COCO_val2014_' + img_str + '.jpg'
                    img = cv2.imread(os.path.join(self._image_dir, img_file))
                    img = cv2.resize(img, (224, 224))
                    heatmap = cv2.resize(weights[0,j, :, :], (224, 224))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
                    cv2.imwrite(os.path.join(dir, 'CNNPlusCNN', str(image_id), 'original.jpg'), img)
                    cv2.imwrite(os.path.join(dir, 'CNNPlusCNN', str(image_id), str(j)+'.jpg'), superimposed_img)

                if bw > 2:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\nsec_prob=%.8f\nthird_prob=%.8f\n')%\
                          (num, image_id, cap, sp[0], sp[1], sp[2])
                else:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\n') % (num, image_id, cap, sp[0])
                cap_dict['caption'] = cap
                cap_dict['image_id'] = int(image_id)
                captions.append(cap_dict)
            else:
                continue
        with open(os.path.join(dir, 'CNNPlusCNN', result_file), 'w') as f:
            json.dump(captions, f)
        self._is_train = True

    def beam_search(self, image, voc, bw=3, max_len=20, softmax=nn.Softmax(dim=-1), is_debug=False):
        """
        :param image: (1, 3, 448, 448)
        :param bw:
        :return:
        """
        token2id = voc['token2id']
        if is_debug:
            start_s = Variable(torch.LongTensor([[token2id['<start>']]]))
            image = Variable(image)
        else:
            start_s = Variable(torch.LongTensor([[token2id['<start>']]])).cuda()
            image = Variable(image).cuda()
        sentence_probs = []
        sentences = []
        for k in range(bw):
            sentences.append(start_s)
            sentence_probs.append(1.0)

        for l in range(max_len):
            temp_probs = []
            temp_s = []
            stop_flag = 0
            for j in range(bw):
                sentence = sentences[j]
                if sentence[0, -1].data.cpu().numpy()[0] == token2id['<end>']:
                    temp_probs.append(sentence_probs[j])
                    temp_s.append(sentence)
                    stop_flag += 1
                    continue
                logits = self.forward(images=image, input_seq=sentence, target_seq=None, mask=None)  # (1, voc_size, l)
                pred = logits[:, :, -1]  # (1, voc_size)
                probs = softmax(pred)
                value, index = torch.topk(probs.view(-1), k=bw)
                if l == 0:
                    for k in range(bw):
                        current_w = index[k]  # Variable
                        sentences[k] = torch.cat([sentences[k], current_w.view(1, 1)], dim=1)
                        sentence_probs[k] = sentence_probs[k] * value[k].data.cpu().numpy()[0]
                    break
                else:
                    for k in range(bw):
                        current_w = index[k]  # Variable
                        temp_s.append(torch.cat([sentence, current_w.view(1, 1)], dim=1))
                        temp_probs.append(sentence_probs[j] * value[k].data.cpu().numpy()[0])
            if l != 0:
                value, index = torch.FloatTensor(temp_probs).topk(bw)
                for k in range(bw):
                    sentences[k] = temp_s[index[k]]
                    sentence_probs[k] = value[k]
            if stop_flag == bw:
                break
        return sentences, sentence_probs


class CNNPlusCNN2(nn.Module):
    def __init__(self, config):
        super(CNNPlusCNN2, self).__init__()
        self._max_epoch = config.max_epoch
        self._batch_size = config.batch_size
        self._encoder_name = config.encoder_name
        self._kernel_size = config.kernel_size
        self._num_layers = config.num_layers
        self._channels = config.channels
        self._prediction_dim = config.prediction_dim
        self._voc_size = config.voc_size
        self._image_dir = config.image_dir
        self._train_ann_file = config.train_ann_file
        self._val_ann_file = config.val_ann_file
        self._split_file = config.split_file
        self._vocab_file = config.vocab_file
        self._trained_model = config.trained_model
        self._log_file = config.log_file
        self._image_width = config.width
        self._image_height = config.height
        self._is_gpu = config.is_gpu
        self._is_train = config.is_train
        self._shuffle = config.shuffle
        self._num_workers = config.num_workers
        self._transformer = config.transformer
        self._is_dotatt = config.is_dotatt
        self._result_file = config.result_file
        self._annfile = config.annfile
        self._keep_prob = config.keep_prob

        self._nonlinear_f = config.f
        self._blocks = nn.Sequential()
        self._modules_list = nn.ModuleList()

        self._build()

    def _build(self):
        # encoder
        if self._encoder_name == 'vgg16':
            encoder = vgg16_features
            if self._is_dotatt:
                att = AttentionDotProduct2(visual_channel=512, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=512, conception_channel=self._channels)
            pred = Prediction2(vis_channel=512, con_channel=self._channels,
                               voc_size=self._voc_size, num_hidden=self._prediction_dim)
        elif self._encoder_name == 'resnet152':
            encoder = resnet152_features
            if self._is_dotatt:
                att = AttentionDotProduct2(visual_channel=2048, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=2048, conception_channel=self._channels)
            pred = Prediction2(vis_channel=2048, con_channel=self._channels,
                               voc_size=self._voc_size, num_hidden=self._prediction_dim)
        else:
            encoder = resnet101_features
            if self._is_dotatt:
                att = AttentionDotProduct2(visual_channel=2048, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=2048, conception_channel=self._channels)
            pred = Prediction2(vis_channel=2048, con_channel=self._channels,
                               voc_size=self._voc_size, num_hidden=self._prediction_dim)

        self._modules_list.append(encoder)
        self._modules_list.append(att)
        self._modules_list.append(pred)
        # word embedding layer
        word_embedding = Embedding(num_embeddings=self._voc_size,
                                   embedding_dim=self._channels)
        self._blocks.add_module(name='word_embedding', module=word_embedding)
        # conv layers
        for i in range(self._num_layers):
            key = 'layer' + str(i+1)
            if isinstance(self._nonlinear_f, nn.GLU):
                conv = CausalConv1d(in_channels=self._channels,
                                    out_channels=2*self._channels,
                                    kernel_size=self._kernel_size)
            else:
                conv = CausalConv1d(in_channels=self._channels,
                                    out_channels=self._channels,
                                    kernel_size=self._kernel_size)
            self._blocks.add_module(name=key, module=conv)
            self._blocks.add_module(name=key + '_f', module=self._nonlinear_f)
        if self._is_gpu:
            self._blocks = self._blocks.cuda()
            self._modules_list = self._modules_list.cuda()

    def data_produceer(self,
                       image_dir,
                       train_ann_file,
                       val_ann_file,
                       split_file,
                       vocab_file,
                       transformer,
                       batch_size,
                       width,
                       height,
                       num_workers,
                       shuffle,
                       split_key):
        data_loader = ak_get_data_loader(
            image_dir=image_dir,
            split_file=split_file,
            split_key=split_key,
            transformer=transformer,
            width=width,
            height=height,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )
        return data_loader

    def forward(self, images, input_seq, target_seq, mask):
        image_feature_map = self._modules_list[0](images)
        conceptions = self._blocks(input_seq)
        att_map = self._modules_list[1](image_feature_map, conceptions)
        logits = self._modules_list[2](att_map, conceptions)
        if self._is_train:
            loss = self._modules_list[2].loss_stable(logits, targets=target_seq, mask=mask)
            return loss
        else:
            return logits

    def save_load_model(self, mode='save', is_debug=False):
        if mode == 'save':
            print "Saving model..."
            torch.save(self.state_dict(), self._trained_model)
        elif mode == 'load':
            print "Loading model..."
            if is_debug:
                self.load_state_dict(torch.load(self._trained_model, map_location=lambda storage, loc: storage))
            else:
                self.load_state_dict(torch.load(self._trained_model))
        else:
            raise IOError

    def _cpu_debug_load_model(self):
        state_dict = torch.load(self._trained_model, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)

    def _get_para_group(self, encoder_lr, decoder_lr):
        encoder_paras = {
            'params': [p[1] for p in self. named_parameters() if '_modules_list.0' in p[0] and p[1].requires_grad],
            'lr': encoder_lr
        }
        decoder_paras = {
            'params':[p[1] for p in self. named_parameters() if '_modules_list.0' not in p[0] and p[1].requires_grad],
            'lr': decoder_lr
        }
        print 'number of encoder parameters is: %d\n' % (len(encoder_paras['params']))
        print 'number of decoder parameters is: %d\n' % (len(decoder_paras['params']))
        return [encoder_paras, decoder_paras]

    def trainer(self):
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key=['train', 'val'],
            transformer=self._transformer,
            batch_size=self._batch_size,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=self._shuffle
        )
        try:
            self.save_load_model('load')
        except:
            print 'The model is randomly initialized......'
            pass
        param_group = self._get_para_group(encoder_lr=1e-5, decoder_lr=3e-4)
        optimizer = torch.optim.Adam(param_group)
        hostname = os.popen('hostname').read()
        fo = open(self._log_file, 'a+')
        fo.write(hostname)
        fo.close()
        # if not os.path.exists(self._result_file):
        #     self.inference(result_file=self._result_file,
        #                    bw=3)
        # eval_scores = scores(res_file=self._result_file, ann_file=self._annfile)
        eval_scores={}
        eval_scores['CIDEr'] = 0
        buf = 'cider=%.5f\n' % (eval_scores['CIDEr'])
        fo = open(self._log_file, 'a+')
        fo.write(buf)
        fo.close()
        cider_scores = [eval_scores['CIDEr']]

        for epoch in range(1, self._max_epoch + 1):
            self.train()
            for i, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
                optimizer.zero_grad()
                if self._is_gpu:
                    loss = self.forward(Variable(images).cuda(),
                                        Variable(inputs).cuda(),
                                        Variable(targets).cuda(),
                                        Variable(mask).cuda())
                else:
                    loss = self.forward(Variable(images),
                                        Variable(inputs),
                                        Variable(targets),
                                        Variable(mask))
                loss.backward()
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, self.parameters()), max_norm=5.0)
                optimizer.step()
                if i % 1000 == 0:
                    buf = 'epoch=%d, batch#=%d, loss=%.3f\n' % (epoch, i, loss.data.cpu().numpy()[0])
                    fo = open(self._log_file, 'a+')
                    fo.write(buf)
                    fo.close()

            previous_cider = max(cider_scores)
            self.inference(result_file=self._result_file,
                           bw=3)
            eval_scores = scores(res_file=self._result_file, ann_file=self._annfile)
            buf = 'cider=%.5f\n' % (eval_scores['CIDEr'])
            fo = open(self._log_file, 'a+')
            fo.write(buf)
            fo.close()
            cider_scores.append(eval_scores['CIDEr'])

            if eval_scores['CIDEr'] > previous_cider:
                self.save_load_model(mode='save')

    def inference(self, result_file='results.json', bw=3, max_len=20, is_debug=False, is_loading_model=False):
        if is_loading_model:
            self.save_load_model(mode='load', is_debug=is_debug)
        self.eval()
        self._is_train = False
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key='test',
            transformer=None,
            batch_size=1,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=False)
        voc = pkl.load(open(self._vocab_file))  # {'token2id':{}, 'id2token': {}}
        id2token = voc['id2token']
        token2id = voc['token2id']
        im_ids = []
        captions = []
        for num, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
            image_id = image_ids.numpy()[0, 0]
            if image_id not in im_ids:
                im_ids.append(image_id)
                cap = ' '
                cap_dict = {}
                s, sp = self.beam_search(image=images, voc=voc, is_debug=is_debug, bw=bw)
                b, l = s[0].shape
                w_list = []
                for i in range(l):
                    w = id2token[s[0][0, i].data.cpu().numpy()[0]]
                    if w != '<start>' and w != '<end>':
                        w_list.append(w)
                cap = cap.join(w_list)
                if bw > 2:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\nsec_prob=%.8f\nthird_prob=%.8f\n')%\
                          (num, image_id, cap, sp[0], sp[1], sp[2])
                else:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\n') % (num, image_id, cap, sp[0])
                cap_dict['caption'] = cap
                cap_dict['image_id'] = int(image_id)
                captions.append(cap_dict)
            else:
                continue
        with open(result_file, 'w') as f:
            json.dump(captions, f)
        self._is_train = True

    def multiple_inference(self, result_file='results.json', bw=3, max_len=20, is_debug=False, is_loading_model=False):
        if is_loading_model:
            self.save_load_model(mode='load')
        self.eval()
        self._is_train = False
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key='test',
            transformer=self._transformer,
            batch_size=1,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=False)
        voc = pkl.load(open(self._vocab_file))  # {'token2id':{}, 'id2token': {}}
        id2token = voc['id2token']
        token2id = voc['token2id']
        im_ids = []
        captions = []
        for num, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
            image_id = image_ids.numpy()[0, 0]
            if image_id not in im_ids:
                im_ids.append(image_id)
            cap = ' '
            cap_dict = {}
            s, sp = self.beam_search(image=images, voc=voc, is_debug=is_debug, bw=bw)
            b, l = s[0].shape
            w_list = []
            for i in range(l):
                w = id2token[s[0][0, i].data.cpu().numpy()[0]]
                if w != '<start>' and w != '<end>':
                    w_list.append(w)
            cap = cap.join(w_list)
            if bw > 2:
                print ('num=%d  image_id=%d  caption=%s  probs=%.8f\nsec_prob=%.8f\nthird_prob=%.8f\n')%\
                        (num, image_id, cap, sp[0], sp[1], sp[2])
            else:
                print ('num=%d  image_id=%d  caption=%s  probs=%.8f\n') % (num, image_id, cap, sp[0])
            cap_dict['caption'] = cap
            cap_dict['image_id'] = int(image_id)
            captions.append(cap_dict)

        with open(result_file, 'w') as f:
            json.dump(captions, f)
        self._is_train = True

    def visualization_forward(self, images, input_seq):
        image_feature_map = self._modules_list[0](images)
        conceptions = self._blocks(input_seq)
        v, weights = self._modules_list[1](image_feature_map, conceptions, True)
        return weights

    def visualization_inference(self, result_file='results.json', dir='visualization', bw=3, max_len=20, is_debug=False,
                                is_loading_model=False):
        if is_loading_model:
            self.save_load_model(mode='load', is_debug=is_debug)
        self.eval()
        self._is_train = False
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key='test',
            transformer=None,
            batch_size=1,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=False)
        voc = pkl.load(open(self._vocab_file))  # {'token2id':{}, 'id2token': {}}
        id2token = voc['id2token']
        token2id = voc['token2id']
        im_ids = []
        captions = []
        for num, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
            image_id = image_ids.numpy()[0, 0]
            if (image_id not in im_ids) and (len(im_ids) < 50):
                im_ids.append(image_id)
                cap = ' '
                cap_dict = {}
                s, sp = self.beam_search(image=images, voc=voc, is_debug=is_debug, bw=bw)
                b, l = s[0].shape
                w_list = []
                for i in range(l):
                    w = id2token[s[0][0, i].data.cpu().numpy()[0]]
                    if w != '<start>' and w != '<end>':
                        w_list.append(w)
                cap = cap.join(w_list)
                # visualization
                if self._is_gpu:
                    att_weights = self.visualization_forward(Variable(images).cuda(), s[0][:, :-1])
                    weights = att_weights.data.cpu().numpy()
                else:
                    att_weights = self.visualization_forward(Variable(images), s[0][:, :-1])
                    weights = att_weights.data.cpu().numpy()
                if not os.path.exists(os.path.join(dir, 'CNNPlusCNN', str(image_id))):
                    os.mkdir(os.path.join(dir, 'CNNPlusCNN', str(image_id)))
                for j in range(l-1):
                    img_str = '%012d' % image_id
                    img_file = 'COCO_val2014_' + img_str + '.jpg'
                    img = cv2.imread(os.path.join(self._image_dir, img_file))
                    img = cv2.resize(img, (224, 224))
                    heatmap = cv2.resize(weights[0,j, :, :], (224, 224), interpolation=cv2.INTER_LINEAR)
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    superimposed_img = cv2.addWeighted(img, 0.4, heatmap, 0.6, 0)
                    cv2.imwrite(os.path.join(dir, 'CNNPlusCNN', str(image_id), 'original.jpg'), img)
                    cv2.imwrite(os.path.join(dir, 'CNNPlusCNN', str(image_id), str(j)+'.jpg'), superimposed_img)

                if bw > 2:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\nsec_prob=%.8f\nthird_prob=%.8f\n')%\
                          (num, image_id, cap, sp[0], sp[1], sp[2])
                else:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\n') % (num, image_id, cap, sp[0])
                cap_dict['caption'] = cap
                cap_dict['image_id'] = int(image_id)
                captions.append(cap_dict)
            else:
                continue
        with open(os.path.join(dir, 'CNNPlusCNN', result_file), 'w') as f:
            json.dump(captions, f)
        self._is_train = True

    def beam_search(self, image, voc, bw=3, max_len=20, softmax=nn.Softmax(dim=-1), is_debug=False):
        """
        :param image: (1, 3, 448, 448)
        :param bw:
        :return:
        """
        token2id = voc['token2id']
        if is_debug:
            start_s = Variable(torch.LongTensor([[token2id['<start>']]]))
            image = Variable(image)
        else:
            start_s = Variable(torch.LongTensor([[token2id['<start>']]])).cuda()
            image = Variable(image).cuda()
        sentence_probs = []
        sentences = []
        for k in range(bw):
            sentences.append(start_s)
            sentence_probs.append(1.0)

        for l in range(max_len):
            temp_probs = []
            temp_s = []
            stop_flag = 0
            for j in range(bw):
                sentence = sentences[j]
                if sentence[0, -1].data.cpu().numpy()[0] == token2id['<end>']:
                    temp_probs.append(sentence_probs[j])
                    temp_s.append(sentence)
                    stop_flag += 1
                    continue
                logits = self.forward(images=image, input_seq=sentence, target_seq=None, mask=None)  # (1, voc_size, l)
                pred = logits[:, :, -1]  # (1, voc_size)
                probs = softmax(pred)
                value, index = torch.topk(probs.view(-1), k=bw)
                if l == 0:
                    for k in range(bw):
                        current_w = index[k]  # Variable
                        sentences[k] = torch.cat([sentences[k], current_w.view(1, 1)], dim=1)
                        sentence_probs[k] = sentence_probs[k] * value[k].data.cpu().numpy()[0]
                    break
                else:
                    for k in range(bw):
                        current_w = index[k]  # Variable
                        temp_s.append(torch.cat([sentence, current_w.view(1, 1)], dim=1))
                        temp_probs.append(sentence_probs[j] * value[k].data.cpu().numpy()[0])
            if l != 0:
                value, index = torch.FloatTensor(temp_probs).topk(bw)
                for k in range(bw):
                    sentences[k] = temp_s[index[k]]
                    sentence_probs[k] = value[k]
            if stop_flag == bw:
                break
        return sentences, sentence_probs


class CNNPlusCNNMultiScaleAtt(nn.Module):
    def __init__(self, config):
        super(CNNPlusCNNMultiScaleAtt, self).__init__()
        self._max_epoch = config.max_epoch
        self._batch_size = config.batch_size
        self._encoder_name = config.encoder_name
        self._kernel_size = config.kernel_size
        self._num_layers = config.num_layers
        self._channels = config.channels
        self._prediction_dim = config.prediction_dim
        self._voc_size = config.voc_size
        self._image_dir = config.image_dir
        self._train_ann_file = config.train_ann_file
        self._val_ann_file = config.val_ann_file
        self._split_file = config.split_file
        self._vocab_file = config.vocab_file
        self._trained_model = config.trained_model
        self._log_file = config.log_file
        self._image_width = config.width
        self._image_height = config.height
        self._is_gpu = config.is_gpu
        self._is_train = config.is_train
        self._shuffle = config.shuffle
        self._num_workers = config.num_workers
        self._transformer = config.transformer
        self._is_dotatt = config.is_dotatt
        self._result_file = config.result_file
        self._annfile = config.annfile
        self._keep_prob = config.keep_prob

        self._nonlinear_f = config.f
        self._blocks = nn.Sequential()
        self._modules_list = nn.ModuleList()

        self._build()

    def _build(self):
        # encoder
        if self._encoder_name == 'vgg16':
            encoder = vgg16_features
            if self._is_dotatt:
                att = AttentionDotProduct(visual_channel=512, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=512, conception_channel=self._channels)
            pred = Prediction(vis_channel=512, con_channel=self._channels, keep_prob=self._keep_prob,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim)
        elif self._encoder_name == 'resnet152':
            encoder = resnet152_features
            if self._is_dotatt:
                att = AttentionDotProduct(visual_channel=2048, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=2048, conception_channel=self._channels)
            pred = Prediction(vis_channel=2048, con_channel=self._channels, keep_prob=self._keep_prob,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim)
        else:
            encoder = FullResNet101()
            if self._is_dotatt:
                # attl = AttentionDotProduct(visual_channel=512, concept_channel=self._channels)
                attm = AttentionDotProduct(visual_channel=1024, concept_channel=self._channels)
                atth = AttentionDotProduct(visual_channel=2048, concept_channel=self._channels)
            else:
                # attl = Attention(visual_channel=512, conception_channel=self._channels)
                attm = Attention(visual_channel=1024, conception_channel=self._channels)
                atth = Attention(visual_channel=2048, conception_channel=self._channels)
            pred = Prediction(vis_channel=2048 + 1024, con_channel=self._channels, keep_prob=self._keep_prob,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim)

        self._modules_list.append(encoder)
        # self._modules_list.append(attl)
        self._modules_list.append(attm)
        self._modules_list.append(atth)
        self._modules_list.append(pred)
        # word embedding layer
        word_embedding = Embedding(num_embeddings=self._voc_size,
                                   embedding_dim=self._channels)
        self._blocks.add_module(name='word_embedding', module=word_embedding)
        # conv layers
        for i in range(self._num_layers):
            key = 'layer' + str(i+1)
            if isinstance(self._nonlinear_f, nn.GLU):
                conv = CausalConv1d(in_channels=self._channels,
                                    out_channels=2*self._channels,
                                    kernel_size=self._kernel_size)
            else:
                conv = CausalConv1d(in_channels=self._channels,
                                    out_channels=self._channels,
                                    kernel_size=self._kernel_size)
            self._blocks.add_module(name=key, module=conv)
            self._blocks.add_module(name=key + '_f', module=self._nonlinear_f)
        if self._is_gpu:
            self._blocks = self._blocks.cuda()
            self._modules_list = self._modules_list.cuda()

    def data_produceer(self,
                       image_dir,
                       train_ann_file,
                       val_ann_file,
                       split_file,
                       vocab_file,
                       transformer,
                       batch_size,
                       width,
                       height,
                       num_workers,
                       shuffle,
                       split_key):
        data_loader = ak_get_data_loader(
            image_dir=image_dir,
            split_file=split_file,
            split_key=split_key,
            transformer=transformer,
            width=width,
            height=height,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )
        return data_loader

    def forward(self, images, input_seq, target_seq, mask):
        image_feature_mapl, image_feature_mapm, image_feature_maph = self._modules_list[0](images)
        conceptions = self._blocks(input_seq)
        # att_mapl = self._modules_list[1](image_feature_mapl, conceptions)
        att_mapm = self._modules_list[1](image_feature_mapm, conceptions)
        att_maph = self._modules_list[2](image_feature_maph, conceptions)
        att_map = torch.cat([att_mapm, att_maph], dim=1)
        logits = self._modules_list[3](att_map, conceptions)
        if self._is_train:
            loss = self._modules_list[3].loss_stable(logits, targets=target_seq, mask=mask)
            return loss
        else:
            return logits

    def save_load_model(self, mode='save'):
        if mode == 'save':
            print "Saving model..."
            torch.save(self.state_dict(), self._trained_model)
        elif mode == 'load':
            print "Loading model..."
            self.load_state_dict(torch.load(self._trained_model))
        else:
            raise IOError

    def _cpu_debug_load_model(self):
        state_dict = torch.load(self._trained_model, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)

    def _get_para_group(self, encoder_lr, decoder_lr):
        encoder_paras = {
            'params': [p[1] for p in self. named_parameters() if '_modules_list.0' in p[0] and p[1].requires_grad],
            'lr': encoder_lr
        }
        decoder_paras = {
            'params':[p[1] for p in self. named_parameters() if '_modules_list.0' not in p[0] and p[1].requires_grad],
            'lr': decoder_lr
        }
        print 'number of encoder parameters is: %d\n' % (len(encoder_paras['params']))
        print 'number of decoder parameters is: %d\n' % (len(decoder_paras['params']))
        return [encoder_paras, decoder_paras]

    def trainer(self):
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key=['train', 'val'],
            transformer=self._transformer,
            batch_size=self._batch_size,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=self._shuffle
        )
        try:
            self.save_load_model('load')
        except:
            print 'The model is randomly initialized......'
            pass
        param_group = self._get_para_group(encoder_lr=1e-5, decoder_lr=3e-4)
        optimizer = torch.optim.Adam(param_group)
        hostname = os.popen('hostname').read()
        fo = open(self._log_file, 'a+')
        fo.write(hostname)
        fo.close()

        # self.inference(result_file=self._result_file,
        #                bw=3)
        # eval_scores = scores(res_file=self._result_file, ann_file=self._annfile)
        eval_scores = {}
        eval_scores['CIDEr'] = 0
        buf = 'cider=%.5f\n' % (eval_scores['CIDEr'])
        fo = open(self._log_file, 'a+')
        fo.write(buf)
        fo.close()
        cider_scores = [eval_scores['CIDEr']]

        for epoch in range(1, self._max_epoch + 1):
            self.train()
            for i, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
                optimizer.zero_grad()
                if self._is_gpu:
                    loss = self.forward(Variable(images).cuda(),
                                        Variable(inputs).cuda(),
                                        Variable(targets).cuda(),
                                        Variable(mask).cuda())
                else:
                    loss = self.forward(Variable(images),
                                        Variable(inputs),
                                        Variable(targets),
                                        Variable(mask))
                loss.backward()
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, self.parameters()), max_norm=5.0)
                optimizer.step()
                if i % 1000 == 0:
                    buf = 'epoch=%d, batch#=%d, loss=%.3f\n' % (epoch, i, loss.data.cpu().numpy()[0])
                    fo = open(self._log_file, 'a+')
                    fo.write(buf)
                    fo.close()

            previous_cider = max(cider_scores)
            self.inference(result_file=self._result_file,
                           bw=3)
            eval_scores = scores(res_file=self._result_file, ann_file=self._annfile)
            buf = 'cider=%.5f\n' % (eval_scores['CIDEr'])
            fo = open(self._log_file, 'a+')
            fo.write(buf)
            fo.close()
            cider_scores.append(eval_scores['CIDEr'])

            if eval_scores['CIDEr'] > previous_cider:
                self.save_load_model(mode='save')

    def inference(self, result_file='results.json', bw=3, max_len=20, is_debug=False, is_loading_model=False):
        if is_loading_model:
            self.save_load_model(mode='load')
        self.eval()
        self._is_train = False
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key='test',
            transformer=None,
            batch_size=1,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=False)
        voc = pkl.load(open(self._vocab_file))  # {'token2id':{}, 'id2token': {}}
        id2token = voc['id2token']
        token2id = voc['token2id']
        im_ids = []
        captions = []
        for num, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
            image_id = image_ids.numpy()[0, 0]
            if image_id not in im_ids:
                im_ids.append(image_id)
                cap = ' '
                cap_dict = {}
                s, sp = self.beam_search(image=images, voc=voc, is_debug=is_debug, bw=bw)
                b, l = s[0].shape
                w_list = []
                for i in range(l):
                    w = id2token[s[0][0, i].data.cpu().numpy()[0]]
                    if w != '<start>' and w != '<end>':
                        w_list.append(w)
                cap = cap.join(w_list)
                if bw > 2:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\nsec_prob=%.8f\nthird_prob=%.8f\n')%\
                          (num, image_id, cap, sp[0], sp[1], sp[2])
                else:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\n') % (num, image_id, cap, sp[0])
                cap_dict['caption'] = cap
                cap_dict['image_id'] = int(image_id)
                captions.append(cap_dict)
            else:
                continue
        with open(result_file, 'w') as f:
            json.dump(captions, f)
        self._is_train = True

    def beam_search(self, image, voc, bw=3, max_len=20, softmax=nn.Softmax(dim=-1), is_debug=False):
        """
        :param image: (1, 3, 448, 448)
        :param bw:
        :return:
        """
        token2id = voc['token2id']
        if is_debug:
            start_s = Variable(torch.LongTensor([[token2id['<start>']]]))
            image = Variable(image)
        else:
            start_s = Variable(torch.LongTensor([[token2id['<start>']]])).cuda()
            image = Variable(image).cuda()
        sentence_probs = []
        sentences = []
        for k in range(bw):
            sentences.append(start_s)
            sentence_probs.append(1.0)

        for l in range(max_len):
            temp_probs = []
            temp_s = []
            stop_flag = 0
            for j in range(bw):
                sentence = sentences[j]
                if sentence[0, -1].data.cpu().numpy()[0] == token2id['<end>']:
                    temp_probs.append(sentence_probs[j])
                    temp_s.append(sentence)
                    stop_flag += 1
                    continue
                logits = self.forward(images=image, input_seq=sentence, target_seq=None, mask=None)  # (1, voc_size, l)
                pred = logits[:, :, -1]  # (1, voc_size)
                probs = softmax(pred)
                value, index = torch.topk(probs.view(-1), k=bw)
                if l == 0:
                    for k in range(bw):
                        current_w = index[k]  # Variable
                        sentences[k] = torch.cat([sentences[k], current_w.view(1, 1)], dim=1)
                        sentence_probs[k] = sentence_probs[k] * value[k].data.cpu().numpy()[0]
                    break
                else:
                    for k in range(bw):
                        current_w = index[k]  # Variable
                        temp_s.append(torch.cat([sentence, current_w.view(1, 1)], dim=1))
                        temp_probs.append(sentence_probs[j] * value[k].data.cpu().numpy()[0])
            if l != 0:
                value, index = torch.FloatTensor(temp_probs).topk(bw)
                for k in range(bw):
                    sentences[k] = temp_s[index[k]]
                    sentence_probs[k] = value[k]
            if stop_flag == bw:
                break
        return sentences, sentence_probs


class NewHierarchicalAttention(nn.Module):
    # h is applied to generate gates instead of GLU gates
    def __init__(self, visual_channel, conception_channel, hidden_size=512, bias=True):
        super(NewHierarchicalAttention, self).__init__()
        input_size = conception_channel + visual_channel
        self.gru = nn.GRUCell(input_size=input_size,
                              hidden_size=hidden_size,
                              bias=bias)

    def forward(self, vis_att_seq, concept_seq, hidden_states):
        """
        :param vis_att_seq: (b, l, cv)
        :param concept_seq: (b, l, cc)
        :param hidden_states: (b, l, num_hidden)
        :return:
        """
        b, l, concept_channel = concept_seq.shape
        _, _, vis_att_channel = vis_att_seq.shape
        _, _, num_hidden = hidden_states.shape
        inputs = torch.cat([concept_seq.contiguous().view(-1, concept_channel),
                            vis_att_seq.contiguous().view(-1, vis_att_channel)],
                           dim=1)  # (b*l, concept_channel+vis_att_channel)
        input_hidden_states = hidden_states.contiguous().view(-1, num_hidden)  # (b*l, num_hidden)
        out_hidden_state = self.gru.forward(inputs, hx=input_hidden_states)  # (b*l, num_hidden)

        return out_hidden_state.view(b, l, num_hidden)


class NewCNNPlusCNNHierAtt(nn.Module):

    def __init__(self, config):
        super(NewCNNPlusCNNHierAtt, self).__init__()
        self._max_epoch = config.max_epoch
        self._batch_size = config.batch_size
        self._encoder_name = config.encoder_name
        self._kernel_size = config.kernel_size
        self._num_layers = config.num_layers
        self._channels = config.channels
        self._prediction_dim = config.prediction_dim
        self._voc_size = config.voc_size
        self._hier_att_hidden_size = config.hier_att_hidden_size
        self._hier_att_lang_hidden_size = config.hier_att_lang_hidden_size
        self._image_dir = config.image_dir
        self._train_ann_file = config.train_ann_file
        self._val_ann_file = config.val_ann_file
        self._split_file = config.split_file
        self._vocab_file = config.vocab_file
        self._trained_model = config.trained_model
        self._log_file = config.log_file
        self._image_width = config.width
        self._image_height = config.height
        self._is_dotatt = config.is_dotatt
        self._is_gpu = config.is_gpu
        self._is_train = config.is_train
        self._shuffle = config.shuffle
        self._num_workers = config.num_workers
        self._transformer = config.transformer
        self._result_file = config.result_file
        self._annfile = config.annfile
        self._weight_decay = config.weight_decay
        self._keep_prob = config.keep_prob

        self._sigmoid = nn.Sigmoid()
        self._module_list = nn.ModuleList()
        self._conv_layers = nn.ModuleList()

        self._build()

    def _build(self):
        # encoder
        if self._encoder_name == 'vgg16':
            encoder = vgg16_features
            if self._is_dotatt:
                att = AttentionDotProduct(visual_channel=512, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=512, conception_channel=self._channels)
            hier_att = NewHierarchicalAttention(visual_channel=512, conception_channel=self._channels,
                                                hidden_size=self._hier_att_hidden_size)
            h2gates_vis = nn.Conv1d(in_channels=self._hier_att_hidden_size, out_channels=512,
                                    kernel_size=1)
            pred = Prediction(vis_channel=512, con_channel=self._channels,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim, keep_prob=self._keep_prob)
        elif self._encoder_name == 'resnet152':
            encoder = resnet152_features
            if self._is_dotatt:
                att = AttentionDotProduct(visual_channel=2048, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=2048, conception_channel=self._channels)
            hier_att = NewHierarchicalAttention(visual_channel=2048, conception_channel=self._channels,
                                                hidden_size = self._hier_att_hidden_size)
            h2gates_vis = nn.Conv1d(in_channels=self._hier_att_hidden_size, out_channels=2048,
                                    kernel_size=1)
            pred = Prediction(vis_channel=2048, con_channel=self._channels,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim, keep_prob=self._keep_prob)
        else:
            encoder = resnet101_features
            if self._is_dotatt:
                att = AttentionDotProduct(visual_channel=2048, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=2048, conception_channel=self._channels)
            hier_att = NewHierarchicalAttention(visual_channel=2048, conception_channel=self._channels,
                                                hidden_size=self._hier_att_hidden_size)
            h2gates_vis = nn.Conv1d(in_channels=self._hier_att_hidden_size, out_channels=2048,
                                    kernel_size=1)
            pred = Prediction(vis_channel=2048, con_channel=self._channels,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim, keep_prob=self._keep_prob)

        h2gates = nn.Conv1d(in_channels=self._hier_att_hidden_size, out_channels=self._channels,
                            kernel_size=1)

        word_embedding = Embedding(num_embeddings=self._voc_size,
                                   embedding_dim=self._channels)
        decoder = []
        for i in range(self._num_layers):
            causalconv = CausalConv1d(in_channels=self._channels,
                                      out_channels=self._channels,
                                      kernel_size=self._kernel_size)
            decoder.append(causalconv)

        self._module_list.extend([encoder, att, hier_att, pred, word_embedding, h2gates, h2gates_vis])

        self._conv_layers.extend(decoder)
        if self._is_gpu:
            self._module_list.cuda()
            self._conv_layers.cuda()

    def forward(self, images, input_seq, target_seq, mask):
        # module_list--0: encoder, 1: att, 2: hier_att, 3: pred, 4: embedding, 5: h2gates, 6: h2gates_vis
        # conv layers--0:num_layers-1
        vis_map = self._module_list[0](images)  # (b, 2048, n, n)
        embeddings = self._module_list[4](input_seq)  # (b, channel, l)

        b, l = input_seq.shape
        init_att_map = self._module_list[1](vis_map, embeddings)  # (b, 2048, l)
        init_att_map = init_att_map.transpose(1, 2)  # (b, l, 2048)
        init_h = Variable(torch.zeros(b, l, self._hier_att_hidden_size))
        if self._is_gpu:
            init_h = init_h.cuda()
        init_concept_map = embeddings.transpose(1, 2)

        for i in range(self._num_layers):
            hidden_state = self._module_list[2](init_att_map, init_concept_map, init_h)
            conv_layer_i_h = self._conv_layers[i](init_concept_map.transpose(1, 2))
            layer_i_gates = self._sigmoid(self._module_list[5](hidden_state.transpose(1, 2)))
            layer_i_vis_gates = self._sigmoid(self._module_list[6](hidden_state.transpose(1, 2)))
            conv_layer_i_h = torch.mul(conv_layer_i_h, layer_i_gates)
            layer_i_att_map = self._module_list[1](vis_map, conv_layer_i_h).transpose(1, 2) + \
                torch.mul(layer_i_vis_gates.transpose(1, 2), init_att_map)
            init_h = hidden_state
            init_att_map = layer_i_att_map
            init_concept_map = conv_layer_i_h.transpose(1, 2)

        logits = self._module_list[3](init_att_map.transpose(1, 2), init_concept_map.transpose(1, 2))
        if self._is_train:
            loss = self._module_list[3].loss_stable(logits, targets=target_seq, mask=mask)
            return loss
        else:
            return logits

    def data_produceer(self,
                       image_dir,
                       train_ann_file,
                       val_ann_file,
                       split_file,
                       vocab_file,
                       transformer,
                       batch_size,
                       width,
                       height,
                       num_workers,
                       shuffle,
                       split_key):

        data_loader = ak_get_data_loader(
            image_dir=image_dir,
            split_file=split_file,
            split_key=split_key,
            transformer=transformer,
            width=width,
            height=height,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )
        return data_loader

    def save_load_model(self, mode='save'):
        if mode == 'save':
            print "Saving model..."
            torch.save(self.state_dict(), self._trained_model)
        elif mode == 'load':
            print "Loading model..."
            self.load_state_dict(torch.load(self._trained_model))
        else:
            raise IOError

    def _get_para_group(self, encoder_lr, decoder_lr):
        encoder_paras = {
            'params': [p[1] for p in self. named_parameters() if '_module_list.0' in p[0] and p[1].requires_grad],
            'lr': encoder_lr
        }
        decoder_paras = {
            'params':[p[1] for p in self. named_parameters() if '_module_list.0' not in p[0] and p[1].requires_grad],
            'lr': decoder_lr
        }
        print 'number of encoder parameters is: %d\n' % (len(encoder_paras['params']))
        print 'number of decoder parameters is: %d\n' % (len(decoder_paras['params']))
        return [encoder_paras, decoder_paras]

    def trainer(self):
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key=['train', 'val'],
            transformer=self._transformer,
            batch_size=self._batch_size,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=self._shuffle
        )
        try:
            self.save_load_model('load')
        except:
            print 'The model is randomly initialized......'
            pass
        params_group = self._get_para_group(encoder_lr=1e-5, decoder_lr=3e-4)
        optimizer = torch.optim.Adam(params_group, weight_decay=self._weight_decay)
        hostname = os.popen('hostname').read()
        fo = open(self._log_file, 'a+')
        fo.write(hostname)
        fo.close()

        self.inference(result_file=self._result_file,
                       bw=3)
        eval_scores = scores(res_file=self._result_file, ann_file=self._annfile)
        # eval_scores = {'CIDEr': 0.0}
        buf = 'cider=%.5f\n' % (eval_scores['CIDEr'])
        fo = open(self._log_file, 'a+')
        fo.write(buf)
        fo.close()
        cider_scores = [eval_scores['CIDEr']]

        for epoch in range(1, self._max_epoch + 1):
            self.train()
            for i, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
                optimizer.zero_grad()
                if self._is_gpu:
                    loss = self.forward(Variable(images).cuda(),
                                        Variable(inputs).cuda(),
                                        Variable(targets).cuda(),
                                        Variable(mask).cuda())
                else:
                    loss = self.forward(Variable(images),
                                        Variable(inputs),
                                        Variable(targets),
                                        Variable(mask))
                loss.backward()
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, self.parameters()), max_norm=5.0)
                optimizer.step()
                if i % 1000 == 0:
                    buf = 'epoch=%d, batch#=%d, loss=%.3f\n'%(epoch, i, loss.data.cpu().numpy()[0])
                    fo = open(self._log_file, 'a+')
                    fo.write(buf)
                    fo.close()

            previous_cider = max(cider_scores)
            self.inference(result_file=self._result_file,
                           bw=3)
            eval_scores = scores(res_file=self._result_file, ann_file=self._annfile)
            buf = 'cider=%.5f\n' % (eval_scores['CIDEr'])
            fo = open(self._log_file, 'a+')
            fo.write(buf)
            fo.close()
            cider_scores.append(eval_scores['CIDEr'])

            if eval_scores['CIDEr'] > previous_cider:
                self.save_load_model(mode='save')

    def _cpu_debug_load_model(self):
        state_dict = torch.load(self._trained_model, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)

    def inference(self, result_file='results.json', bw=3, max_len=20, is_debug=False, is_loading_model=False):
        if is_loading_model:
            self.save_load_model(mode='load')
        self.eval()
        self._is_train = False
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key='test',
            transformer=None,
            batch_size=1,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=False)
        with open(self._vocab_file) as f:
            voc = pkl.load(f)  # {'token2id':{}, 'id2token': {}}
        id2token = voc['id2token']
        token2id = voc['token2id']
        im_ids = []
        captions = []
        for num, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
            image_id = image_ids.numpy()[0, 0]
            if image_id not in im_ids:
                im_ids.append(image_id)
                cap = ' '
                cap_dict = {}
                s, sp = self.beam_search(image=images, voc=voc, is_debug=is_debug, bw=bw)
                b, l = s[0].shape
                w_list = []
                for i in range(l):
                    w = id2token[s[0][0, i].data.cpu().numpy()[0]]
                    if w != '<start>' and w != '<end>':
                        w_list.append(w)
                cap = cap.join(w_list)
                if bw > 2:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\nsec_prob=%.8f\nthird_prob=%.8f\n')%\
                          (num, image_id, cap, sp[0], sp[1], sp[2])
                else:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\n') % (num, image_id, cap, sp[0])
                cap_dict['image_id'] = int(image_id)
                cap_dict['caption'] = cap
                captions.append(cap_dict)
            else:
                continue
        with open(result_file, 'w') as f:
            json.dump(captions, f)
        self._is_train = True

    def inference_test(self, image_dir, result_file='results.json',
                       bw=3, max_len=20, is_debug=False, is_loading_model=False):
        if is_loading_model:
            self.save_load_model(mode='load')
        self.eval()
        self._is_train = False
        image_dict_list = get_image_path(image_dir)
        with open(self._vocab_file) as f:
            voc = pkl.load(f)  # {'token2id':{}, 'id2token': {}}
        id2token = voc['id2token']
        token2id = voc['token2id']
        im_ids = []
        captions = []
        num = 0
        totensor = tv.transforms.ToTensor()
        normalizer = tv.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        for image_dict in image_dict_list:
            image_id = image_dict['image_id']
            image_path = image_dict['image_path']
            if image_id not in im_ids:
                num += 1
                im_ids.append(image_id)
                image = Image.open(image_path).convert('RGB')
                image = image.resize((self._image_width, self._image_height), resample=Image.BILINEAR)
                image = totensor(image)
                image = normalizer(image)
                images = image.unsqueeze(0)
                cap = ' '
                cap_dict = {}
                s, sp = self.beam_search(image=images, voc=voc, is_debug=is_debug, bw=bw)
                b, l = s[0].shape
                w_list = []
                for i in range(l):
                    w = id2token[s[0][0, i].data.cpu().numpy()[0]]
                    if w != '<start>' and w != '<end>':
                        w_list.append(w)
                cap = cap.join(w_list)
                if bw > 2:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\nsec_prob=%.8f\nthird_prob=%.8f\n')%\
                          (num, image_id, cap, sp[0], sp[1], sp[2])
                else:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\n') % (num, image_id, cap, sp[0])
                cap_dict['image_id'] = int(image_id)
                cap_dict['caption'] = cap
                captions.append(cap_dict)
            else:
                continue
        with open(result_file, 'w') as f:
            json.dump(captions, f)
        self._is_train = True

    def beam_search(self, image, voc, bw=3, max_len=20, softmax=nn.Softmax(dim=-1), is_debug=False):
        """
        :param image: (1, 3, 448, 448)
        :param bw:
        :return:
        """
        token2id = voc['token2id']
        if is_debug:
            start_s = Variable(torch.LongTensor([[token2id['<start>']]]))
            image = Variable(image)
        else:
            start_s = Variable(torch.LongTensor([[token2id['<start>']]])).cuda()
            image = Variable(image).cuda()
        sentence_probs = []
        sentences = []
        for k in range(bw):
            sentences.append(start_s)
            sentence_probs.append(1.0)

        for l in range(max_len):
            temp_probs = []
            temp_s = []
            stop_flag = 0
            for j in range(bw):
                sentence = sentences[j]
                if sentence[0, -1].data.cpu().numpy()[0] == token2id['<end>']:
                    temp_probs.append(sentence_probs[j])
                    temp_s.append(sentence)
                    stop_flag += 1
                    continue
                logits = self.forward(images=image, input_seq=sentence, target_seq=None, mask=None)  # (1, voc_size, l)
                pred = logits[:, :, -1]  # (1, voc_size)
                probs = softmax(pred)
                value, index = torch.topk(probs.view(-1), k=bw)
                if l == 0:
                    for k in range(bw):
                        current_w = index[k]  # Variable
                        sentences[k] = torch.cat([sentences[k], current_w.view(1, 1)], dim=1)
                        sentence_probs[k] = sentence_probs[k] * value[k].data.cpu().numpy()[0]
                    break
                else:
                    for k in range(bw):
                        current_w = index[k]  # Variable
                        temp_s.append(torch.cat([sentence, current_w.view(1, 1)], dim=1))
                        temp_probs.append(sentence_probs[j] * value[k].data.cpu().numpy()[0])
            if l != 0:
                value, index = torch.FloatTensor(temp_probs).topk(bw)
                for k in range(bw):
                    sentences[k] = temp_s[index[k]]
                    sentence_probs[k] = value[k]
            if stop_flag == bw:
                break
        return sentences, sentence_probs


class NewCNNPlusCNNHierMultiscaleAtt(nn.Module):

    def __init__(self, config):
        super(NewCNNPlusCNNHierMultiscaleAtt, self).__init__()
        self._max_epoch = config.max_epoch
        self._batch_size = config.batch_size
        self._encoder_name = config.encoder_name
        self._kernel_size = config.kernel_size
        self._num_layers = config.num_layers
        self._channels = config.channels
        self._prediction_dim = config.prediction_dim
        self._voc_size = config.voc_size
        self._hier_att_hidden_size = config.hier_att_hidden_size
        self._hier_att_lang_hidden_size = config.hier_att_lang_hidden_size
        self._image_dir = config.image_dir
        self._train_ann_file = config.train_ann_file
        self._val_ann_file = config.val_ann_file
        self._split_file = config.split_file
        self._vocab_file = config.vocab_file
        self._trained_model = config.trained_model
        self._log_file = config.log_file
        self._image_width = config.width
        self._image_height = config.height
        self._is_dotatt = config.is_dotatt
        self._is_gpu = config.is_gpu
        self._is_train = config.is_train
        self._shuffle = config.shuffle
        self._num_workers = config.num_workers
        self._transformer = config.transformer
        self._result_file = config.result_file
        self._annfile = config.annfile
        self._weight_decay = config.weight_decay
        self._keep_prob = config.keep_prob

        self._sigmoid = nn.Sigmoid()
        self._module_list = nn.ModuleList()
        self._conv_layers = nn.ModuleList()

        self._build()

    def _build(self):
        # encoder
        if self._encoder_name == 'vgg16':
            encoder = FullVgg16()
            if self._is_dotatt:
                att = AttentionDotProduct(visual_channel=512, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=512, conception_channel=self._channels)
            hier_att = NewHierarchicalAttention(visual_channel=512, conception_channel=self._channels,
                                                hidden_size=self._hier_att_hidden_size)
            h2gates_vis = nn.Conv1d(in_channels=self._hier_att_hidden_size, out_channels=512,
                                    kernel_size=1)
            pred = Prediction(vis_channel=512, con_channel=self._channels,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim, keep_prob=self._keep_prob)
        elif self._encoder_name == 'resnet152':
            encoder = resnet152_features
            if self._is_dotatt:
                att = AttentionDotProduct(visual_channel=2048, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=2048, conception_channel=self._channels)
            hier_att = NewHierarchicalAttention(visual_channel=2048, conception_channel=self._channels,
                                                hidden_size = self._hier_att_hidden_size)
            h2gates_vis = nn.Conv1d(in_channels=self._hier_att_hidden_size, out_channels=2048,
                                    kernel_size=1)
            pred = Prediction(vis_channel=2048, con_channel=self._channels,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim, keep_prob=self._keep_prob)
        else:
            encoder = FullResNet101(is_transfer=True)
            if self._is_dotatt:
                att = AttentionDotProduct(visual_channel=2048, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=2048, conception_channel=self._channels)
            hier_att = NewHierarchicalAttention(visual_channel=2048, conception_channel=self._channels,
                                                hidden_size=self._hier_att_hidden_size)
            h2gates_vis = nn.Conv1d(in_channels=self._hier_att_hidden_size, out_channels=2048,
                                    kernel_size=1)
            pred = Prediction(vis_channel=2048, con_channel=self._channels,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim, keep_prob=self._keep_prob)

        h2gates = nn.Conv1d(in_channels=self._hier_att_hidden_size, out_channels=self._channels,
                            kernel_size=1)

        word_embedding = Embedding(num_embeddings=self._voc_size,
                                   embedding_dim=self._channels)
        decoder = []
        for i in range(self._num_layers):
            causalconv = CausalConv1d(in_channels=self._channels,
                                      out_channels=self._channels,
                                      kernel_size=self._kernel_size)
            decoder.append(causalconv)

        self._module_list.extend([encoder, att, hier_att, pred, word_embedding, h2gates, h2gates_vis])

        self._conv_layers.extend(decoder)
        if self._is_gpu:
            self._module_list.cuda()
            self._conv_layers.cuda()

    def forward(self, images, input_seq, target_seq, mask):
        # module_list--0: encoder, 1: att, 2: hier_att, 3: pred, 4: embedding, 5: h2gates, 6: h2gates_vis
        # conv layers--0:num_layers-1
        if self._encoder_name == 'vgg16':
            vis_map_l, vis_map = self._module_list[0](images)
            vis_map_m = vis_map
        else:
            vis_map_l, vis_map_m, vis_map = self._module_list[0](images)  # (b, 2048, n, n)
        embeddings = self._module_list[4](input_seq)  # (b, channel, l)

        b, l = input_seq.shape
        init_att_map = self._module_list[1](vis_map_l, embeddings)  # (b, 2048, l)
        init_att_map = init_att_map.transpose(1, 2)  # (b, l, 2048)
        init_h = Variable(torch.zeros(b, l, self._hier_att_hidden_size))
        if self._is_gpu:
            init_h = init_h.cuda()
        init_concept_map = embeddings.transpose(1, 2)

        for i in range(self._num_layers):
            hidden_state = self._module_list[2](init_att_map, init_concept_map, init_h)
            conv_layer_i_h = self._conv_layers[i](init_concept_map.transpose(1, 2))
            layer_i_gates = self._sigmoid(self._module_list[5](hidden_state.transpose(1, 2)))
            layer_i_vis_gates = self._sigmoid(self._module_list[6](hidden_state.transpose(1, 2)))
            conv_layer_i_h = torch.mul(conv_layer_i_h, layer_i_gates)
            if i < 2:
                layer_i_att_map = self._module_list[1](vis_map_l, conv_layer_i_h).transpose(1, 2) + \
                    torch.mul(layer_i_vis_gates.transpose(1, 2), init_att_map)
            elif i < 4:
                layer_i_att_map = self._module_list[1](vis_map_m, conv_layer_i_h).transpose(1, 2) + \
                                  torch.mul(layer_i_vis_gates.transpose(1, 2), init_att_map)
            else:
                layer_i_att_map = self._module_list[1](vis_map, conv_layer_i_h).transpose(1, 2) + \
                                  torch.mul(layer_i_vis_gates.transpose(1, 2), init_att_map)
            init_h = hidden_state
            init_att_map = layer_i_att_map
            init_concept_map = conv_layer_i_h.transpose(1, 2)

        logits = self._module_list[3](init_att_map.transpose(1, 2), init_concept_map.transpose(1, 2))
        if self._is_train:
            loss = self._module_list[3].loss_stable(logits, targets=target_seq, mask=mask)
            return loss
        else:
            return logits

    def data_produceer(self,
                       image_dir,
                       train_ann_file,
                       val_ann_file,
                       split_file,
                       vocab_file,
                       transformer,
                       batch_size,
                       width,
                       height,
                       num_workers,
                       shuffle,
                       split_key):

        data_loader = ak_get_data_loader(
            image_dir=image_dir,
            split_file=split_file,
            split_key=split_key,
            transformer=transformer,
            width=width,
            height=height,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )
        return data_loader

    def save_load_model(self, mode='save'):
        if mode == 'save':
            print "Saving model..."
            torch.save(self.state_dict(), self._trained_model)
        elif mode == 'load':
            print "Loading model..."
            self.load_state_dict(torch.load(self._trained_model))
        else:
            raise IOError

    def _get_para_group(self, encoder_lr, decoder_lr):
        encoder_paras = {
            'params': [p[1] for p in self. named_parameters() if '_module_list.0' in p[0] and p[1].requires_grad],
            'lr': encoder_lr
        }
        decoder_paras = {
            'params':[p[1] for p in self. named_parameters() if '_module_list.0' not in p[0] and p[1].requires_grad],
            'lr': decoder_lr
        }
        print 'number of encoder parameters is: %d\n' % (len(encoder_paras['params']))
        print 'number of decoder parameters is: %d\n' % (len(decoder_paras['params']))
        return [encoder_paras, decoder_paras]

    def trainer(self):
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key=['train', 'val'],
            transformer=self._transformer,
            batch_size=self._batch_size,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=self._shuffle
        )
        try:
            self.save_load_model('load')
        except:
            print 'The model is randomly initialized......'
            pass
        params_group = self._get_para_group(encoder_lr=1e-5, decoder_lr=3e-4)
        optimizer = torch.optim.Adam(params_group, weight_decay=self._weight_decay)
        hostname = os.popen('hostname').read()
        fo = open(self._log_file, 'a+')
        fo.write(hostname)
        fo.close()

        # self.inference(result_file=self._result_file,
        #                bw=3)
        # eval_scores = scores(res_file=self._result_file, ann_file=self._annfile)
        eval_scores = {'CIDEr': 0.0}
        buf = 'cider=%.5f\n' % (eval_scores['CIDEr'])
        fo = open(self._log_file, 'a+')
        fo.write(buf)
        fo.close()
        cider_scores = [eval_scores['CIDEr']]

        for epoch in range(1, self._max_epoch + 1):
            self.train()
            for i, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
                optimizer.zero_grad()
                if self._is_gpu:
                    loss = self.forward(Variable(images).cuda(),
                                        Variable(inputs).cuda(),
                                        Variable(targets).cuda(),
                                        Variable(mask).cuda())
                else:
                    loss = self.forward(Variable(images),
                                        Variable(inputs),
                                        Variable(targets),
                                        Variable(mask))
                loss.backward()
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, self.parameters()), max_norm=5.0)
                optimizer.step()
                if i % 1000 == 0:
                    buf = 'epoch=%d, batch#=%d, loss=%.3f\n'%(epoch, i, loss.data.cpu().numpy()[0])
                    fo = open(self._log_file, 'a+')
                    fo.write(buf)
                    fo.close()

            previous_cider = max(cider_scores)
            self.inference(result_file=self._result_file,
                           bw=3)
            eval_scores = scores(res_file=self._result_file, ann_file=self._annfile)
            buf = 'cider=%.5f\n' % (eval_scores['CIDEr'])
            fo = open(self._log_file, 'a+')
            fo.write(buf)
            fo.close()
            cider_scores.append(eval_scores['CIDEr'])

            if eval_scores['CIDEr'] > previous_cider:
                self.save_load_model(mode='save')

    def _cpu_debug_load_model(self):
        state_dict = torch.load(self._trained_model, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)

    def inference(self, result_file='results.json', bw=3, max_len=20, is_debug=False, is_loading_model=False):
        if is_loading_model:
            self.save_load_model(mode='load')
        self.eval()
        self._is_train = False
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key='test',
            transformer=None,
            batch_size=1,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=False)
        with open(self._vocab_file) as f:
            voc = pkl.load(f)  # {'token2id':{}, 'id2token': {}}
        id2token = voc['id2token']
        token2id = voc['token2id']
        im_ids = []
        captions = []
        for num, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
            image_id = image_ids.numpy()[0, 0]
            if image_id not in im_ids:
                im_ids.append(image_id)
                cap = ' '
                cap_dict = {}
                s, sp = self.beam_search(image=images, voc=voc, is_debug=is_debug, bw=bw)
                b, l = s[0].shape
                w_list = []
                for i in range(l):
                    w = id2token[s[0][0, i].data.cpu().numpy()[0]]
                    if w != '<start>' and w != '<end>':
                        w_list.append(w)
                cap = cap.join(w_list)
                if bw > 2:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\nsec_prob=%.8f\nthird_prob=%.8f\n')%\
                          (num, image_id, cap, sp[0], sp[1], sp[2])
                else:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\n') % (num, image_id, cap, sp[0])
                cap_dict['image_id'] = int(image_id)
                cap_dict['caption'] = cap
                captions.append(cap_dict)
            else:
                continue
        with open(result_file, 'w') as f:
            json.dump(captions, f)
        self._is_train = True

    def inference_test(self, image_dir, result_file='results.json',
                       bw=3, max_len=20, is_debug=False, is_loading_model=False):
        if is_loading_model:
            self.save_load_model(mode='load')
        self.eval()
        self._is_train = False
        image_dict_list = get_image_path(image_dir)
        with open(self._vocab_file) as f:
            voc = pkl.load(f)  # {'token2id':{}, 'id2token': {}}
        id2token = voc['id2token']
        token2id = voc['token2id']
        im_ids = []
        captions = []
        num = 0
        totensor = tv.transforms.ToTensor()
        normalizer = tv.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        for image_dict in image_dict_list:
            image_id = image_dict['image_id']
            image_path = image_dict['image_path']
            if image_id not in im_ids:
                num += 1
                im_ids.append(image_id)
                image = Image.open(image_path).convert('RGB')
                image = image.resize((self._image_width, self._image_height), resample=Image.BILINEAR)
                image = totensor(image)
                image = normalizer(image)
                images = image.unsqueeze(0)
                cap = ' '
                cap_dict = {}
                s, sp = self.beam_search(image=images, voc=voc, is_debug=is_debug, bw=bw)
                b, l = s[0].shape
                w_list = []
                for i in range(l):
                    w = id2token[s[0][0, i].data.cpu().numpy()[0]]
                    if w != '<start>' and w != '<end>':
                        w_list.append(w)
                cap = cap.join(w_list)
                if bw > 2:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\nsec_prob=%.8f\nthird_prob=%.8f\n')%\
                          (num, image_id, cap, sp[0], sp[1], sp[2])
                else:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\n') % (num, image_id, cap, sp[0])
                cap_dict['image_id'] = int(image_id)
                cap_dict['caption'] = cap
                captions.append(cap_dict)
            else:
                continue
        with open(result_file, 'w') as f:
            json.dump(captions, f)
        self._is_train = True


    def beam_search(self, image, voc, bw=3, max_len=20, softmax=nn.Softmax(dim=-1), is_debug=False):
        """
        :param image: (1, 3, 448, 448)
        :param bw:
        :return:
        """
        token2id = voc['token2id']
        if is_debug:
            start_s = Variable(torch.LongTensor([[token2id['<start>']]]))
            image = Variable(image)
        else:
            start_s = Variable(torch.LongTensor([[token2id['<start>']]])).cuda()
            image = Variable(image).cuda()
        sentence_probs = []
        sentences = []
        for k in range(bw):
            sentences.append(start_s)
            sentence_probs.append(1.0)

        for l in range(max_len):
            temp_probs = []
            temp_s = []
            stop_flag = 0
            for j in range(bw):
                sentence = sentences[j]
                if sentence[0, -1].data.cpu().numpy()[0] == token2id['<end>']:
                    temp_probs.append(sentence_probs[j])
                    temp_s.append(sentence)
                    stop_flag += 1
                    continue
                logits = self.forward(images=image, input_seq=sentence, target_seq=None, mask=None)  # (1, voc_size, l)
                pred = logits[:, :, -1]  # (1, voc_size)
                probs = softmax(pred)
                value, index = torch.topk(probs.view(-1), k=bw)
                if l == 0:
                    for k in range(bw):
                        current_w = index[k]  # Variable
                        sentences[k] = torch.cat([sentences[k], current_w.view(1, 1)], dim=1)
                        sentence_probs[k] = sentence_probs[k] * value[k].data.cpu().numpy()[0]
                    break
                else:
                    for k in range(bw):
                        current_w = index[k]  # Variable
                        temp_s.append(torch.cat([sentence, current_w.view(1, 1)], dim=1))
                        temp_probs.append(sentence_probs[j] * value[k].data.cpu().numpy()[0])
            if l != 0:
                value, index = torch.FloatTensor(temp_probs).topk(bw)
                for k in range(bw):
                    sentences[k] = temp_s[index[k]]
                    sentence_probs[k] = value[k]
            if stop_flag == bw:
                break
        return sentences, sentence_probs


class CNNPlusCNNBottleneck(nn.Module):
    def __init__(self, config):
        super(CNNPlusCNNBottleneck, self).__init__()
        self._max_epoch = config.max_epoch
        self._batch_size = config.batch_size
        self._encoder_name = config.encoder_name
        self._kernel_size = config.kernel_size
        self._num_layers = config.num_layers
        self._channels = config.channels
        self._prediction_dim = config.prediction_dim
        self._voc_size = config.voc_size
        self._image_dir = config.image_dir
        self._train_ann_file = config.train_ann_file
        self._val_ann_file = config.val_ann_file
        self._split_file = config.split_file
        self._vocab_file = config.vocab_file
        self._trained_model = config.trained_model
        self._log_file = config.log_file
        self._image_width = config.width
        self._image_height = config.height
        self._is_gpu = config.is_gpu
        self._is_train = config.is_train
        self._shuffle = config.shuffle
        self._num_workers = config.num_workers
        self._transformer = config.transformer
        self._is_dotatt = config.is_dotatt
        self._result_file = config.result_file
        self._annfile = config.annfile
        self._keep_prob = config.keep_prob

        self._nonlinear_f = config.f
        self._blocks = nn.Sequential()
        self._modules_list = nn.ModuleList()

        self._build()

    def _build(self):
        # encoder
        if self._encoder_name == 'vgg16':
            encoder = vgg16_features
            if self._is_dotatt:
                att = AttentionDotProduct(visual_channel=512, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=512, conception_channel=self._channels)
            pred = Prediction(vis_channel=512, con_channel=self._channels, keep_prob=self._keep_prob,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim)
        elif self._encoder_name == 'resnet152':
            encoder = resnet152_features
            if self._is_dotatt:
                att = AttentionDotProduct(visual_channel=2048, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=2048, conception_channel=self._channels)
            pred = Prediction(vis_channel=2048, con_channel=self._channels, keep_prob=self._keep_prob,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim)
        else:
            encoder = resnet101_features
            if self._is_dotatt:
                att = AttentionDotProduct(visual_channel=2048, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=2048, conception_channel=self._channels)
            pred = Prediction(vis_channel=2048, con_channel=self._channels, keep_prob=self._keep_prob,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim)

        self._modules_list.append(encoder)
        self._modules_list.append(att)
        self._modules_list.append(pred)
        # word embedding layer
        word_embedding = Embedding(num_embeddings=self._voc_size,
                                   embedding_dim=self._channels)
        self._blocks.add_module(name='word_embedding', module=word_embedding)
        # conv layers
        for i in range(self._num_layers):
            key = 'layer' + str(i+1)
            conv = Bottleneck(in_channels=self._channels,
                              out_channels=self._channels,
                              kernel_size=self._kernel_size)
            self._blocks.add_module(name=key, module=conv)
            # self._blocks.add_module(name=key + '_f', module=self._nonlinear_f)
        if self._is_gpu:
            self._blocks = self._blocks.cuda()
            self._modules_list = self._modules_list.cuda()

    def data_produceer(self,
                       image_dir,
                       train_ann_file,
                       val_ann_file,
                       split_file,
                       vocab_file,
                       transformer,
                       batch_size,
                       width,
                       height,
                       num_workers,
                       shuffle,
                       split_key):
        data_loader = ak_get_data_loader(
            image_dir=image_dir,
            split_file=split_file,
            split_key=split_key,
            transformer=transformer,
            width=width,
            height=height,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )
        return data_loader

    def forward(self, images, input_seq, target_seq, mask):
        image_feature_map = self._modules_list[0](images)
        conceptions = self._blocks(input_seq)
        att_map = self._modules_list[1](image_feature_map, conceptions)
        logits = self._modules_list[2](att_map, conceptions)
        if self._is_train:
            loss = self._modules_list[2].loss_stable(logits, targets=target_seq, mask=mask)
            return loss
        else:
            return logits

    def save_load_model(self, mode='save'):
        if mode == 'save':
            print "Saving model..."
            torch.save(self.state_dict(), self._trained_model)
        elif mode == 'load':
            print "Loading model..."
            self.load_state_dict(torch.load(self._trained_model))
        else:
            raise IOError

    def _cpu_debug_load_model(self):
        state_dict = torch.load(self._trained_model, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)

    def _get_para_group(self, encoder_lr, decoder_lr):
        encoder_paras = {
            'params': [p[1] for p in self. named_parameters() if '_modules_list.0' in p[0] and p[1].requires_grad],
            'lr': encoder_lr
        }
        decoder_paras = {
            'params':[p[1] for p in self. named_parameters() if '_modules_list.0' not in p[0] and p[1].requires_grad],
            'lr': decoder_lr
        }
        print 'number of encoder parameters is: %d\n' % (len(encoder_paras['params']))
        print 'number of decoder parameters is: %d\n' % (len(decoder_paras['params']))
        return [encoder_paras, decoder_paras]

    def trainer(self):
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key=['train', 'val'],
            transformer=self._transformer,
            batch_size=self._batch_size,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=self._shuffle
        )
        try:
            self.save_load_model('load')
        except:
            print 'The model is randomly initialized......'
            pass
        param_group = self._get_para_group(encoder_lr=1e-5, decoder_lr=3e-4)
        optimizer = torch.optim.Adam(param_group)
        hostname = os.popen('hostname').read()
        fo = open(self._log_file, 'a+')
        fo.write(hostname)
        fo.close()
        if not os.path.exists(self._result_file):
            self.inference(result_file=self._result_file,
                           bw=3)
        eval_scores = scores(res_file=self._result_file, ann_file=self._annfile)
        buf = 'cider=%.5f\n' % (eval_scores['CIDEr'])
        fo = open(self._log_file, 'a+')
        fo.write(buf)
        fo.close()
        cider_scores = [eval_scores['CIDEr']]

        for epoch in range(1, self._max_epoch + 1):
            self.train()
            for i, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
                optimizer.zero_grad()
                if self._is_gpu:
                    loss = self.forward(Variable(images).cuda(),
                                        Variable(inputs).cuda(),
                                        Variable(targets).cuda(),
                                        Variable(mask).cuda())
                else:
                    loss = self.forward(Variable(images),
                                        Variable(inputs),
                                        Variable(targets),
                                        Variable(mask))
                loss.backward()
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, self.parameters()), max_norm=5.0)
                optimizer.step()
                if i % 1000 == 0:
                    buf = 'epoch=%d, batch#=%d, loss=%.3f\n' % (epoch, i, loss.data.cpu().numpy()[0])
                    fo = open(self._log_file, 'a+')
                    fo.write(buf)
                    fo.close()

            previous_cider = max(cider_scores)
            self.inference(result_file=self._result_file,
                           bw=3)
            eval_scores = scores(res_file=self._result_file, ann_file=self._annfile)
            buf = 'cider=%.5f\n' % (eval_scores['CIDEr'])
            fo = open(self._log_file, 'a+')
            fo.write(buf)
            fo.close()
            cider_scores.append(eval_scores['CIDEr'])

            if eval_scores['CIDEr'] > previous_cider:
                self.save_load_model(mode='save')

    def inference(self, result_file='results.json', bw=3, max_len=20, is_debug=False, is_loading_model=False):
        if is_loading_model:
            self.save_load_model(mode='load')
        self.eval()
        self._is_train = False
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key='test',
            transformer=None,
            batch_size=1,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=False)
        voc = pkl.load(open(self._vocab_file))  # {'token2id':{}, 'id2token': {}}
        id2token = voc['id2token']
        token2id = voc['token2id']
        im_ids = []
        captions = []
        for num, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
            image_id = image_ids.numpy()[0, 0]
            if image_id not in im_ids:
                im_ids.append(image_id)
                cap = ' '
                cap_dict = {}
                s, sp = self.beam_search(image=images, voc=voc, is_debug=is_debug, bw=bw)
                b, l = s[0].shape
                w_list = []
                for i in range(l):
                    w = id2token[s[0][0, i].data.cpu().numpy()[0]]
                    if w != '<start>' and w != '<end>':
                        w_list.append(w)
                cap = cap.join(w_list)
                if bw > 2:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\nsec_prob=%.8f\nthird_prob=%.8f\n')%\
                          (num, image_id, cap, sp[0], sp[1], sp[2])
                else:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\n') % (num, image_id, cap, sp[0])
                cap_dict['caption'] = cap
                cap_dict['image_id'] = int(image_id)
                captions.append(cap_dict)
            else:
                continue
        with open(result_file, 'w') as f:
            json.dump(captions, f)
        self._is_train = True

    def beam_search(self, image, voc, bw=3, max_len=20, softmax=nn.Softmax(dim=-1), is_debug=False):
        """
        :param image: (1, 3, 448, 448)
        :param bw:
        :return:
        """
        token2id = voc['token2id']
        if is_debug:
            start_s = Variable(torch.LongTensor([[token2id['<start>']]]))
            image = Variable(image)
        else:
            start_s = Variable(torch.LongTensor([[token2id['<start>']]])).cuda()
            image = Variable(image).cuda()
        sentence_probs = []
        sentences = []
        for k in range(bw):
            sentences.append(start_s)
            sentence_probs.append(1.0)

        for l in range(max_len):
            temp_probs = []
            temp_s = []
            stop_flag = 0
            for j in range(bw):
                sentence = sentences[j]
                if sentence[0, -1].data.cpu().numpy()[0] == token2id['<end>']:
                    temp_probs.append(sentence_probs[j])
                    temp_s.append(sentence)
                    stop_flag += 1
                    continue
                logits = self.forward(images=image, input_seq=sentence, target_seq=None, mask=None)  # (1, voc_size, l)
                pred = logits[:, :, -1]  # (1, voc_size)
                probs = softmax(pred)
                value, index = torch.topk(probs.view(-1), k=bw)
                if l == 0:
                    for k in range(bw):
                        current_w = index[k]  # Variable
                        sentences[k] = torch.cat([sentences[k], current_w.view(1, 1)], dim=1)
                        sentence_probs[k] = sentence_probs[k] * value[k].data.cpu().numpy()[0]
                    break
                else:
                    for k in range(bw):
                        current_w = index[k]  # Variable
                        temp_s.append(torch.cat([sentence, current_w.view(1, 1)], dim=1))
                        temp_probs.append(sentence_probs[j] * value[k].data.cpu().numpy()[0])
            if l != 0:
                value, index = torch.FloatTensor(temp_probs).topk(bw)
                for k in range(bw):
                    sentences[k] = temp_s[index[k]]
                    sentence_probs[k] = value[k]
            if stop_flag == bw:
                break
        return sentences, sentence_probs


class NewCNNPlusCNNHierAttBottleneck(nn.Module):

    def __init__(self, config):
        super(NewCNNPlusCNNHierAttBottleneck, self).__init__()
        self._max_epoch = config.max_epoch
        self._batch_size = config.batch_size
        self._encoder_name = config.encoder_name
        self._kernel_size = config.kernel_size
        self._num_layers = config.num_layers
        self._channels = config.channels
        self._prediction_dim = config.prediction_dim
        self._voc_size = config.voc_size
        self._hier_att_hidden_size = config.hier_att_hidden_size
        self._hier_att_lang_hidden_size = config.hier_att_lang_hidden_size
        self._image_dir = config.image_dir
        self._train_ann_file = config.train_ann_file
        self._val_ann_file = config.val_ann_file
        self._split_file = config.split_file
        self._vocab_file = config.vocab_file
        self._trained_model = config.trained_model
        self._log_file = config.log_file
        self._image_width = config.width
        self._image_height = config.height
        self._is_dotatt = config.is_dotatt
        self._is_gpu = config.is_gpu
        self._is_train = config.is_train
        self._shuffle = config.shuffle
        self._num_workers = config.num_workers
        self._transformer = config.transformer
        self._result_file = config.result_file
        self._annfile = config.annfile
        self._weight_decay = config.weight_decay
        self._keep_prob = config.keep_prob

        self._sigmoid = nn.Sigmoid()
        self._module_list = nn.ModuleList()
        self._conv_layers = nn.ModuleList()

        self._build()

    def _build(self):
        # encoder
        if self._encoder_name == 'vgg16':
            encoder = vgg16_features
            if self._is_dotatt:
                att = AttentionDotProduct(visual_channel=512, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=512, conception_channel=self._channels)
            hier_att = NewHierarchicalAttention(visual_channel=512, conception_channel=self._channels,
                                                hidden_size=self._hier_att_hidden_size)
            h2gates_vis = nn.Conv1d(in_channels=self._hier_att_hidden_size, out_channels=512,
                                    kernel_size=1)
            pred = Prediction(vis_channel=512, con_channel=self._channels,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim, keep_prob=self._keep_prob)
        elif self._encoder_name == 'resnet152':
            encoder = resnet152_features
            if self._is_dotatt:
                att = AttentionDotProduct(visual_channel=2048, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=2048, conception_channel=self._channels)
            hier_att = NewHierarchicalAttention(visual_channel=2048, conception_channel=self._channels,
                                                hidden_size = self._hier_att_hidden_size)
            h2gates_vis = nn.Conv1d(in_channels=self._hier_att_hidden_size, out_channels=2048,
                                    kernel_size=1)
            pred = Prediction(vis_channel=2048, con_channel=self._channels,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim, keep_prob=self._keep_prob)
        else:
            encoder = resnet101_features
            if self._is_dotatt:
                att = AttentionDotProduct(visual_channel=2048, concept_channel=self._channels)
            else:
                att = Attention(visual_channel=2048, conception_channel=self._channels)
            hier_att = NewHierarchicalAttention(visual_channel=2048, conception_channel=self._channels,
                                                hidden_size=self._hier_att_hidden_size)
            h2gates_vis = nn.Conv1d(in_channels=self._hier_att_hidden_size, out_channels=2048,
                                    kernel_size=1)
            pred = Prediction(vis_channel=2048, con_channel=self._channels,
                              voc_size=self._voc_size, num_hidden=self._prediction_dim, keep_prob=self._keep_prob)

        h2gates = nn.Conv1d(in_channels=self._hier_att_hidden_size, out_channels=self._channels,
                            kernel_size=1)

        word_embedding = Embedding(num_embeddings=self._voc_size,
                                   embedding_dim=self._channels)
        decoder = []
        for i in range(self._num_layers):
            causalconv = Bottleneck(in_channels=self._channels,
                                      out_channels=self._channels,
                                      kernel_size=self._kernel_size)
            decoder.append(causalconv)

        self._module_list.extend([encoder, att, hier_att, pred, word_embedding, h2gates, h2gates_vis])

        self._conv_layers.extend(decoder)
        if self._is_gpu:
            self._module_list.cuda()
            self._conv_layers.cuda()

    def forward(self, images, input_seq, target_seq, mask):
        # module_list--0: encoder, 1: att, 2: hier_att, 3: pred, 4: embedding, 5: h2gates, 6: h2gates_vis
        # conv layers--0:num_layers-1
        vis_map = self._module_list[0](images)  # (b, 2048, n, n)
        embeddings = self._module_list[4](input_seq)  # (b, channel, l)

        b, l = input_seq.shape
        init_att_map = self._module_list[1](vis_map, embeddings)  # (b, 2048, l)
        init_att_map = init_att_map.transpose(1, 2)  # (b, l, 2048)
        init_h = Variable(torch.zeros(b, l, self._hier_att_hidden_size))
        if self._is_gpu:
            init_h = init_h.cuda()
        init_concept_map = embeddings.transpose(1, 2)

        for i in range(self._num_layers):
            hidden_state = self._module_list[2](init_att_map, init_concept_map, init_h)
            conv_layer_i_h = self._conv_layers[i](init_concept_map.transpose(1, 2))
            layer_i_gates = self._sigmoid(self._module_list[5](hidden_state.transpose(1, 2)))
            layer_i_vis_gates = self._sigmoid(self._module_list[6](hidden_state.transpose(1, 2)))
            conv_layer_i_h = torch.mul(conv_layer_i_h, layer_i_gates)
            layer_i_att_map = self._module_list[1](vis_map, conv_layer_i_h).transpose(1, 2) + \
                torch.mul(layer_i_vis_gates.transpose(1, 2), init_att_map)
            init_h = hidden_state
            init_att_map = layer_i_att_map
            init_concept_map = conv_layer_i_h.transpose(1, 2)

        logits = self._module_list[3](init_att_map.transpose(1, 2), init_concept_map.transpose(1, 2))
        if self._is_train:
            loss = self._module_list[3].loss_stable(logits, targets=target_seq, mask=mask)
            return loss
        else:
            return logits

    def data_produceer(self,
                       image_dir,
                       train_ann_file,
                       val_ann_file,
                       split_file,
                       vocab_file,
                       transformer,
                       batch_size,
                       width,
                       height,
                       num_workers,
                       shuffle,
                       split_key):

        data_loader = ak_get_data_loader(
            image_dir=image_dir,
            split_file=split_file,
            split_key=split_key,
            transformer=transformer,
            width=width,
            height=height,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )
        return data_loader

    def save_load_model(self, mode='save'):
        if mode == 'save':
            print "Saving model..."
            torch.save(self.state_dict(), self._trained_model)
        elif mode == 'load':
            print "Loading model..."
            self.load_state_dict(torch.load(self._trained_model))
        else:
            raise IOError

    def _get_para_group(self, encoder_lr, decoder_lr):
        encoder_paras = {
            'params': [p[1] for p in self. named_parameters() if '_module_list.0' in p[0] and p[1].requires_grad],
            'lr': encoder_lr
        }
        decoder_paras = {
            'params':[p[1] for p in self. named_parameters() if '_module_list.0' not in p[0] and p[1].requires_grad],
            'lr': decoder_lr
        }
        print 'number of encoder parameters is: %d\n' % (len(encoder_paras['params']))
        print 'number of decoder parameters is: %d\n' % (len(decoder_paras['params']))
        return [encoder_paras, decoder_paras]

    def trainer(self):
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key=['train', 'val'],
            transformer=self._transformer,
            batch_size=self._batch_size,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=self._shuffle
        )
        try:
            self.save_load_model('load')
        except:
            print 'The model is randomly initialized......'
            pass
        params_group = self._get_para_group(encoder_lr=1e-5, decoder_lr=3e-4)
        optimizer = torch.optim.Adam(params_group, weight_decay=self._weight_decay)
        hostname = os.popen('hostname').read()
        fo = open(self._log_file, 'a+')
        fo.write(hostname)
        fo.close()

        if not os.path.exists(self._result_file):
            self.inference(result_file=self._result_file,
                           bw=3)
        eval_scores = scores(res_file=self._result_file, ann_file=self._annfile)
        # eval_scores = {'CIDEr': 0.0}
        buf = 'cider=%.5f\n' % (eval_scores['CIDEr'])
        fo = open(self._log_file, 'a+')
        fo.write(buf)
        fo.close()
        cider_scores = [eval_scores['CIDEr']]

        for epoch in range(1, self._max_epoch + 1):
            self.train()
            for i, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
                optimizer.zero_grad()
                if self._is_gpu:
                    loss = self.forward(Variable(images).cuda(),
                                        Variable(inputs).cuda(),
                                        Variable(targets).cuda(),
                                        Variable(mask).cuda())
                else:
                    loss = self.forward(Variable(images),
                                        Variable(inputs),
                                        Variable(targets),
                                        Variable(mask))
                loss.backward()
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, self.parameters()), max_norm=5.0)
                optimizer.step()
                if i % 1000 == 0:
                    buf = 'epoch=%d, batch#=%d, loss=%.3f\n'%(epoch, i, loss.data.cpu().numpy()[0])
                    fo = open(self._log_file, 'a+')
                    fo.write(buf)
                    fo.close()

            previous_cider = max(cider_scores)
            self.inference(result_file=self._result_file,
                           bw=3)
            eval_scores = scores(res_file=self._result_file, ann_file=self._annfile)
            buf = 'cider=%.5f\n' % (eval_scores['CIDEr'])
            fo = open(self._log_file, 'a+')
            fo.write(buf)
            fo.close()
            cider_scores.append(eval_scores['CIDEr'])

            if eval_scores['CIDEr'] > previous_cider:
                self.save_load_model(mode='save')

    def _cpu_debug_load_model(self):
        state_dict = torch.load(self._trained_model, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)

    def inference(self, result_file='results.json', bw=3, max_len=20, is_debug=False, is_loading_model=False):
        if is_loading_model:
            self.save_load_model(mode='load')
        self.eval()
        self._is_train = False
        data_loader = self.data_produceer(
            image_dir=self._image_dir,
            train_ann_file=self._train_ann_file,
            val_ann_file=self._val_ann_file,
            split_file=self._split_file,
            vocab_file=self._vocab_file,
            split_key='test',
            transformer=None,
            batch_size=1,
            width=self._image_width,
            height=self._image_height,
            num_workers=self._num_workers,
            shuffle=False)
        with open(self._vocab_file) as f:
            voc = pkl.load(f)  # {'token2id':{}, 'id2token': {}}
        id2token = voc['id2token']
        token2id = voc['token2id']
        im_ids = []
        captions = []
        for num, (image_ids, images, inputs, targets, mask) in enumerate(data_loader):
            image_id = image_ids.numpy()[0, 0]
            if image_id not in im_ids:
                im_ids.append(image_id)
                cap = ' '
                cap_dict = {}
                s, sp = self.beam_search(image=images, voc=voc, is_debug=is_debug, bw=bw)
                b, l = s[0].shape
                w_list = []
                for i in range(l):
                    w = id2token[s[0][0, i].data.cpu().numpy()[0]]
                    if w != '<start>' and w != '<end>':
                        w_list.append(w)
                cap = cap.join(w_list)
                if bw > 2:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\nsec_prob=%.8f\nthird_prob=%.8f\n')%\
                          (num, image_id, cap, sp[0], sp[1], sp[2])
                else:
                    print ('num=%d  image_id=%d  caption=%s  probs=%.8f\n') % (num, image_id, cap, sp[0])
                cap_dict['image_id'] = int(image_id)
                cap_dict['caption'] = cap
                captions.append(cap_dict)
            else:
                continue
        with open(result_file, 'w') as f:
            json.dump(captions, f)
        self._is_train = True

    def beam_search(self, image, voc, bw=3, max_len=20, softmax=nn.Softmax(dim=-1), is_debug=False):
        """
        :param image: (1, 3, 448, 448)
        :param bw:
        :return:
        """
        token2id = voc['token2id']
        if is_debug:
            start_s = Variable(torch.LongTensor([[token2id['<start>']]]))
            image = Variable(image)
        else:
            start_s = Variable(torch.LongTensor([[token2id['<start>']]])).cuda()
            image = Variable(image).cuda()
        sentence_probs = []
        sentences = []
        for k in range(bw):
            sentences.append(start_s)
            sentence_probs.append(1.0)

        for l in range(max_len):
            temp_probs = []
            temp_s = []
            stop_flag = 0
            for j in range(bw):
                sentence = sentences[j]
                if sentence[0, -1].data.cpu().numpy()[0] == token2id['<end>']:
                    temp_probs.append(sentence_probs[j])
                    temp_s.append(sentence)
                    stop_flag += 1
                    continue
                logits = self.forward(images=image, input_seq=sentence, target_seq=None, mask=None)  # (1, voc_size, l)
                pred = logits[:, :, -1]  # (1, voc_size)
                probs = softmax(pred)
                value, index = torch.topk(probs.view(-1), k=bw)
                if l == 0:
                    for k in range(bw):
                        current_w = index[k]  # Variable
                        sentences[k] = torch.cat([sentences[k], current_w.view(1, 1)], dim=1)
                        sentence_probs[k] = sentence_probs[k] * value[k].data.cpu().numpy()[0]
                    break
                else:
                    for k in range(bw):
                        current_w = index[k]  # Variable
                        temp_s.append(torch.cat([sentence, current_w.view(1, 1)], dim=1))
                        temp_probs.append(sentence_probs[j] * value[k].data.cpu().numpy()[0])
            if l != 0:
                value, index = torch.FloatTensor(temp_probs).topk(bw)
                for k in range(bw):
                    sentences[k] = temp_s[index[k]]
                    sentence_probs[k] = value[k]
            if stop_flag == bw:
                break
        return sentences, sentence_probs


