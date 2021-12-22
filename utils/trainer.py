from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from miscc.util import mkdir_p
from miscc.util import load_params, copy_G_params, build_super_images
from utils.model import GENERATOR
from utils.nt_xent import NT_Xent
from utils.masks import mask_correlated_samples
from utils.loss import discriminator_loss, generator_loss, KL_loss, words_loss
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
from miscc.config import cfg
from PIL import Image

import numpy as np
import os
import time

#################################################
# DO NOT CHANGE
from utils.model import RNN_ENCODER, CNN_ENCODER, GENERATOR, DISCRIMINATOR
#################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class trainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword, dataset):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        # torch.cuda.set_device(cfg.GPU_ID)
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID
        cudnn.benchmark = True

        self.batch_size = cfg.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.dataset = dataset
        self.writer = SummaryWriter('runs/visualize')

    def prepare_data(self, data):
        """
        Prepares data given by dataloader
        e.g., x = Variable(x).cuda()
        """
        imgs, captions, captions_lens, class_ids, keys, captions_2, captions_lens_2 = data

        # sort data by the length in a decreasing order
        sorted_cap_lens, sorted_cap_indices = \
            torch.sort(captions_lens, 0, True)
        
        sorted_cap_lens_2, sorted_cap_indices_2 = \
            torch.sort(captions_lens_2, 0, True)

        imgs_2 = imgs.copy()

        real_imgs = []
        for i in range(len(imgs)):
            imgs[i] = imgs[i][sorted_cap_indices]
            if cfg.CUDA:
                real_imgs.append(Variable(imgs[i]).cuda())
            else:
                real_imgs.append(Variable(imgs[i]))

        real_imgs_2 = []
        for i in range(len(imgs_2)):
            imgs_2[i] = imgs_2[i][sorted_cap_indices_2]
            if cfg.CUDA:
                real_imgs_2.append(Variable(imgs_2[i]).cuda())
            else:
                real_imgs_2.append(Variable(imgs_2[i]))


        captions = captions[sorted_cap_indices].squeeze()
        
        captions_2 = captions_2[sorted_cap_indices_2].squeeze()
        # sorted_captions_lens_2 = captions_lens_2[sorted_cap_indices].squeeze()

        # captions = torch.cat([captions, captions_2], dim=0)
        # sorted_cap_lens = torch.cat([sorted_cap_lens, sorted_captions_lens_2], dim=0)


        
        class_ids_1 = class_ids[sorted_cap_indices].numpy()
        class_ids_2 = class_ids[sorted_cap_indices_2].numpy()

        
        # sent_indices = sent_indices[sorted_cap_indices]
        keys = [keys[i] for i in sorted_cap_indices.numpy()]
        # print('keys', type(keys), keys[-1])  # list
        if cfg.CUDA:
            captions = Variable(captions).cuda()
            sorted_cap_lens = Variable(sorted_cap_lens).cuda()

            captions_2 = Variable(captions_2).cuda()
            sorted_cap_lens_2 = Variable(sorted_cap_lens_2).cuda()

            sorted_cap_indices = sorted_cap_indices.cuda()
            sorted_cap_indices_2 = sorted_cap_indices_2.cuda()       
            
        else:
            captions = Variable(captions)
            sorted_cap_lens = Variable(sorted_cap_lens)

            captions_2 = Variable(captions_2)
            sorted_cap_lens_2 = Variable(sorted_cap_lens_2)

        return [real_imgs, real_imgs_2, captions, sorted_cap_lens,
                class_ids_1, keys, captions_2, sorted_cap_lens_2, class_ids_2, sorted_cap_indices, sorted_cap_indices_2]

    def build_models(self):
        # ###################encoders######################################## #
        # if cfg.TRAIN.NET_E == '':
        #     print('Error: no pretrained text-image encoders')
        #     return

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        # img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        img_encoder_path = cfg.TRAIN.CNN_ENCODER
        state_dict = \
            torch.load(img_encoder_path,
                       map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = RNN_ENCODER(
            self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.RNN_ENCODER,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.RNN_ENCODER)
        text_encoder.eval()

        # #######################generator and discriminators############## #

        # if cfg.GAN.B_DCGAN:
        # if cfg.TREE.BRANCH_NUM ==1:
        #     from model import D_NET64 as D_NET
        # elif cfg.TREE.BRANCH_NUM == 2:
        #     from model import D_NET128 as D_NET
        from utils.model import DISCRIMINATOR as D_NET
        netG = GENERATOR()
        netsD = [D_NET(b_jcu=False)]
        # else:
        #     from model import D_NET64, D_NET128, D_NET256
        #     netG = G_NET()
        #     if cfg.TREE.BRANCH_NUM > 0:
        #         netsD.append(D_NET64())
        #     if cfg.TREE.BRANCH_NUM > 1:
        #         netsD.append(D_NET128())
        #     if cfg.TREE.BRANCH_NUM > 2:
        #         netsD.append(D_NET256())

        netG.apply(weights_init)
        # print(netG)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        #
        epoch = 0
        if cfg.TRAIN.GENERATOR != '':
            state_dict = \
                torch.load(cfg.TRAIN.GENERATOR,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.GENERATOR)
            istart = cfg.TRAIN.GENERATOR.rfind('_') + 1
            iend = cfg.TRAIN.GENERATOR.rfind('.')
            epoch = cfg.TRAIN.GENERATOR[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.GENERATOR
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = \
                        torch.load(
                            Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
        return [text_encoder, image_encoder, netG, netsD, epoch]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
                   '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(),
                       '%s/netD%d.pth' % (self.model_dir, i))
        print('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, name='current'):
        # Save images
        fake_imgs, attention_maps, _, _ = netG(
            noise, sent_emb, words_embs, mask)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

    def train(self):
        """
        e.g., for epoch in range(cfg.TRAIN.MAX_EPOCH):
                  for step, data in enumerate(self.train_dataloader, 0):
                      x = self.prepare_data()
                      .....
        """
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        real_labels_2, fake_labels_2, match_labels_2 = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0

        mask = mask_correlated_samples(self)

        temperature = 0.5
        device = noise.get_device()
        criterion = NT_Xent(batch_size, temperature, mask, device)

        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0

            D_total_loss = 0
            G_total_loss = 0

            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, imgs_2, captions, cap_lens, class_ids, keys, captions_2, cap_lens_2, class_ids_2, \
                    sort_ind, sort_ind_2 = self.prepare_data(
                        data)  # [real_imgs, captions, sorted_cap_lens, class_ids, keys, sentence_idx]

                # hidden = text_encoder.init_hidden(batch_size)
                hidden = text_encoder.init_hidden(batch_size)

                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                words_embs_2, sent_emb_2 = text_encoder(
                    captions_2, cap_lens_2, hidden)
                words_embs_2, sent_emb_2 = words_embs_2.detach(), sent_emb_2.detach()
                mask_2 = (captions_2 == 0)
                num_words_2 = words_embs_2.size(2)
                if mask_2.size(1) > num_words_2:
                    mask_2 = mask_2[:, :num_words_2]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = netG(
                    noise, sent_emb, words_embs, mask)
                fake_imgs_2, _, mu_2, logvar_2 = netG(
                    noise, sent_emb_2, words_embs_2, mask_2)

                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                              sent_emb, real_labels, fake_labels)
                    errD_2 = discriminator_loss(netsD[i], imgs_2[i], fake_imgs_2[i],
                                                sent_emb_2, real_labels_2, fake_labels_2)
                    errD += errD_2

                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errG_total, G_logs, cnn_code = \
                    generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                   words_embs, sent_emb, match_labels, cap_lens, class_ids)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()

                errG_total_2, G_logs_2, cnn_code_2 = \
                    generator_loss(netsD, image_encoder, fake_imgs_2, real_labels_2,
                                   words_embs_2, sent_emb_2, match_labels_2, cap_lens_2, class_ids_2)
                kl_loss_2 = KL_loss(mu_2, logvar_2)
                errG_total_2 += kl_loss_2
                G_logs_2 += 'kl_loss: %.2f ' % kl_loss_2.item()

                errG_total += errG_total_2

                _, ori_indices = torch.sort(sort_ind, 0)
                _, ori_indices_2 = torch.sort(sort_ind_2, 0)

                total_contra_loss = 0
                i = -1
                cnn_code = cnn_code[ori_indices]
                cnn_code_2 = cnn_code_2[ori_indices_2]

                cnn_code = l2norm(cnn_code, dim=1)
                cnn_code_2 = l2norm(cnn_code_2, dim=1)

                # TODO : contrastive_loss 없이 학습시킨 결과를 report 에 포함시키기
                contrative_loss = criterion(cnn_code, cnn_code_2)
                total_contra_loss += contrative_loss * 0.2
                G_logs += 'contrative_loss: %.2f ' % total_contra_loss.item()
                errG_total += total_contra_loss
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs + '\n' + G_logs_2)
                # save images
                if gen_iterations % 1000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    # self.save_img_results(netG, fixed_noise, sent_emb,
                    #                       words_embs, mask, image_encoder,
                    #                       captions, cap_lens, epoch, name='average')
                    load_params(netG, backup_para)
                    #
                    # self.save_img_results(netG, fixed_noise, sent_emb,
                    #                       words_embs, mask, image_encoder,
                    #                       captions, cap_lens,
                    #                       epoch, name='current')
                D_total_loss += errD_total.item()
                G_total_loss += errG_total.item()

            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.item(), errG_total.item(),
                     end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, epoch)

            D_total_loss = D_total_loss / step
            G_total_loss = G_total_loss / step
            # self.writer.add_scalar('Loss_D', D_total_loss , epoch  + 1)
            # self.writer.add_scalar('Loss_G', G_total_loss , epoch  + 1)
            self.writer.add_scalars('Loss_D and Loss_G', {
                                    'Loss_D': D_total_loss, 'Loss_G': G_total_loss}, epoch + 1)

        self.writer.close()

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)
        #################################################

    def generate_eval_data(self):
        # load the text encoder model to generate images for evaluation
        self.text_encoder = RNN_ENCODER(
            self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(os.path.join(
            cfg.CHECKPOINT_DIR, cfg.TRAIN.RNN_ENCODER), map_location=lambda storage, loc: storage)
        self.text_encoder.load_state_dict(state_dict)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.RNN_ENCODER)
        self.text_encoder.eval()

        # load the generator model to generate images for evaluation
        self.netG = GENERATOR()
        state_dict = torch.load(os.path.join(
            cfg.CHECKPOINT_DIR, cfg.TRAIN.GENERATOR), map_location=lambda storage, loc: storage)
        self.netG.load_state_dict(state_dict)
        for p in self.netG.parameters():
            p.requires_grad = False
        print('Load generator from:', cfg.TRAIN.GENERATOR)
        self.netG.eval()

        noise = Variable(torch.FloatTensor(self.batch_size, cfg.GAN.Z_DIM))

        if cfg.CUDA:
            self.text_encoder = self.text_encoder.cuda()
            self.netG = self.netG.cuda()
            noise = noise.cuda()
            
        # for key in data_dic:
        for step, data in enumerate(self.test_dataloader, 0):
            # save_dir = '%s/%s' % (s_tmp, key)
            captions = data['caps']
            captions_lens = data['cap_len']
            class_ids = data['cls_id']
            keys = data['key']
            sent_idx = data['sent_ix']
            sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)

            captions = captions[sorted_cap_indices].squeeze()
            class_ids = class_ids[sorted_cap_indices].numpy()
            keys = [keys[i] for i in sorted_cap_indices.numpy()]

            if cfg.CUDA:
                captions = captions.cuda()
                sorted_cap_lens = sorted_cap_lens.cuda()
            # else:
            #     captions = Variable(captions)
            #     sorted_cap_lens = Variable(sorted_cap_lens)

            #######################################################
            # (1) Extract text embeddings
            ######################################################
            hidden = self.text_encoder.init_hidden(self.batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = self.text_encoder(captions, sorted_cap_lens, hidden)
            mask = (captions == 0)
            #######################################################
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            fake_imgs, attention_maps, _, _ = self.netG(noise, sent_emb, words_embs, mask)
            fake_imgs = fake_imgs[0]

            for j in range(self.batch_size):
                if not os.path.exists(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j].split('/')[0])):
                    os.mkdir(os.path.join(
                        cfg.TEST.GENERATED_TEST_IMAGES, keys[j].split('/')[0]))

                im = fake_imgs[j].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                print(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES,
                      keys[j] + '_{}.png'.format(sent_idx[j])))
                im.save(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES,
                        keys[j] + '_{}.png'.format(sent_idx[j])))