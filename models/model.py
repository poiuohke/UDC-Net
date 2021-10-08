import math, time
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from base import BaseModel
from utils.helpers import set_trainable
from utils.losses import *
from models.decoders import *
from models.encoder import Encoder
from torch.nn import BCEWithLogitsLoss, MSELoss


class CCT(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, cons_w_unsup=None, testing=False,
            pretrained=True, use_weak_lables=False, weakly_loss_w=0.4, cons_w_flip=None):

        # if not testing:
        #     assert (ignore_index is not None) and (sup_loss is not None) and (cons_w_unsup is not None)

        super(CCT, self).__init__()
        assert int(conf['supervised']) + int(conf['semi']) == 1, 'one mode only'
        if conf['supervised']:
            self.mode = 'supervised'
        else:
            self.mode = 'semi'

        # Supervised and unsupervised losses
        # self.ignore_index = ignore_index
        if conf['un_loss'] == "KL":
        	self.unsuper_loss = softmax_kl_loss
        elif conf['un_loss'] == "MSE":
        	self.unsuper_loss = softmax_mse_loss
        elif conf['un_loss'] == "JS":
        	self.unsuper_loss = softmax_js_loss
        else:
        	raise ValueError(f"Invalid supervised loss {conf['un_loss']}")

        if 'pseudo_label' in conf.keys():
            self.pseudo = conf['pseudo_label']
        else:
            self.pseudo = False
        self.pseudo_soft_loss = softmax_kl_loss

        if 'flip' in conf.keys():
            self.flip_flag = conf['flip']
        else:
            self.flip_flag = False
        if 'flip_only' in conf.keys():
            self.flip_only = conf['flip_only']
        else:
            self.flip_only = False
        self.flip_loss = softmax_mse_loss
        self.flip_loss_w = cons_w_flip

        self.unsup_loss_w = cons_w_unsup
        self.sup_loss_w = conf['supervised_w']
        self.softmax_temp = conf['softmax_temp']
        self.sup_loss = sup_loss
        self.sup_type = conf['sup_loss']

        # Use weak labels
        self.use_weak_lables = use_weak_lables
        self.weakly_loss_w = weakly_loss_w
        # pair wise loss (sup mat)
        self.aux_constraint = conf['aux_constraint']
        self.aux_constraint_w = conf['aux_constraint_w']
        # confidence masking (sup mat)
        self.confidence_th = conf['confidence_th']
        self.confidence_masking = conf['confidence_masking']

        # Create the model
        self.encoder = Encoder(pretrained=pretrained)

        # The main encoder
        upscale = 16
        num_out_ch = 1024
        decoder_in_ch = num_out_ch // 4
        self.main_decoder = MainDecoder(n_channels=1, n_classes=2, n_filters=16, normalization='none', has_dropout=False)



        # The auxilary decoders
        if self.mode == 'semi' or self.mode == 'weakly_semi':
            vat_decoder = [VATDecoder(upscale, decoder_in_ch, n_filters=16, n_classes=2, normalization='batchnorm', xi=conf['xi'],
            							eps=conf['eps']) for _ in range(conf['vat'])]
            drop_decoder = [DropOutDecoder(upscale, decoder_in_ch, n_filters=16, n_classes=2, normalization='batchnorm',
            							drop_rate=conf['drop_rate'])
            							for _ in range(conf['drop'])]
            cut_decoder = [CutOutDecoder(upscale, decoder_in_ch, n_filters=16, n_classes=2, normalization='batchnorm', erase=conf['erase'])
            							for _ in range(conf['cutout'])]
            context_m_decoder = [ContextMaskingDecoder(upscale, decoder_in_ch, n_filters=16, n_classes=2, normalization='batchnorm')
            							for _ in range(conf['context_masking'])]
            object_masking = [ObjectMaskingDecoder(upscale, decoder_in_ch, n_filters=16, n_classes=2, normalization='batchnorm')
            							for _ in range(conf['object_masking'])]
            feature_drop = [FeatureDropDecoder(upscale, decoder_in_ch, n_filters=16, n_classes=2, normalization='batchnorm')
            							for _ in range(conf['feature_drop'])]
            feature_noise = [FeatureNoiseDecoder(upscale, decoder_in_ch, n_filters=16, n_classes=2, normalization='batchnorm',
            							uniform_range=conf['uniform_range'])
            							for _ in range(conf['feature_noise'])]

            self.aux_decoders = nn.ModuleList([ *vat_decoder, *drop_decoder, *cut_decoder,
                                    *context_m_decoder, *object_masking, *feature_drop, *feature_noise])#*vat_decoder,



    def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None, x_l_flip=None, target_l_flip=None, x_ul_flip=None,
                flip_index=None, curr_iter=None, epoch=None, mode_D=None):
        if not self.training:
            return self.main_decoder(self.encoder(x_l))
        if mode_D:
            x_all = torch.cat([x_l, x_ul])
            return self.main_decoder(self.encoder(x_all))

        # We compute the losses in the forward pass to avoid problems encountered in muti-gpu 

        # Forward pass the labels example
        input_size = (x_l.size(2), x_l.size(3), x_l.size(4))
        temp_output = self.encoder(x_l)
        output_l = self.main_decoder(temp_output)

        # Supervised loss
        if self.sup_type == 'CE':
            loss_sup = self.sup_loss(output_l, target_l, temperature=self.softmax_temp) * self.sup_loss_w
        elif self.sup_type == 'multi':
            ce_loss = self.sup_loss[0](output_l, target_l, temperature=self.softmax_temp)
            output_soft_l = F.softmax(output_l, dim=1)
            dc_loss = self.sup_loss[1](output_soft_l[:,1,:,:,:], target_l==1)
            print ('')
            print ('dice_loss',dc_loss)
            loss_sup = dc_loss
        elif self.sup_type == 'dice':
            output_soft_l = F.softmax(output_l, dim=1)
            loss_sup = self.sup_loss(output_soft_l[:,1,:,:,:], target_l==1)
        else:
            loss_sup = self.sup_loss(output_l, target_l, curr_iter=curr_iter, epoch=epoch) * self.sup_loss_w

        # If supervised mode only, return
        if self.mode == 'supervised':
            curr_losses = {'loss_sup': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup #+loss_sdf*0.5
            return total_loss, curr_losses, outputs

        # If semi supervised mode
        elif self.mode == 'semi':
            # Get main prediction
            x_org = x_ul
            x_ul = self.encoder(x_ul)
            output_ul = self.main_decoder(x_ul)
            curr_losses = {'loss_sup': loss_sup}

            if self.flip_only:
                x_l_flip = self.encoder(x_l_flip)
                output_l_flip = self.main_decoder(x_l_flip)
                output_l_flip = self.flip(output_l_flip, flip_index)

                target_l_flip = F.softmax(output_l.detach(), dim=1)
                loss_l_flip = self.flip_loss(output_l_flip, target_l_flip, use_softmax=False)
                # print(loss_l_flip)
                if self.mode == 'semi':
                    x_ul_flip = self.encoder(x_ul_flip)
                    output_ul_flip = self.main_decoder(x_ul_flip)
                    output_ul_flip = self.flip(output_ul_flip, flip_index)

                    target_ul_flip = F.softmax(output_ul.detach(), dim=1)
                    loss_ul_flip = self.flip_loss(output_ul_flip, target_ul_flip, use_softmax=False)
                    # print(loss_ul_flip)

                    loss_flip = (loss_l_flip + loss_ul_flip) / 2
                else:
                    loss_flip = loss_l_flip
                outputs = {'sup_pred': output_l, 'unsup_pred': output_ul}
                weight_f = self.flip_loss_w(epoch=epoch,
                                            curr_iter=curr_iter)  # weight_u / self.unsup_loss_w.final_w) * self.flip_loss_w

                loss_flip = loss_flip * weight_f
                curr_losses['loss_flip'] = loss_flip
                total_loss = loss_flip + loss_sup
                return total_loss, curr_losses, outputs

            # print('aux decoder')
            # Get auxiliary predictions
            outputs_ul = [aux_decoder(x_ul, output_ul.detach()) for aux_decoder in self.aux_decoders]

            # cal uncertainty_map #############################################################
            outputs_targets_ul = [F.softmax(u.detach(),dim=1) for u in outputs_ul]
            outputs_targets_ul.append(F.softmax(output_ul.detach(), dim=1))
            mean_outputs_targets_ul = torch.mean(torch.stack(outputs_targets_ul), 0)

            MSE_func = nn.MSELoss(reduce=False)
            MSE_outputs_targets_ul = [MSE_func(u, mean_outputs_targets_ul) for u in outputs_targets_ul]
            MSE_outputs_targets_ul.append(MSE_func(input=F.softmax(output_ul.detach(),dim=1), target=mean_outputs_targets_ul))
            mean_MSE_outputs_targets_ul = torch.mean(torch.stack(MSE_outputs_targets_ul), 0)

            targets = F.softmax(output_ul.detach(), dim=1)

            # Compute unsupervised loss
            loss_unsup = sum([self.unsuper_loss(inputs=u, targets=targets, uncertainty_map_mean = mean_outputs_targets_ul,
                                                uncertainty_map_mse = mean_MSE_outputs_targets_ul, \
                            conf_mask=self.confidence_masking, threshold=self.confidence_th, use_softmax=False)#
                            for u in outputs_ul])
            loss_unsup = (loss_unsup / len(outputs_ul))

            outputs = {'sup_pred': output_l, 'unsup_pred': output_ul}

            # Compute the unsupervised loss
            weight_u = self.unsup_loss_w(epoch=epoch, curr_iter=curr_iter)
            loss_unsup = loss_unsup * weight_u
            curr_losses['loss_unsup'] = loss_unsup
            total_loss = loss_unsup  + loss_sup

            ##################################  online pseudo label ###############################################
            if self.pseudo:
                weight_p = (weight_u / self.unsup_loss_w.final_w) * self.weakly_loss_w
                loss_pseudo  = self.pseudo_soft_loss(inputs=output_ul, targets=mean_outputs_targets_ul, uncertainty_map_mse=mean_MSE_outputs_targets_ul,
                                                     conf_mask=self.confidence_masking, threshold=0.12, use_softmax=False)
                # print (loss_pseudo)
                loss_pseudo = loss_pseudo * weight_p
                curr_losses['loss_pseudo'] = loss_pseudo
                total_loss += loss_pseudo
            ###########################################
            # If case we're using weak lables, add the weak loss term with a weight (self.weakly_loss_w)
            if self.use_weak_lables:
                weight_w = (weight_u / self.unsup_loss_w.final_w) * self.weakly_loss_w
                loss_weakly = sum([self.sup_loss(outp, target_ul) for outp in outputs_ul]) / len(outputs_ul)
                loss_weakly = loss_weakly * weight_w
                curr_losses['loss_weakly'] = loss_weakly
                total_loss += loss_weakly

            # Pair-wise loss
            if self.aux_constraint:
                pair_wise = pair_wise_loss(outputs_ul) * self.aux_constraint_w
                curr_losses['pair_wise'] = pair_wise
                loss_unsup += pair_wise

        if self.flip_flag:
            x_l_flip = self.encoder(x_l_flip)
            output_l_flip = self.main_decoder(x_l_flip)
            output_l_flip = self.flip(output_l_flip, flip_index)

            target_l_flip = F.softmax(output_l.detach(), dim=1)
            loss_l_flip = self.flip_loss(output_l_flip, target_l_flip, use_softmax=False)
            # print(loss_l_flip)
            if self.mode == 'semi':
                x_ul_flip = self.encoder(x_ul_flip)
                output_ul_flip = self.main_decoder(x_ul_flip)
                output_ul_flip = self.flip(output_ul_flip, flip_index)

                target_ul_flip = F.softmax(output_ul.detach(), dim=1)
                loss_ul_flip = self.flip_loss(output_ul_flip, target_ul_flip, use_softmax=False)
                # print(loss_ul_flip)

                loss_flip = (loss_l_flip + loss_ul_flip)/2
            else:
                loss_flip = loss_l_flip

            # print(loss_flip)
            weight_f = self.flip_loss_w(epoch=epoch, curr_iter=curr_iter)

            loss_flip = loss_flip * weight_f
            curr_losses['loss_flip'] = loss_flip
            total_loss += loss_flip


        return total_loss, curr_losses, outputs

    def get_backbone_params(self):
        return self.encoder.parameters()

    def get_other_params(self):
        if self.mode == 'semi':
            return chain( self.main_decoder.parameters(),
                        self.aux_decoders.parameters())

        return chain(self.main_decoder.parameters())

    def flip(self, x, dim):
        dim = x.dim() + dim if dim < 0 else dim
        return x[tuple(slice(None, None) if i != dim
                       else torch.arange(x.size(i) - 1, -1, -1).long()
                       for i in range(x.dim()))]
