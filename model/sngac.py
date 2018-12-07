# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
GRAYSCALE_AVG = 127.5
TINIEST_LR = 0.000125

import matplotlib.pyplot as plt



import sys
sys.path.append('..')

import copy as cp
import random as rd

import tensorflow as tf
import numpy as np
import random as rnd
import scipy.misc as misc
import os
import shutil
import time
from collections import namedtuple
from dataset.dataset import DataProvider

from utilities.utils import scale_back_for_img, scale_back_for_dif, merge, correct_ckpt_path
from utilities.utils import image_show
from model.gan_networks import discriminator_mdy_5_convs
from model.gan_networks import discriminator_mdy_6_convs
from model.gan_networks import discriminator_mdy_6_convs_tower_version1


from model.gan_networks import vgg_16_net as feature_ebddactor_network


from model.gan_networks import generator_framework as generator_implementation
from model.gan_networks import generator_inferring
from model.gan_networks import encoder_framework as encoder_implementation


import math



# Auxiliary wrapper classes
# Used to save handles(important nodes in computation graph) for later evaluation
SummaryHandle = namedtuple("SummaryHandle", ["d_merged", "g_merged",
                                             "check_validate_image_summary", "check_train_image_summary",
                                             "check_validate_image", "check_train_image",
                                             "learning_rate",
                                             "trn_real_dis_ebdd_summaries","val_real_dis_ebdd_summaries",
                                             "trn_fake_dis_ebdd_summaries","val_fake_dis_ebdd_summaries"])

EvalHandle = namedtuple("EvalHandle",["inferring_generated_images","training_generated_images",
                                      "inferring_discriminator_categorical_logits"])

GeneratorHandle = namedtuple("Generator",
                             ["generated_target_train","generated_target_infer"])

DiscriminatorHandle = namedtuple("Discriminator",
                                 ["current_critic_logit_penalty","infer_label0",
                                  "content_infer","style_infer","infer_categorical_logits"])

FeatureExtractorHandle = namedtuple("FeatureExtractor",
                                    ["infer_input_img","true_label0","true_label1"])



discriminator_dict = {"DisMdy5conv": discriminator_mdy_5_convs,
                      "DisMdy6conv": discriminator_mdy_6_convs,
                      "DisMdy6conv-TowerVersion1": discriminator_mdy_6_convs_tower_version1}

eps = 1e-9

class SnGac(object):

    # constructor
    def __init__(self,
                 debug_mode=-1,
                 training_mode = '',
                 print_info_seconds=-1,
                 train_data_augment=-1,
                 init_training_epochs=-1,
                 final_training_epochs=-1,

                 experiment_dir='/tmp/',
                 log_dir='/tmp/',
                 experiment_id='0',
                 content_data_dir='/tmp/',
                 style_train_data_dir='/tmp/',
                 style_validation_data_dir='/tmp/',
                 training_from_model=None,
                 file_list_txt_content=None,
                 file_list_txt_style_train=None,
                 file_list_txt_style_validation=None,
                 channels=-1,
                 epoch=-1,

                 optimization_method='adam',

                 batch_size=8, img_width=256,
                 lr=0.001, final_learning_rate_pctg=0.2,


                 Pixel_Reconstruction_Penalty=100,
                 Lconst_style_Penalty=15,
                 Discriminative_Penalty=1,
                 Discriminator_Categorical_Penalty=1,
                 Discriminator_Gradient_Penalty=1,



                 generator_weight_decay_penalty = 0.001,
                 discriminator_weight_decay_penalty = 0.004,

                 resume_training=0,

                 generator_devices='/device:CPU:0',
                 discriminator_devices='/device:CPU:0',
                 style_embedder_devices='/device:CPU:0',

                 generator_residual_at_layer=3,
                 generator_residual_blocks=5,
                 discriminator='DisMdy6conv',
                 style_embedder_dir='/tmp/',

                 ## for infer only
                 model_dir='./', save_path='./',

                 # for styleadd infer only
                 targeted_content_input_txt='./',
                 target_file_path='-1',
                 save_mode='-1',
                 known_style_img_path='./',


                 ):

        self.initializer = 'XavierInit'

        self.print_info_seconds=print_info_seconds
        self.discriminator_initialization_iters=25
        self.init_training_epochs=init_training_epochs
        self.final_training_epochs=final_training_epochs
        self.model_save_epochs=3
        self.debug_mode = debug_mode
        self.training_mode = training_mode
        self.experiment_dir = experiment_dir
        self.log_dir=log_dir
        self.experiment_id = experiment_id
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoint")
        self.training_from_model = training_from_model
        self.style_embedder_dir = style_embedder_dir
        self.epoch=epoch

        self.inf_data_dir = os.path.join(self.experiment_dir, "infs")
        self.img2img_width = img_width
        self.source_img_width = img_width


        self.content_data_dir = content_data_dir
        self.style_train_data_dir = style_train_data_dir
        self.style_validation_data_dir = style_validation_data_dir
        self.file_list_txt_content = file_list_txt_content
        self.file_list_txt_style_train = file_list_txt_style_train
        self.file_list_txt_style_validation = file_list_txt_style_validation
        self.input_output_img_filter_num = channels


        self.optimization_method = optimization_method
        self.batch_size = batch_size
        self.final_learning_rate_pctg = final_learning_rate_pctg


        self.resume_training = resume_training

        self.train_data_augment = (train_data_augment==1)


        self.Discriminative_Penalty = Discriminative_Penalty + eps
        self.Discriminator_Gradient_Penalty = Discriminator_Gradient_Penalty + eps
        self.Pixel_Reconstruction_Penalty = Pixel_Reconstruction_Penalty + eps
        self.Lconst_style_Penalty = Lconst_style_Penalty + eps
        self.lr = lr
        # if self.training_mode == 'DiscriminatorFineTune':
        #     self.lr = self.lr / 2


        self.Discriminator_Categorical_Penalty = Discriminator_Categorical_Penalty + eps
        self.generator_weight_decay_penalty = generator_weight_decay_penalty + eps
        self.discriminator_weight_decay_penalty = discriminator_weight_decay_penalty + eps

        if self.generator_weight_decay_penalty > 10 * eps:
            self.weight_decay_generator = True
        else:
            self.weight_decay_generator = False
        if self.discriminator_weight_decay_penalty > 10 * eps:
            self.weight_decay_discriminator = True
        else:
            self.weight_decay_discriminator = False

        self.generator_devices = generator_devices
        self.discriminator_devices = discriminator_devices
        self.style_embedder_device=style_embedder_devices

        self.generator_residual_at_layer = generator_residual_at_layer
        self.generator_residual_blocks = generator_residual_blocks
        self.discriminator = discriminator



        self.discriminator_implementation = discriminator_dict[self.discriminator]

        self.accuracy_k=[1,3,5,10,20,50]

        # properties for inferring
        self.model_dir=model_dir
        self.save_path=save_path
        if os.path.exists(self.save_path) and not (self.save_path == './'):
            shutil.rmtree(self.save_path)
        if not self.save_path == './':
            os.makedirs(self.save_path)

        # for styleadd infer only
        self.targeted_content_input_txt=targeted_content_input_txt
        self.target_file_path=target_file_path
        self.save_mode=save_mode
        self.known_style_img_path=known_style_img_path





        # init all the directories
        self.sess = None
        self.print_separater = "#################################################################"


        if self.training_from_model==None and \
                (self.training_mode == 'DiscriminatorFineTune' or self.training_mode == 'DiscriminatorReTrain') \
                and self.resume_training==0:
            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)
            print("ERROR!!! FineTune or ReTrain Discriminator without loaded initial model!!!")
            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)
            return



    def find_bn_avg_var(self,var_list):
        var_list_new = list()
        for ii in var_list:
            var_list_new.append(ii)

        all_vars = tf.global_variables()
        bn_var_list = [var for var in var_list if 'bn' in var.name]
        output_avg_var = list()
        for bn_var in bn_var_list:
            if 'gamma' in bn_var.name:
                continue
            bn_var_name = bn_var.name
            variance_name = bn_var_name.replace('beta', 'moving_variance')
            average_name = bn_var_name.replace('beta', 'moving_mean')
            variance_var = [var for var in all_vars if variance_name in var.name][0]
            average_var = [var for var in all_vars if average_name in var.name][0]
            output_avg_var.append(variance_var)
            output_avg_var.append(average_var)

        var_list_new.extend(output_avg_var)

        output = list()
        for ii in var_list_new:
            if ii not in output:
                output.append(ii)

        return output

    def variable_dict(self,var_input,delete_name_from_character):
        var_output = {}
        for ii in var_input:
            prefix_pos = ii.name.find(delete_name_from_character)
            renamed = ii.name[prefix_pos + 1:]
            parafix_pos = renamed.find(':')
            renamed = renamed[0:parafix_pos]
            var_output.update({renamed: ii})
        return var_output

    def get_model_id_and_dir_for_train(self):
        if (not self.training_mode == 'DiscriminatorFineTune') \
                and (not self.training_mode == 'GeneratorInit') \
                and (not self.training_mode == 'DiscriminatorReTrain') :
            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)
            print("TrainingMode Setting Error:%s" % self.training_mode)
            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)
            return
        encoder_decoder_layer_num = int(np.floor(math.log(self.img2img_width) / math.log(2)))
        model_id = "Exp%s-%s-GenEncDec%d-Res%d@Lyr%d_%s" % \
                   (self.experiment_id,
                    self.training_mode,
                    encoder_decoder_layer_num,
                    self.generator_residual_blocks,
                    self.generator_residual_at_layer,
                    self.discriminator)



        model_ckpt_dir = os.path.join(self.checkpoint_dir, model_id)
        model_log_dir = os.path.join(self.log_dir, model_id)
        model_save_path = os.path.join(self.inf_data_dir, model_id)
        return model_id, model_ckpt_dir, model_log_dir, model_save_path

    def checkpoint(self, saver, model_dir,global_step):
        model_name = "img2img.model"
        step = global_step.eval(session=self.sess)
        if step==0:
            step=1
        print(os.path.join(model_dir, model_name))
        saver.save(self.sess, os.path.join(model_dir, model_name), global_step=int(step))

    def restore_model(self, saver, model_dir, model_name):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        corrected_ckpt_path = correct_ckpt_path(real_dir=model_dir,
                                                maybe_path=ckpt.model_checkpoint_path)
        if ckpt:
            saver.restore(self.sess, corrected_ckpt_path)
            print("ModelRestored:%s" % model_name)
            print("@%s" % model_dir)
            print(self.print_separater)
            return True
        else:
            print("fail to restore model %s" % model_dir)
            print(self.print_separater)
            return False

    def generate_fake_samples(self,training_mark,current_iterator,
                              train_img_infer_list=list()):

        evalHandle = getattr(self, "eval_handle")

        if training_mark:
            fake_images,\
            content_prototypes,style_references,\
            label0_onehot, label1_onehot,\
            label0_dense,label1_dense,\
            output_training_img_list\
                = self.sess.run([evalHandle.training_generated_images,
                                 current_iterator.output_tensor_list[0],
                                 current_iterator.output_tensor_list[1],
                                 current_iterator.output_tensor_list[2],
                                 current_iterator.output_tensor_list[3],
                                 current_iterator.output_tensor_list[4],
                                 current_iterator.output_tensor_list[5],
                                 train_img_infer_list])
        else:
            embedder_handle = getattr(self,"embedder_handle")
            fake_images, \
            content_prototypes, style_references, \
            label0_onehot, label1_onehot, \
            label0_dense, label1_dense\
                = self.sess.run([evalHandle.inferring_generated_images,
                                 current_iterator.output_tensor_list[0],
                                 current_iterator.output_tensor_list[1],
                                 current_iterator.output_tensor_list[2],
                                 current_iterator.output_tensor_list[3],
                                 current_iterator.output_tensor_list[4],
                                 current_iterator.output_tensor_list[5]])
            output_training_img_list=list()

        return fake_images,\
               content_prototypes, style_references, output_training_img_list, \
               label0_onehot, label1_onehot, label0_dense, label1_dense


    def discriminate_sample(self, content_infer_batch, style_infer_batch):
        evalHandle = getattr(self, "eval_handle")
        discriminator_handle = getattr(self,"discriminator_handle")
        categorical_logits = \
            self.sess.run(evalHandle.inferring_discriminator_categorical_logits,
                          feed_dict={discriminator_handle.content_infer: content_infer_batch,
                                     discriminator_handle.style_infer: style_infer_batch})
        return categorical_logits


    def validate_model(self,
                       train_mark,
                       summary_writer, global_step,
                       data_provider,
                       discriminator_handle,generator_handle,embedder_handle):
        summary_handle = getattr(self,"summary_handle")

        if train_mark:
            merged_real_dis_ebdd_summaries = summary_handle.trn_real_dis_ebdd_summaries
            merged_fake_dis_ebdd_summaries = summary_handle.trn_fake_dis_ebdd_summaries
            check_image = summary_handle.check_train_image_summary
            check_image_input = summary_handle.check_train_image
            current_iterator = data_provider.train_iterator
            # batch_true_style, \
            # batch_prototype, batch_reference, \
            # batch_label0_onehot, batch_label1_onehot, \
            # batch_label0_dense, batch_label1_dense = \
            #     current_iterator.get_next_batch(sess=self.sess)

            train_img_infer_list = list()
            train_img_infer_list.append(generator_handle.generated_target_train)


        else:
            merged_real_dis_ebdd_summaries = summary_handle.val_real_dis_ebdd_summaries
            merged_fake_dis_ebdd_summaries = summary_handle.val_fake_dis_ebdd_summaries
            check_image = summary_handle.check_validate_image_summary
            check_image_input = summary_handle.check_validate_image
            current_iterator = data_provider.validate_iterator
            # batch_true_style, \
            # batch_prototype, batch_reference, \
            # batch_label0_onehot, batch_label1_onehot, \
            # batch_label0_dense, batch_label1_dense = \
            #     current_iterator.get_next_batch(sess=self.sess)
            train_img_infer_list = list()




        generated_content_batch, \
        true_content_batch,style_batch,\
        training_img_list,\
        label0_onehot,label1_onehot,\
        label0_dense,label1_dense,\
            = self.generate_fake_samples(training_mark=train_mark,
                                         current_iterator=current_iterator,
                                         train_img_infer_list=train_img_infer_list)

        summary_fake_output = self.sess.run(merged_fake_dis_ebdd_summaries,
                                            feed_dict={embedder_handle.infer_input_img: style_batch,
                                                       embedder_handle.true_label0: label0_onehot,
                                                       embedder_handle.true_label1: label1_onehot,
                                                       discriminator_handle.infer_label0: label0_onehot,
                                                       discriminator_handle.content_infer: generated_content_batch,
                                                       discriminator_handle.style_infer: style_batch})


        summary_real_output = self.sess.run(merged_real_dis_ebdd_summaries,
                                            feed_dict={embedder_handle.infer_input_img:style_batch,
                                                       embedder_handle.true_label0: label0_onehot,
                                                       embedder_handle.true_label1: label1_onehot,
                                                       discriminator_handle.infer_label0:label0_onehot,
                                                       discriminator_handle.content_infer:true_content_batch,
                                                       discriminator_handle.style_infer:style_batch})

        generated_content_batch = scale_back_for_img(images=generated_content_batch)
        true_content_batch = scale_back_for_img(images=true_content_batch)
        style_batch = scale_back_for_img(images=style_batch)
        diff_between_generated_and_true = scale_back_for_dif(generated_content_batch - true_content_batch)

        generated_content_batch = merge(generated_content_batch, [self.batch_size, 1])
        true_content_batch = merge(true_content_batch, [self.batch_size, 1])
        style_batch = merge(style_batch, [self.batch_size, 1])
        diff_between_generated_and_true = merge(diff_between_generated_and_true, [self.batch_size, 1])




        if train_mark:
            generated_train_content_batch = scale_back_for_img(images=training_img_list[0])

            generated_train_content_batch = merge(generated_train_content_batch,[self.batch_size,1])
            diff_between_generated_train_and_true = scale_back_for_dif(generated_train_content_batch - true_content_batch)

            merged_disp = np.concatenate([style_batch,
                                          generated_content_batch,
                                          diff_between_generated_and_true,
                                          generated_train_content_batch,
                                          diff_between_generated_train_and_true,
                                          true_content_batch], axis=1)

        else:
            merged_disp = np.concatenate([style_batch,
                                          generated_content_batch,
                                          diff_between_generated_and_true,
                                          true_content_batch], axis=1)


        summray_img = self.sess.run(check_image,
                                    feed_dict={check_image_input:
                                                   np.reshape(merged_disp, (1, merged_disp.shape[0],
                                                                            merged_disp.shape[1],
                                                                            merged_disp.shape[2]))})
        summary_writer.add_summary(summray_img, global_step.eval(session=self.sess))

        if self.debug_mode==1 or \
                ((self.debug_mode==0)
                 and (global_step.eval(session=self.sess)>=2500)
                 or (self.training_mode == 'FineTuneDiscriminator'
                     or self.training_mode == 'DiscriminatorReTrain')):
            summary_writer.add_summary(summary_real_output, global_step.eval(session=self.sess))
            summary_writer.add_summary(summary_fake_output, global_step.eval(session=self.sess))








    def summary_finalization(self,
                             g_loss_summary,
                             d_loss_summary,
                             trn_real_dis_ebdd_summaries, val_real_dis_ebdd_summaries,
                             trn_fake_dis_ebdd_summaries, val_fake_dis_ebdd_summaries,
                             learning_rate):

        check_train_image = tf.placeholder(tf.float32, [1, self.batch_size * self.img2img_width,
                                                        self.img2img_width * 6,
                                                        3])

        check_validate_image = tf.placeholder(tf.float32, [1, self.batch_size * self.img2img_width,
                                                           self.img2img_width * 4,
                                                           3])



        check_train_image_summary = tf.summary.image('TrnImg', check_train_image)
        check_validate_image_summary = tf.summary.image('ValImg', check_validate_image)


        learning_rate_summary = tf.summary.scalar('LearningRate', learning_rate)

        summary_handle = SummaryHandle(d_merged=d_loss_summary,
                                       g_merged=g_loss_summary,
                                       check_validate_image_summary=check_validate_image_summary,
                                       check_train_image_summary=check_train_image_summary,
                                       check_validate_image=check_validate_image,
                                       check_train_image=check_train_image,
                                       learning_rate=learning_rate_summary,
                                       trn_real_dis_ebdd_summaries=trn_real_dis_ebdd_summaries,
                                       val_real_dis_ebdd_summaries=val_real_dis_ebdd_summaries,
                                       trn_fake_dis_ebdd_summaries=trn_fake_dis_ebdd_summaries,
                                       val_fake_dis_ebdd_summaries=val_fake_dis_ebdd_summaries)
        setattr(self, "summary_handle", summary_handle)


    def framework_building(self):
        # for model base frameworks
        with tf.device('/device:CPU:0'):
            global_step = tf.get_variable('global_step',
                                          [],
                                          initializer=tf.constant_initializer(0),
                                          trainable=False,
                                          dtype=tf.int32)
            epoch_step = tf.get_variable('epoch_step',
                                         [],
                                         initializer=tf.constant_initializer(0),
                                         trainable=False,
                                         dtype=tf.int32)
            epoch_step_increase_one_op = tf.assign(epoch_step, epoch_step + 1)
            learning_rate = tf.placeholder(tf.float32, name="learning_rate")

            framework_var_list = list()
            framework_var_list.append(global_step)
            framework_var_list.append(epoch_step)

        saver_frameworks = tf.train.Saver(max_to_keep=self.model_save_epochs, var_list=framework_var_list)


        print("Framework built @%s." % '/device:CPU:0')
        return epoch_step_increase_one_op, learning_rate, global_step, epoch_step, saver_frameworks



    def embedder_build(self,  data_provider):


        def build_embedder(input_target_infer,
                           label0_length, label1_length):

            with tf.variable_scope(tf.get_variable_scope()):
                with tf.device(self.style_embedder_device):

		    #print(data_provider.train_iterator.output_tensor_list[1].shape)
                    train_label1_logits, train_label0_logits, network_info = \
                        feature_ebddactor_network(image=data_provider.train_iterator.output_tensor_list[1],
                                                  batch_size=self.batch_size,
                                                  device=self.style_embedder_device,
                                                  label0_length=label0_length,
                                                  label1_length=label1_length,
                                                  reuse=False,
                                                  keep_prob=1,
                                                  initializer=self.initializer,
                                                  network_usage='embedder')


                    infer_label1_logits_for_generation, infer_label0_logits_for_generation, _ = \
                                            feature_ebddactor_network(image=data_provider.validate_iterator.output_tensor_list[1],
                                                                      batch_size=self.batch_size,
                                                                      device=self.style_embedder_device,
                                                                      label0_length=label0_length,
                                                                      label1_length=label1_length,
                                                                      reuse=True,
                                                                      keep_prob=1,
                                                                      initializer=self.initializer,
                                                                      network_usage='embedder')

                    infer_label1_logits_for_validation, infer_label0_logits_validation, _ = \
                        feature_ebddactor_network(image=input_target_infer,
                                                  batch_size=self.batch_size,
                                                  device=self.style_embedder_device,
                                                  label0_length=label0_length,
                                                  label1_length=label1_length,
                                                  reuse=True,
                                                  keep_prob=1,
                                                  initializer=self.initializer,
                                                  network_usage='embedder')


            return train_label0_logits, train_label1_logits, \
                   infer_label1_logits_for_generation, infer_label0_logits_for_generation,\
                   infer_label1_logits_for_validation, infer_label0_logits_validation,\
                   network_info

        def define_entropy_accuracy_calculation_op(true_labels, infer_logits, summary_name):
            ebdd_prdt = tf.argmax(infer_logits, axis=1)
            ebdd_true = tf.argmax(true_labels, axis=1)

            correct = tf.equal(ebdd_prdt, ebdd_true)
            accuarcy = tf.reduce_mean(tf.cast(correct, tf.float32)) * 100

            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=infer_logits, labels=tf.nn.softmax(infer_logits))
            entropy = tf.reduce_mean(entropy)

            trn_acry_real = tf.summary.scalar("Accuracy_" + summary_name + "/Train", accuarcy)
            trn_enpy_real = tf.summary.scalar("Entropy_" + summary_name + "/Train", entropy)
            val_acry_real = tf.summary.scalar("Accuracy_" + summary_name + "/Test", accuarcy)
            val_enpy_real = tf.summary.scalar("Entropy_" + summary_name + "/Test", entropy)

            trn_merged = tf.summary.merge([trn_acry_real, trn_enpy_real])
            val_merged = tf.summary.merge([val_acry_real, val_enpy_real])

            return trn_merged, val_merged

        ebdd_trn_real_merged = []
        ebdd_trn_fake_merged = []
        ebdd_val_real_merged = []
        ebdd_val_fake_merged = []

        input_target_infer = tf.placeholder(tf.float32,
                                            [self.batch_size,
                                             self.source_img_width,
                                             self.source_img_width,
                                             self.input_output_img_filter_num],
                                            name='embedder_infer_img_input')
        true_label0 = tf.placeholder(tf.float32, [self.batch_size, len(self.involved_label0_list)])
        true_label1 = tf.placeholder(tf.float32, [self.batch_size, len(self.involved_label1_list)])

        train_label0_logits, train_label1_logits, \
        infer_label1_logits_for_generation, infer_label0_logits_for_generation,\
        infer_label1_logits_for_validation, infer_label0_logits_for_validation,\
        network_info = \
            build_embedder(input_target_infer=input_target_infer,
                           label0_length=len(self.involved_label0_list),
                           label1_length=len(self.involved_label1_list))

        ebdd_vars_true_fake = [var for var in tf.trainable_variables() if 'embedder' in var.name]
        ebdd_vars_true_fake = self.find_bn_avg_var(ebdd_vars_true_fake)
        ebdd_vars_true_fake = self.variable_dict(var_input=ebdd_vars_true_fake, delete_name_from_character='/')
        saver_ebddactor_true_fake = tf.train.Saver(max_to_keep=1, var_list=ebdd_vars_true_fake)


        summary_train_merged_label0,\
        summary_val_merged_label0 =  \
            define_entropy_accuracy_calculation_op(true_labels=true_label0,
                                                   infer_logits=infer_label0_logits_for_validation,
                                                   summary_name="Embedder/Lb0")

        summary_train_merged_label1,\
        summary_val_merged_label1 = \
            define_entropy_accuracy_calculation_op(true_labels=true_label1,
                                                   infer_logits=infer_label1_logits_for_validation,
                                                   summary_name="Embedder/Lb1")

        ebdd_trn_real_merged = tf.summary.merge(
            [ebdd_trn_real_merged, summary_train_merged_label0, summary_train_merged_label1])
        ebdd_trn_fake_merged = tf.summary.merge([ebdd_trn_fake_merged])
        ebdd_val_real_merged = tf.summary.merge(
            [ebdd_val_real_merged, summary_val_merged_label0, summary_val_merged_label1])
        ebdd_val_fake_merged = tf.summary.merge([ebdd_val_fake_merged])

        print("TrueFakeExtractor @ %s with %s;" % (self.style_embedder_device, network_info))

        embedder_handle = FeatureExtractorHandle(infer_input_img=input_target_infer,
                                                 true_label0=true_label0,
                                                 true_label1=true_label1)
        setattr(self, "embedder_handle", embedder_handle)



        output_embedder_train_logits = tf.concat([train_label0_logits, train_label1_logits], axis=1)
        output_embedder_infer_logits_for_validation = tf.concat([infer_label0_logits_for_validation,
                                                                 infer_label1_logits_for_validation], axis=1)
        output_embedder_infer_logits_for_generation = tf.concat([infer_label0_logits_for_generation,
                                                                 infer_label1_logits_for_generation], axis=1)
        return saver_ebddactor_true_fake,\
               output_embedder_train_logits,output_embedder_infer_logits_for_generation,output_embedder_infer_logits_for_validation,\
               ebdd_trn_real_merged,ebdd_trn_fake_merged,ebdd_val_real_merged,ebdd_val_fake_merged



    def generator_build(self,data_provider, embedder_train_logits, embedder_infer_logits):

        name_prefix = 'generator'

        # network architechture
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device(self.generator_devices):


                content_prototype_train = data_provider.train_iterator.output_tensor_list[0]
                style_reference_train = data_provider.train_iterator.output_tensor_list[1]

                # build the generator
                generated_target_train, encoded_style_reference_train,\
                network_info = \
                    generator_implementation(style_reference=style_reference_train,
                                             ebdd_logits=embedder_train_logits,
                                             is_training=True,
                                             batch_size=self.batch_size,
                                             generator_device=self.generator_devices,
                                             residual_at_layer=self.generator_residual_at_layer,
                                             residual_block_num=self.generator_residual_blocks,
                                             scope=name_prefix,
                                             reuse=False,
                                             initializer=self.initializer,
                                             weight_decay=self.weight_decay_generator,
                                             weight_decay_rate=self.generator_weight_decay_penalty)


                # encoded of the generated target on the style reference encoder
                encoded_style_reference_generated_target = \
                        encoder_implementation(images=generated_target_train,
                                               is_training=True,
                                               encoder_device=self.generator_devices,
                                               residual_at_layer=self.generator_residual_at_layer,
                                               residual_connection_mode='Multi',
                                               scope=name_prefix + '/style_encoder',
                                               reuse=True,
                                               initializer=self.initializer,
                                               weight_decay=False,
                                               weight_decay_rate=self.generator_weight_decay_penalty)[0]


                # for inferring
                style_reference_infer = data_provider.validate_iterator.output_tensor_list[1]
                generated_target_infer = \
                    generator_implementation(style_reference=style_reference_infer,
                                             is_training=False,
                                             batch_size=self.batch_size,
                                             generator_device=self.generator_devices,
                                             residual_at_layer=self.generator_residual_at_layer,
                                             residual_block_num=self.generator_residual_blocks,
                                             scope=name_prefix,
                                             reuse=True,
                                             initializer=self.initializer,
                                             weight_decay=False,
                                             weight_decay_rate=eps,
                                             ebdd_logits=embedder_infer_logits)[0]

                curt_generator_handle = GeneratorHandle(generated_target_train=generated_target_train,
                                                        generated_target_infer=generated_target_infer)
                setattr(self, "generator_handle", curt_generator_handle)



        # loss build
        g_loss=0
        g_merged_summary = []

        # weight_decay_loss
        generator_weight_decay_loss = tf.get_collection('generator_weight_decay')
        weight_decay_loss = 0
        if generator_weight_decay_loss:
            for ii in generator_weight_decay_loss:
                weight_decay_loss = ii + weight_decay_loss
            weight_decay_loss = weight_decay_loss / len(generator_weight_decay_loss)
            generator_weight_decay_loss_summary = tf.summary.scalar("Loss_Generator/WeightDecay",
                                                                    tf.abs(weight_decay_loss)/self.generator_weight_decay_penalty)
            g_loss += weight_decay_loss
            g_merged_summary = tf.summary.merge([g_merged_summary, generator_weight_decay_loss_summary])


        # const loss for both source and real target
        if self.Lconst_style_Penalty > eps * 10:
            current_const_loss_style = tf.square(encoded_style_reference_generated_target - encoded_style_reference_train)
            current_const_loss_style = tf.reduce_mean(current_const_loss_style) * self.Lconst_style_Penalty
            g_loss += current_const_loss_style
            const_style_loss_summary = tf.summary.scalar("Loss_Generator/ConstStyleReference",
                                                         tf.abs(current_const_loss_style) / self.Lconst_style_Penalty)
            g_merged_summary=tf.summary.merge([g_merged_summary, const_style_loss_summary])



        # l1 loss
        if self.Pixel_Reconstruction_Penalty > eps * 10:
            l1_loss = tf.abs(generated_target_train - content_prototype_train)
            l1_loss = tf.reduce_mean(l1_loss) * self.Pixel_Reconstruction_Penalty
            l1_loss_summary = tf.summary.scalar("Loss_Reconstruction/Pixel_L1",
                                                tf.abs(l1_loss) / self.Pixel_Reconstruction_Penalty)
            g_loss+=l1_loss
            g_merged_summary = tf.summary.merge([g_merged_summary, l1_loss_summary])



        gen_vars_train = [var for var in tf.trainable_variables() if 'generator' in var.name]
        gen_vars_save = self.find_bn_avg_var(gen_vars_train)

        saver_generator = tf.train.Saver(max_to_keep=self.model_save_epochs, var_list=gen_vars_save)


        print(
            "Generator @%s with %s;" % (self.generator_devices, network_info))
        return generated_target_infer, generated_target_train,\
               g_loss, g_merged_summary, \
               gen_vars_train, saver_generator


    def discriminator_build(self,
                            g_loss,
                            g_merged_summary,
                            data_provider):


        def _calculate_accuracy_and_entropy(logits, true_labels, summary_name_parafix):
            prdt_labels = tf.argmax(logits,axis=1)
            true_labels = tf.argmax(true_labels,axis=1)
            correct_prediction = tf.equal(prdt_labels,true_labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) * 100
            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.nn.softmax(logits))
            entropy = tf.reduce_mean(entropy)
            acry_summary = tf.summary.scalar("Accuracy_Discriminator/AuxClassifier_"+summary_name_parafix, accuracy)
            enpy_summary = tf.summary.scalar("Entropy_Discriminator/AuxClassifier_"+summary_name_parafix, entropy)
            return acry_summary,enpy_summary

        generator_handle = getattr(self,'generator_handle')

        name_prefix = 'discriminator'

        discriminator_category_logit_length = len(self.involved_label0_list)

        critic_logit_length = int(np.floor(math.log(discriminator_category_logit_length) / math.log(2)))
        critic_logit_length = np.power(2,critic_logit_length+1)
        current_critic_logit_penalty = tf.placeholder(tf.float32, [], name='current_critic_logit_penalty')

        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device(self.discriminator_devices):
                true_content_train = data_provider.train_iterator.output_tensor_list[0]
                style_train = data_provider.train_iterator.output_tensor_list[1]
                fake_content_train = generator_handle.generated_target_train
                true_label0_train = data_provider.train_iterator.output_tensor_list[2]

                real_pack_train = tf.concat([true_content_train, style_train], axis=3)
                fake_pack_train = tf.concat([fake_content_train, style_train], axis=3)

                real_C_logits,real_Discriminator_logits,network_info = \
                    self.discriminator_implementation(image=real_pack_train,
                                                      is_training=True,
                                                      parameter_update_device=self.discriminator_devices,
                                                      category_logit_num=discriminator_category_logit_length,
                                                      batch_size=self.batch_size,
                                                      critic_length=critic_logit_length,
                                                      reuse=False,
                                                      initializer=self.initializer,
                                                      weight_decay=self.weight_decay_discriminator,
                                                      scope=name_prefix,
                                                      weight_decay_rate=self.discriminator_weight_decay_penalty)

                fake_C_logits,fake_Discriminator_logits,_ = \
                    self.discriminator_implementation(image=fake_pack_train,
                                                      is_training=True,
                                                      parameter_update_device=self.discriminator_devices,
                                                      category_logit_num=discriminator_category_logit_length,
                                                      batch_size=self.batch_size,
                                                      critic_length=critic_logit_length,
                                                      reuse=True,
                                                      initializer=self.initializer,
                                                      weight_decay=self.weight_decay_discriminator,
                                                      scope=name_prefix,
                                                      weight_decay_rate=self.discriminator_weight_decay_penalty)

                epsilon = tf.random_uniform([], 0.0, 1.0)
                interpolated_pair = real_pack_train*epsilon + (1-epsilon)*fake_pack_train
                _,intepolated_Cr_logits,_ = self.discriminator_implementation(image=interpolated_pair,
                                                                              is_training=True,
                                                                              parameter_update_device=self.discriminator_devices,
                                                                              category_logit_num=discriminator_category_logit_length,
                                                                              batch_size=self.batch_size,
                                                                              critic_length=critic_logit_length,
                                                                              reuse=True,
                                                                              initializer=self.initializer,
                                                                              weight_decay=self.weight_decay_discriminator,
                                                                              scope=name_prefix,
                                                                              weight_decay_rate=self.discriminator_weight_decay_penalty)
                discriminator_gradients = tf.gradients(intepolated_Cr_logits,interpolated_pair)[0]
                discriminator_slopes = tf.sqrt(eps+tf.reduce_sum(tf.square(discriminator_gradients),reduction_indices=[1]))
                discriminator_slopes = (discriminator_slopes-1.0)**2




                # discriminator infer
                content_infer = tf.placeholder(tf.float32, [self.batch_size,
                                                            self.img2img_width,
                                                            self.img2img_width,
                                                            self.input_output_img_filter_num])
                style_infer = tf.placeholder(tf.float32, [self.batch_size,
                                                         self.img2img_width,
                                                         self.img2img_width,
                                                         self.input_output_img_filter_num])
                infer_label0 = tf.placeholder(tf.float32,[self.batch_size,len(self.involved_label0_list)])

                infer_pack = tf.concat([content_infer, style_infer], axis=3)
                infer_categorical_logits,_,_ = \
                    self.discriminator_implementation(image=infer_pack,
                                                      is_training=False,
                                                      parameter_update_device=self.discriminator_devices,
                                                      category_logit_num=discriminator_category_logit_length,
                                                      batch_size=self.batch_size,
                                                      critic_length=critic_logit_length,
                                                      reuse=True,
                                                      initializer=self.initializer,
                                                      weight_decay=False,
                                                      weight_decay_rate=eps,
                                                      scope=name_prefix)
                infer_categorical_logits = tf.nn.softmax(infer_categorical_logits)

                curt_discriminator_handle = DiscriminatorHandle(current_critic_logit_penalty=current_critic_logit_penalty,
                                                                content_infer=content_infer,
                                                                style_infer=style_infer,
                                                                infer_label0=infer_label0,
                                                                infer_categorical_logits=infer_categorical_logits)
                setattr(self, "discriminator_handle", curt_discriminator_handle)



        # loss build
        d_loss = 0
        d_merged_summary=[]

        # weight_decay_loss
        discriminator_weight_decay_loss = tf.get_collection('discriminator_weight_decay')
        weight_decay_loss = 0
        if discriminator_weight_decay_loss:
            for ii in discriminator_weight_decay_loss:
                weight_decay_loss += ii
            weight_decay_loss = weight_decay_loss / len(discriminator_weight_decay_loss)
            discriminator_weight_decay_loss_summary = tf.summary.scalar("Loss_Discriminator/WeightDecay",
                                                                        tf.abs(weight_decay_loss)/self.discriminator_weight_decay_penalty)
            d_loss += weight_decay_loss
            d_merged_summary = tf.summary.merge([d_merged_summary,
                                                 discriminator_weight_decay_loss_summary])

        # category loss
        if self.Discriminator_Categorical_Penalty > 10 * eps:
            real_category_loss = tf.nn.softmax_cross_entropy_with_logits(logits=real_C_logits,
                                                                         labels=true_label0_train)
            fake_category_loss = tf.nn.softmax_cross_entropy_with_logits(logits=fake_C_logits,
                                                                         labels=true_label0_train)

            real_category_loss = tf.reduce_mean(real_category_loss) * self.Discriminator_Categorical_Penalty
            fake_category_loss = tf.reduce_mean(fake_category_loss) * self.Discriminator_Categorical_Penalty

            if self.training_mode == 'GeneratorInit':
                category_loss = (real_category_loss + fake_category_loss) / 2.0
            elif self.training_mode == 'DiscriminatorFineTune' or self.training_mode == 'DiscriminatorReTrain':
                category_loss = fake_category_loss + real_category_loss * eps




            real_category_loss_summary = tf.summary.scalar("Loss_Discriminator/CategoryReal",
                                                           tf.abs(real_category_loss) / self.Discriminator_Categorical_Penalty)
            fake_category_loss_summary = tf.summary.scalar("Loss_Discriminator/CategoryFake",
                                                           tf.abs(fake_category_loss) / self.Discriminator_Categorical_Penalty)
            category_loss_summary = tf.summary.scalar("Loss_Discriminator/Category", tf.abs(category_loss) / self.Discriminator_Categorical_Penalty)

            d_loss += category_loss
            d_merged_summary = tf.summary.merge([d_merged_summary,
                                                 real_category_loss_summary,
                                                 fake_category_loss_summary,
                                                 category_loss_summary])

            g_loss+=fake_category_loss
            g_merged_summary=tf.summary.merge([g_merged_summary,fake_category_loss_summary])


        # discriminative loss
        if self.training_mode == 'GeneratorInit':
            if self.Discriminative_Penalty > 10 * eps:
                d_loss_real = real_Discriminator_logits
                d_loss_fake = -fake_Discriminator_logits

                d_norm_real_loss = tf.abs(tf.abs(d_loss_real) - 1)
                d_norm_fake_loss = tf.abs(tf.abs(d_loss_fake) - 1)

                d_norm_real_loss = tf.reduce_mean(d_norm_real_loss) * current_critic_logit_penalty
                d_norm_fake_loss = tf.reduce_mean(d_norm_fake_loss) * current_critic_logit_penalty
                d_norm_loss = (d_norm_real_loss + d_norm_fake_loss) / 2

                d_norm_real_loss_summary = tf.summary.scalar("Loss_Discriminator/CriticLogit_NormReal",
                                                             d_norm_real_loss / current_critic_logit_penalty)
                d_norm_fake_loss_summary = tf.summary.scalar("Loss_Discriminator/CriticLogit_NormFake",
                                                             d_norm_fake_loss / current_critic_logit_penalty)
                d_norm_loss_summary = tf.summary.scalar("Loss_Discriminator/CriticLogit_Norm", d_norm_loss / current_critic_logit_penalty)

                d_loss += d_norm_loss
                d_merged_summary = tf.summary.merge([d_merged_summary,
                                                     d_norm_real_loss_summary,
                                                     d_norm_fake_loss_summary,
                                                     d_norm_loss_summary])

                d_loss_real = tf.reduce_mean(d_loss_real) * self.Discriminative_Penalty
                d_loss_fake = tf.reduce_mean(d_loss_fake) * self.Discriminative_Penalty
                d_loss_real_fake_summary = tf.summary.scalar("TrainingProgress_DiscriminatorRealFakeLoss",
                                                             tf.abs(
                                                                 d_loss_real + d_loss_fake) / self.Discriminative_Penalty)
                if self.Discriminator_Gradient_Penalty > 10 * eps:
                    d_gradient_loss = discriminator_slopes
                    d_gradient_loss = tf.reduce_mean(d_gradient_loss) * self.Discriminator_Gradient_Penalty
                    d_gradient_loss_summary = tf.summary.scalar("Loss_Discriminator/D_Gradient",
                                                                tf.abs(
                                                                    d_gradient_loss) / self.Discriminator_Gradient_Penalty)
                    d_loss += d_gradient_loss
                    d_merged_summary = tf.summary.merge([d_merged_summary,
                                                         d_gradient_loss_summary,
                                                         d_loss_real_fake_summary])

                cheat_loss = fake_Discriminator_logits


                d_loss_real_summary = tf.summary.scalar("Loss_Discriminator/AdversarialReal",
                                                        tf.abs(d_loss_real) / self.Discriminative_Penalty)
                d_loss_fake_summary = tf.summary.scalar("Loss_Discriminator/AdversarialFake",
                                                        tf.abs(d_loss_fake) / self.Discriminative_Penalty)

                d_loss += (d_loss_real+d_loss_fake)/2
                d_merged_summary = tf.summary.merge([d_merged_summary,
                                                     d_loss_fake_summary,
                                                     d_loss_real_summary])

                cheat_loss = tf.reduce_mean(cheat_loss) * self.Discriminative_Penalty
                cheat_loss_summary = tf.summary.scalar("Loss_Generator/Cheat", tf.abs(cheat_loss) / self.Discriminative_Penalty)
                g_loss+=cheat_loss
                g_merged_summary=tf.summary.merge([g_merged_summary,cheat_loss_summary])



        # d_loss_final and g_loss_final
        d_loss_summary = tf.summary.scalar("Loss_Discriminator/Total", tf.abs(d_loss))
        g_loss_summary = tf.summary.scalar("Loss_Generator/Total", tf.abs(g_loss))
        d_merged_summary=tf.summary.merge([d_merged_summary,d_loss_summary])
        g_merged_summary = tf.summary.merge([g_merged_summary, g_loss_summary])




        # build accuracy and entropy claculation
        # discriminator reference build here
        trn_real_acry, trn_real_enty = _calculate_accuracy_and_entropy(logits=infer_categorical_logits,
                                                                       true_labels=infer_label0,
                                                                       summary_name_parafix="TrainReal")
        trn_fake_acry, trn_fake_enty = _calculate_accuracy_and_entropy(logits=infer_categorical_logits,
                                                                       true_labels=infer_label0,
                                                                       summary_name_parafix="TrainFake")
        val_real_acry, val_real_enty = _calculate_accuracy_and_entropy(logits=infer_categorical_logits,
                                                                       true_labels=infer_label0,
                                                                       summary_name_parafix="TestReal")
        val_fake_acry, val_fake_enty = _calculate_accuracy_and_entropy(logits=infer_categorical_logits,
                                                                       true_labels=infer_label0,
                                                                       summary_name_parafix="TestFake")


        trn_real_summary = tf.summary.merge([trn_real_acry, trn_real_enty])
        trn_fake_summary = tf.summary.merge([trn_fake_acry, trn_fake_enty])

        tst_real_summary = tf.summary.merge([val_real_acry, val_real_enty])
        tst_fake_summary = tf.summary.merge([val_fake_acry, val_fake_enty])

        dis_vars_train = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        dis_vars_save = self.find_bn_avg_var(dis_vars_train)



        saver_discriminator = tf.train.Saver(max_to_keep=self.model_save_epochs, var_list=dis_vars_save)


        print("Discriminator @ %s with %s;" % (self.discriminator_devices,network_info))
        return g_merged_summary, d_merged_summary,\
               g_loss,d_loss, \
               trn_real_summary, trn_fake_summary, tst_real_summary, tst_fake_summary, \
               dis_vars_train,saver_discriminator,infer_categorical_logits


    def create_optimizer(self,
                         learning_rate,global_step,
                         gen_vars_train,generator_loss_train,
                         dis_vars_train,discriminator_loss_train):

        print(self.print_separater)

        if dis_vars_train:
            if self.optimization_method == 'adam':
                d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(discriminator_loss_train,
                                                                                        var_list=dis_vars_train,
                                                                                        global_step=global_step)
            elif self.optimization_method == 'gradient_descent':
                d_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(discriminator_loss_train,
                                                                                        var_list=dis_vars_train,
                                                                                        global_step=global_step)
            print("Optimizer Discriminator @ %s;" % (self.discriminator_devices))
        else:
            print("The discriminator is frozen.")
            d_optimizer=None

        if gen_vars_train and self.training_mode == 'GeneratorInit':
            if self.optimization_method == 'adam':
                g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(generator_loss_train,
                                                                                        var_list=gen_vars_train)
            elif self.optimization_method == 'gradient_descent':
                g_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(generator_loss_train,
                                                                                        var_list=gen_vars_train)

            print(
                "Optimizer Generator @ %s;" % (self.generator_devices))

        else:
            g_optimizer = None
            print("The generator is frozen.")
        print(self.print_separater)


        return g_optimizer, d_optimizer


    def divide_variables(self,base_vars, current_model_vars):

        re_init_vars=list()
        non_init_vars=list()


        counter=0
        for model_var in current_model_vars:
            model_var_name=str(model_var.name)
            model_var_name=model_var_name.replace(':0','')




            base_var_shape=[ii for ii in base_vars if model_var_name in ii[0]][0][1]

            same_dimension=True
            for ii in range(len(base_var_shape)):
                if int(model_var.shape[ii]) != base_var_shape[ii]:
                    same_dimension=False
                    break
            if same_dimension:
                non_init_vars.append(model_var)
            else:
                re_init_vars.append(model_var)

            counter+=1
        return re_init_vars,non_init_vars

    def restore_from_previous_model(self, saver_generator, saver_discriminator):
        def list_diff(first, second):
            second = set(second)
            return [item for item in first if item not in second]

        def checking_var_consistency(checking_var, stored_var_name, stored_var_shape):
            check_name = (stored_var_name == str(checking_var.name[:len(checking_var.name) - 2]))
            check_dimension = len(checking_var.shape.dims) == len(stored_var_shape)
            checking_shape_consistent = True
            if check_dimension:
                checking_shape = checking_var.shape

                for ii in range(len(checking_shape.dims)):
                    current_checking_shape = int(checking_shape[ii])
                    current_stored_shape = stored_var_shape[ii]
                    if not current_checking_shape == current_stored_shape:
                        checking_shape_consistent = False
                        break
            return check_name and check_dimension and checking_shape_consistent

        def variable_comparison_and_restore(current_saver, restore_model_dir, model_name):
            ckpt = tf.train.get_checkpoint_state(restore_model_dir)
            output_var_tensor_list = list()
            saved_var_name_list = list()
            current_var_name_list = list()
            for var_name, var_shape in tf.contrib.framework.list_variables(ckpt.model_checkpoint_path):
                for checking_var in current_saver._var_list:
                    found_var = checking_var_consistency(checking_var=checking_var,
                                                         stored_var_name=var_name,
                                                         stored_var_shape=var_shape)
                    if found_var:
                        output_var_tensor_list.append(checking_var)
                    current_var_name_list.append(str(checking_var.name[:len(checking_var.name) - 2]))
                saved_var_name_list.append(var_name)

            ignore_var_tensor_list = list_diff(first=current_saver._var_list,
                                               second=output_var_tensor_list)
            if ignore_var_tensor_list:
                print("IgnoreVars_ForVar in current model but not in the stored model:")
                counter = 0
                for ii in ignore_var_tensor_list:
                    print("No.%d, %s" % (counter, ii))
                    counter += 1
                if not self.debug_mode == 1:
                    raw_input("Press enter to continue")

            current_var_name_list = np.unique(current_var_name_list)
            ignore_var_name_list = list_diff(first=saved_var_name_list,
                                             second=current_var_name_list)
            if ignore_var_name_list:
                print("IgnoreVars_ForVar in stored model but not in the current model:")
                counter = 0
                for ii in ignore_var_name_list:
                    print("No.%d, %s" % (counter, ii))
                    counter += 1
                if not self.debug_mode == 1:
                    raw_input("Press enter to continue")

            saver = tf.train.Saver(max_to_keep=1, var_list=output_var_tensor_list)
            self.restore_model(saver=saver,
                               model_dir=restore_model_dir,
                               model_name=model_name)

        if not self.training_from_model == None:
            variable_comparison_and_restore(current_saver=saver_generator,
                                            restore_model_dir=os.path.join(self.training_from_model, 'generator'),
                                            model_name='Generator_ForPreviousTrainedBaseModel')

            if self.training_mode =='DiscriminatorFineTune':
                variable_comparison_and_restore(current_saver=saver_discriminator,
                                                restore_model_dir=os.path.join(self.training_from_model,
                                                                               'discriminator'),
                                                model_name='Discriminator_ForPreviousTrainedBaseModel')
            else:
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)
                print("The Classifier attached on the discriminator is trained from scratch.")
                print(self.print_separater)
                print(self.print_separater)
                print(self.print_separater)

    def model_initialization(self,
                             saver_generator, saver_discriminator,
                             saver_framework,
                             embedder_saver):
        # initialization of all the variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())



        # restore of high_level feature ebddactor
        ebdd_restored = self.restore_model(saver=embedder_saver,
                                           model_dir=self.style_embedder_dir,
                                           model_name="Embedder")



        # restore of the model frameworks
        if self.resume_training == 1:
            framework_restored = self.restore_model(saver=saver_framework,
                                                    model_dir=os.path.join(self.checkpoint_dir, 'frameworks'),
                                                    model_name="Frameworks")
            generator_restored = self.restore_model(saver=saver_generator,
                                                    model_dir=os.path.join(self.checkpoint_dir, 'generator'),
                                                    model_name="Generator")
            discriminator_restored = self.restore_model(saver=saver_discriminator,
                                                        model_dir=os.path.join(self.checkpoint_dir, 'discriminator'),
                                                        model_name="Discriminator")

        else:
            print("Framework initialized randomly.")
            print("Generator initialized randomly.")
            print("Discriminator initialized randomly.")
        print(self.print_separater)

    def train_procedures(self):

        if self.debug_mode == 1:
            self.sample_seconds = 5
            self.summary_seconds = 5
            self.record_seconds = 5
            self.print_info_seconds = 5
        else:
            self.summary_seconds = self.print_info_seconds * 1
            self.sample_seconds = self.print_info_seconds * 7
            self.record_seconds = self.print_info_seconds * 9

        with tf.Graph().as_default():

            # tensorflow parameters
            # DO NOT MODIFY!!!
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)



            # define the data set
            print(self.print_separater)
            data_provider = DataProvider(batch_size=self.batch_size,
                                         info_print_interval=self.print_info_seconds / 10,
                                         input_width=self.source_img_width,
                                         input_filters=self.input_output_img_filter_num,
                                         augment_train_data=self.train_data_augment,
                                         content_data_dir=self.content_data_dir,
                                         style_train_data_dir=self.style_train_data_dir,
                                         style_validation_data_dir=self.style_validation_data_dir,
                                         file_list_txt_content=self.file_list_txt_content,
                                         file_list_txt_style_train=self.file_list_txt_style_train,
                                         file_list_txt_style_validation=self.file_list_txt_style_validation,
                                         debug_mode=self.debug_mode)

            self.involved_label0_list, self.involved_label1_list = data_provider.get_involved_label_list()


            # ignore
            delete_items=list()
            involved_label_list = self.involved_label1_list
            for ii in self.accuracy_k:
                if ii>len(involved_label_list):
                    delete_items.append(ii)
            for ii in delete_items:
                self.accuracy_k.remove(ii)
            if delete_items and (not self.accuracy_k[len(self.accuracy_k)-1] == len(involved_label_list)):
                self.accuracy_k.append(len(involved_label_list))

            self.train_data_repeat_time = 1
            learning_rate_decay_rate = np.power(self.final_learning_rate_pctg, 1.0 / (self.epoch - 1))

            # define the directory name for model saving location and log saving location
            # delete or create relevant directories
            id, \
            self.checkpoint_dir, \
            self.log_dir, \
            self.inf_data_dir = self.get_model_id_and_dir_for_train()
            if (not self.resume_training == 1) and os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir) # delete!
            if (not self.resume_training == 1) and os.path.exists(self.checkpoint_dir):
                shutil.rmtree(self.checkpoint_dir)
            if (not self.resume_training == 1) and os.path.exists(self.inf_data_dir):
                shutil.rmtree(self.inf_data_dir)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(os.path.join(self.checkpoint_dir, 'discriminator'))
                os.makedirs(os.path.join(self.checkpoint_dir, 'generator'))
                os.makedirs(os.path.join(self.checkpoint_dir, 'frameworks'))
            if not os.path.exists(self.inf_data_dir):
                os.makedirs(self.inf_data_dir)


            #######################################################################################
            #######################################################################################
            #                                model building
            #######################################################################################
            #######################################################################################

            # for framework building
            epoch_step_increase_one_op, \
            learning_rate, \
            global_step, \
            epoch_step, \
            saver_frameworks = self.framework_building()

            # for embedder building
            embedder_saver, embedder_train_logits, \
            embedder_infer_logits_for_generation, embedder_infer_logits_for_validation,\
            ebdd_trn_real_merged, ebdd_trn_fake_merged, ebdd_val_real_merged, ebdd_val_fake_merged = \
                self.embedder_build(data_provider=data_provider)


            # for generator building
            generated_batch_infer, generated_batch_train,\
            g_loss, g_merged_summary, \
            gen_vars_train, saver_generator \
                = self.generator_build(data_provider=data_provider,
                                       embedder_train_logits=embedder_train_logits,
                                       embedder_infer_logits=embedder_infer_logits_for_generation)


            # for discriminator building
            g_merged_summary, d_merged_summary, \
            g_loss, d_loss, \
            dis_trn_real_summary, dis_trn_fake_summary, dis_val_real_summary, dis_val_fake_summary, \
            dis_vars_train, saver_discriminator, inferring_discriminator_categorical_logits =\
                self.discriminator_build(g_loss=g_loss,
                                         g_merged_summary=g_merged_summary,
                                         data_provider=data_provider)
            evalHandle = EvalHandle(inferring_generated_images=generated_batch_infer,
                                    training_generated_images=generated_batch_train,
                                    inferring_discriminator_categorical_logits=inferring_discriminator_categorical_logits)
            setattr(self, "eval_handle", evalHandle)


            # # for optimizer creation
            optimizer_g, optimizer_d = \
                self.create_optimizer(learning_rate=learning_rate,
                                      global_step=global_step,
                                      gen_vars_train=gen_vars_train,
                                      generator_loss_train=g_loss,
                                      dis_vars_train=dis_vars_train,
                                      discriminator_loss_train=d_loss)

            trn_real_dis_ebdd_summary_merged = tf.summary.merge([dis_trn_real_summary, ebdd_trn_real_merged])
            trn_fake_dis_ebdd_summary_merged = tf.summary.merge([dis_trn_fake_summary, ebdd_trn_fake_merged])
            val_real_dis_ebdd_summary_merged = tf.summary.merge([dis_val_real_summary, ebdd_val_real_merged])
            val_fake_dis_ebdd_summary_merged = tf.summary.merge([dis_val_fake_summary, ebdd_val_fake_merged])

            # summaries
            self.summary_finalization(g_loss_summary=g_merged_summary,
                                      d_loss_summary=d_merged_summary,
                                      trn_real_dis_ebdd_summaries=trn_real_dis_ebdd_summary_merged,
                                      val_real_dis_ebdd_summaries=val_real_dis_ebdd_summary_merged,
                                      trn_fake_dis_ebdd_summaries=trn_fake_dis_ebdd_summary_merged,
                                      val_fake_dis_ebdd_summaries=val_fake_dis_ebdd_summary_merged,
                                      learning_rate=learning_rate)


            # model initialization
            self.model_initialization(saver_framework=saver_frameworks,
                                      saver_generator=saver_generator,
                                      saver_discriminator=saver_discriminator,
                                      embedder_saver=embedder_saver)
            self.restore_from_previous_model(saver_generator=saver_generator,
                                             saver_discriminator=saver_discriminator)
            print(self.print_separater)
            summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

            print(self.print_separater)
            print(self.print_separater)
            print(self.print_separater)
            self.previous_highest_arcy_label = list()
            for ii in range(len(self.accuracy_k)):
                self.previous_highest_arcy_label.append(-1)
            self.previous_highest_accuracy_info_list = list()
            self.infer_epochs = 1


            print("%d Threads to read the data" % (data_provider.thread_num))
            print("BatchSize:%d, EpochNum:%d, LearningRateDecay:%.10f Per Epoch"
                  % (self.batch_size, self.epoch, learning_rate_decay_rate))
            print("TrainingSize:%d, ValidateSize:%d, StyleLabel0_Vec:%d, StyleLabel1_Vec:%d" %
                  (len(data_provider.train_iterator.data_content_path_list),
                   len(data_provider.validate_iterator.data_content_path_list),
                   len(self.involved_label0_list),
                   len(self.involved_label1_list)))
            print("PrintInfo:%ds, Summary:%ds, Sample:%ds, PrintRecord:%ds"%(self.print_info_seconds,
                                                                             self.summary_seconds,
                                                                             self.sample_seconds,
                                                                             self.record_seconds))
            print("ForInitTraining: InvolvedLabel0:%d, InvolvedLabel1:%d" % (len(self.involved_label0_list),
                                                                             len(self.involved_label1_list)))
            print(self.print_separater)
            print("Penalties:")
            print("Generator: PixelL1:%.3f,ConstSR:%.3f,Wgt:%.6f;" % (self.Pixel_Reconstruction_Penalty,
                                                                      self.Lconst_style_Penalty,
                                                                      self.generator_weight_decay_penalty))
            print("Discriminator: Cat:%.3f,Dis:%.3f,WST-Grdt:%.3f,Wgt:%.6f;" % (self.Discriminator_Categorical_Penalty,
                                                                                self.Discriminative_Penalty,
                                                                                self.Discriminator_Gradient_Penalty,
                                                                                self.discriminator_weight_decay_penalty))


            print("InitLearningRate:%.3f" % self.lr)
            print(self.print_separater)
            print("Initialization completed, and training started right now.")

            self.train_implementation(data_provider=data_provider,
                                      summary_writer=summary_writer,
                                      learning_rate_decay_rate=learning_rate_decay_rate, learning_rate=learning_rate,
                                      global_step=global_step, epoch_step_increase_one_op=epoch_step_increase_one_op,
                                      dis_vars_train=dis_vars_train, gen_vars_train=gen_vars_train,
                                      optimizer_d=optimizer_d, optimizer_g=optimizer_g,
                                      saver_discriminator=saver_discriminator, saver_generator=saver_generator,
                                      saver_frameworks=saver_frameworks,
                                      epoch_step=epoch_step)

    def train_implementation(self,
                             data_provider,
                             summary_writer,
                             learning_rate_decay_rate,
                             learning_rate,
                             global_step,
                             epoch_step_increase_one_op,
                             dis_vars_train,
                             gen_vars_train,
                             optimizer_d,
                             optimizer_g,
                             saver_discriminator,saver_generator,saver_frameworks,
                             epoch_step
                             ):
        def W_GAN(current_epoch,discriminator_handle):

            if current_epoch <= self.final_training_epochs:
                self.g_iters = 5
            else:
                self.g_iters = 2

            if global_step.eval(session=self.sess) <= self.g_iters * 5 * self.discriminator_initialization_iters:
                g_iters = self.g_iters * 5
            else:
                g_iters = self.g_iters

            info=""

            #batch_train_prototype, batch_train_reference, \
            #batch_train_label0_onehot, batch_train_label1_onehot,\
            #batch_train_label0_dense, batch_train_label1_dense = \
            #    data_provider.train_iterator.get_next_batch(sess=self.sess)
	    #print(batch_train_prototype.shape)
	    #print(batch+train_reference.shape)

            optimization_start = time.time()

            if dis_vars_train \
                    or global_step.eval(session=self.sess) == global_step_start:
                _ = self.sess.run(optimizer_d, feed_dict={learning_rate: current_lr_real,
                                                          discriminator_handle.current_critic_logit_penalty: current_critic_logit_penalty_value})
                info=info+"OptimizeOnD"

            # optimization for generator every (g_iters) iterations
            if ((global_step.eval(session=self.sess)) % g_iters == 0
                or global_step.eval(session=self.sess) == global_step_start + 1) and gen_vars_train and self.training_mode == 'GeneratorInit':
                _ = self.sess.run(optimizer_g, feed_dict={learning_rate: current_lr_real})

                info = info + "&&G"
            optimization_elapsed = time.time() - optimization_start

            return optimization_elapsed,info



        summary_start = time.time()
        sample_start = time.time()
        print_info_start = time.time()
        record_start = time.time()
        discriminator_handle = getattr(self, "discriminator_handle")
        embedder_handle = getattr(self, "embedder_handle")
        generator_handle = getattr(self, "generator_handle")

        

        if self.resume_training==1:
            ei_start = epoch_step.eval(self.sess)
            current_lr = self.lr * np.power(learning_rate_decay_rate, ei_start)
        else:
            ei_start = 0
            current_lr = self.lr
        current_lr = max(current_lr,TINIEST_LR)
        global_step_start = global_step.eval(session=self.sess)
        print("InitTrainingEpochs:%d, FinalTrainingEpochStartAt:%d" % (self.init_training_epochs,self.final_training_epochs))
        print("TrainingStart:Epoch:%d, GlobalStep:%d, LearnRate:%.5f" % (ei_start+1,global_step_start+1,current_lr))


        if self.debug_mode == 0:
            raw_input("Press enter to continue")
        print(self.print_separater)


        self.found_new_record_on_the_previous_epoch = True
        summary_handle = getattr(self, "summary_handle")
        training_start_time=time.time()

        training_epoch_list = range(ei_start,self.epoch,1)
        self.highest_test_accuracy = -1
        self.highest_test_accuracy_epoch = -1
        self.highest_test_accuracy_info_line1 = ''
        self.highest_test_accuracy_info_line2 = ''

        for ei in training_epoch_list:

            init_val=False
            if ei==ei_start:
                init_val=True
            data_provider.dataset_reinitialization(sess=self.sess, init_for_val=init_val,
                                                   info_interval=self.print_info_seconds/10)
            self.itrs_for_current_epoch = data_provider.compute_total_batch_num()

            print(self.print_separater)
            print("Epoch:%d/%d with Iters:%d is now commencing" % (ei + 1, self.epoch, self.itrs_for_current_epoch))
            print(self.print_separater)

            if not ei == ei_start:
                update_lr = current_lr * learning_rate_decay_rate
                print("decay learning rate from %.7f to %.7f" % (current_lr, update_lr))
                print(self.print_separater)
                current_lr = update_lr

            current_test_accuracy, current_top_k_test_accuracy\
                = self.validate_full_validation_dataset(data_provider=data_provider,
                                                        print_interval=self.print_info_seconds/10,
                                                        epoch_index=ei)

            if self.training_mode == 'GeneratorInit' or \
                    ((self.training_mode=='DiscriminatorFineTune' or self.training_mode=='DiscriminatorReTrain')
                     and current_test_accuracy > self.highest_test_accuracy):
                current_time = time.strftime('%Y-%m-%d @ %H:%M:%S', time.localtime())
                print("Time:%s,Checkpoint:SaveCheckpoint@step:%d" % (current_time, global_step.eval(session=self.sess)))
                self.checkpoint(saver=saver_discriminator,
                                model_dir=os.path.join(self.checkpoint_dir, 'discriminator'),
                                global_step=global_step)
                self.checkpoint(saver=saver_frameworks,
                                model_dir=os.path.join(self.checkpoint_dir, 'frameworks'),
                                global_step=global_step)
                self.checkpoint(saver=saver_generator,
                                model_dir=os.path.join(self.checkpoint_dir, 'generator'),
                                global_step=global_step)
                print(self.print_separater)

                if (current_test_accuracy > self.highest_test_accuracy
                    and (ei > self.init_training_epochs + 3
                         or (self.training_mode == 'DiscriminatorFineTune' or self.training_mode == 'DiscriminatorReTrain'))) \
                        or self.debug_mode == 1:
                    self.highest_test_accuracy = current_test_accuracy
                    self.highest_test_accuracy_epoch = ei
                    self.highest_test_accuracy_info_line1 = "CurrentHighestGeneratedTestAccuracy:%.3f @ Epoch:%d" %\
                                                            (self.highest_test_accuracy, self.highest_test_accuracy_epoch)

                    line2 = "GeneratedTopK @: "
                    for ii in self.accuracy_k:
                        if not ii == self.accuracy_k[len(self.accuracy_k) - 1]:
                            line2 = line2 + '%d/' % ii
                        else:
                            line2 = line2 + '%d:' % ii
                    tmp_counter = 0
                    for ii in current_top_k_test_accuracy:
                        if not tmp_counter == len(current_top_k_test_accuracy) - 1:
                            line2 = line2 + '%.3f/' % ii
                        else:
                            line2 = line2 + '%.3f;' % ii
                        tmp_counter += 1
                    self.highest_test_accuracy_info_line2 = line2
                elif ei <= self.init_training_epochs + 3 and self.training_mode == 'GeneratorInit':
                    self.highest_test_accuracy_info_line1 = 'N/A@Epoch:%d' % ei
                    self.highest_test_accuracy_info_line2 = 'N/A@Epoch:%d' % ei





            for bid in range(self.itrs_for_current_epoch):

                if time.time() - training_start_time < 600:
                    summary_seconds = 60
                    sample_seconds = 60
                    print_info_seconds = 60
                else:
                    summary_seconds = self.summary_seconds
                    sample_seconds = self.sample_seconds
                    print_info_seconds = self.print_info_seconds
                record_seconds = self.record_seconds


                this_itr_start = time.time()


                if epoch_step.eval(session=self.sess) < self.init_training_epochs:
                    current_critic_logit_penalty_value = (float(global_step.eval(session=self.sess))/float(self.init_training_epochs*self.itrs_for_current_epoch))*self.Discriminative_Penalty + eps
                    if self.training_mode == 'GeneratorInit':
                        current_lr_real = current_lr * 0.1
                    else:
                        current_lr_real = current_lr
                else:
                    current_critic_logit_penalty_value = self.Discriminative_Penalty
                    current_lr_real = current_lr
                current_critic_logit_penalty_value = current_critic_logit_penalty_value * 0.001


                optimization_consumed, \
                info = W_GAN(current_epoch=epoch_step.eval(session=self.sess),
                             discriminator_handle=discriminator_handle)

                passed_full = time.time() - training_start_time
                passed_itr = time.time() - this_itr_start

                if time.time()-print_info_start>print_info_seconds or global_step.eval(session=self.sess)==global_step_start+1:
                    print_info_start = time.time()
                    current_time = time.strftime('%Y-%m-%d@%H:%M:%S', time.localtime())
                    print("Time:%s,Epoch:%d/%d,Itr:%d/%d;" %
                          (current_time,
                           ei + 1, self.epoch,
                           bid + 1, self.itrs_for_current_epoch))


                    print("ItrDuration:%.2fses,FullDuration:%.2fhrs(%.2fdays);" %
                          (passed_itr, passed_full / 3600, passed_full / (3600 * 24)))
                    print(self.highest_test_accuracy_info_line1)
                    print(self.highest_test_accuracy_info_line2)

                    percentage_completed = float(global_step.eval(session=self.sess)) / float((self.epoch - ei_start) * self.itrs_for_current_epoch) * 100
                    percentage_to_be_fulfilled = 100 - percentage_completed
                    hrs_estimated_remaining = (float(passed_full) / (
                            percentage_completed + eps)) * percentage_to_be_fulfilled / 3600
                    print("CompletePctg:%.2f,TimeRemainingEstm:%.2fhrs(%.2fdays)" % (
                        percentage_completed, hrs_estimated_remaining,
                        hrs_estimated_remaining / 24))
                    print("CriticPenalty:%.5f/%.3f;" % (current_critic_logit_penalty_value,
                                                        self.Discriminative_Penalty))
                    print("TrainingInfo:%s" % info)

                    print(self.print_separater)

                if time.time()-record_start>record_seconds or global_step.eval(session=self.sess)==global_step_start+1:
                    record_start=time.time()
                    if not len(self.previous_highest_accuracy_info_list)==0:
                        for info in self.previous_highest_accuracy_info_list:
                            print(info)
                        print(self.print_separater)


                if ((time.time()-summary_start>summary_seconds)
                    and (self.debug_mode==0)
                    and (global_step.eval(session=self.sess)>=2500
                         or (self.training_mode=='DiscriminatorFineTune'
                             or self.training_mode=='DiscriminatorReTrain'))) \
                        or self.debug_mode==1:
                    summary_start = time.time()

                    if dis_vars_train:
                        d_summary = self.sess.run(
                            summary_handle.d_merged,
                            feed_dict={discriminator_handle.current_critic_logit_penalty:current_critic_logit_penalty_value})


                        summary_writer.add_summary(d_summary, global_step.eval(session=self.sess))
                    if gen_vars_train:
                        g_summary = self.sess.run(summary_handle.g_merged)
                        summary_writer.add_summary(g_summary, global_step.eval(session=self.sess))

                    learning_rate_summary = self.sess.run(summary_handle.learning_rate,
                                                          feed_dict={learning_rate: current_lr_real})
                    summary_writer.add_summary(learning_rate_summary, global_step.eval(session=self.sess))
                    summary_writer.flush()


                if time.time()-sample_start>sample_seconds or global_step.eval(session=self.sess)==global_step_start+1 or bid==self.itrs_for_current_epoch-1:
                    sample_start = time.time()

                    # check for train set
                    self.validate_model(train_mark=True,
                                        summary_writer=summary_writer,
                                        global_step=global_step,
                                        discriminator_handle=discriminator_handle,
                                        embedder_handle=embedder_handle,
                                        generator_handle=generator_handle,
                                        data_provider=data_provider)

                    # check for validation set
                    self.validate_model(train_mark=False,
                                        summary_writer=summary_writer,
                                        global_step=global_step,
                                        discriminator_handle=discriminator_handle,
                                        embedder_handle=embedder_handle,
                                        generator_handle=generator_handle,
                                        data_provider=data_provider)

                    summary_writer.flush()

            # self-increase the epoch number
            self.sess.run(epoch_step_increase_one_op)

        print("Training Completed.")


    def validate_full_validation_dataset(self,
                                         data_provider, print_interval,epoch_index):

        def top_k_correct_calculation(logits, true_label):
            k = len(self.accuracy_k)
            top_k_indices = np.argsort(-logits, axis=1)[:, 0:k]
            for ii in range(k):
                estm_label = top_k_indices[:, ii]
                diff = np.abs(estm_label - true_label)
                if ii == 0:
                    full_diff = np.reshape(diff, [diff.shape[0], 1])
                else:
                    full_diff = np.concatenate([full_diff, np.reshape(diff, [diff.shape[0], 1])], axis=1)
            top_k_correct_list = list()
            for ii in range(len(self.accuracy_k)):
                this_k = self.accuracy_k[ii]
                this_k_diff = full_diff[:, 0:this_k]
                if this_k == 0:
                    this_k_diff = np.reshape(this_k_diff, [this_k_diff.shape[0], 1])
                min_v = np.min(this_k_diff, axis=1)
                correct = [i for i, v in enumerate(min_v) if v == 0]
                top_k_correct_list.append(len(correct))
            return top_k_correct_list


        iter_num = len(data_provider.validate_iterator.data_content_path_list) / self.batch_size + 1
        full_generated_correct = 0
        full_true_correct = 0
        full_counter = 0
        full_generated_top_k_correct_list = list()
        full_true_top_k_correct_list = list()
        current_generated_top_k_accuracy_list = list()
        current_true_top_k_accuracy_list = list()
        final_generated_top_k_accuracy_list = list()
        final_true_top_k_accuracy_list = list()
        for ii in range(len(self.accuracy_k)):
            full_generated_top_k_correct_list.append(0)
            full_true_top_k_correct_list.append(0)
            current_generated_top_k_accuracy_list.append(-1)
            current_true_top_k_accuracy_list.append(-1)
            final_generated_top_k_accuracy_list.append(-1)
            final_true_top_k_accuracy_list.append(-1)

        timer_start = time.time()
        for curt_iter in range(iter_num):


            generated_content_batch, \
            true_content_batch, style_batch, \
            training_img_list, \
            label0_onehot, label1_onehot, \
            label0_dense, label1_dense, \
                = self.generate_fake_samples(training_mark=False,
                                             current_iterator=data_provider.validate_iterator)

            generated_categorical_logits = \
                self.discriminate_sample(content_infer_batch=generated_content_batch,
                                         style_infer_batch=style_batch)
            true_categorical_logits = \
                self.discriminate_sample(content_infer_batch=true_content_batch,
                                         style_infer_batch=style_batch)


            if curt_iter == iter_num - 1:
                add_num = iter_num * self.batch_size - len(data_provider.validate_iterator.data_content_path_list)
                remain_num = self.batch_size - add_num
                generated_categorical_logits = generated_categorical_logits[0:remain_num,:]
                true_categorical_logits = true_categorical_logits[0:remain_num, :]
                label0_onehot = label0_onehot[0:remain_num, :]

            generated_estm_label = np.argmax(generated_categorical_logits, axis=1)
            true_estm_label = np.argmax(true_categorical_logits, axis=1)
            true_label = np.argmax(label0_onehot, axis=1)
            # if curt_iter == iter_num - 1:
            #     add_num = iter_num * self.batch_size - len(data_provider.validate_iterator.data_content_path_list)
            #     remain_num = self.batch_size - add_num
            #     generated_estm_label = generated_estm_label[0:remain_num]
            #     true_estm_label = true_estm_label[0:remain_num]
            #     true_label = true_label[0:remain_num]

            if curt_iter == iter_num - 1:
                full_counter+=remain_num
            else:
                full_counter+=self.batch_size

            generated_diff = generated_estm_label - true_label
            true_diff = true_estm_label - true_label

            current_generated_correct = [i for i, v in enumerate(generated_diff) if v == 0]
            current_true_correct = [i for i, v in enumerate(true_diff) if v == 0]
            current_generated_correct = len(current_generated_correct)
            current_true_correct = len(current_true_correct)

            full_generated_correct+=current_generated_correct
            full_true_correct+=current_true_correct

            current_generated_top_k_correct_list = top_k_correct_calculation(logits=generated_categorical_logits,
                                                                             true_label=true_label)
            current_true_top_k_correct_list = top_k_correct_calculation(logits=true_categorical_logits,
                                                                        true_label=true_label)
            current_accuracy_generated = np.float32(full_generated_correct) / np.float32(full_counter) * 100
            current_accuracy_true = np.float32(full_true_correct) / np.float32(full_counter) * 100

            for ii in range(len(self.accuracy_k)):
                full_generated_top_k_correct_list[ii] += current_generated_top_k_correct_list[ii]
                full_true_top_k_correct_list[ii] += current_true_top_k_correct_list[ii]
                current_generated_top_k_accuracy_list[ii] = np.float32(full_generated_top_k_correct_list[ii]) / np.float32(full_counter) * 100
                current_true_top_k_accuracy_list[ii] = np.float32(full_true_top_k_correct_list[ii]) / np.float32(full_counter) * 100


            if time.time() - timer_start > print_interval or curt_iter==0 or curt_iter==iter_num-1:
                timer_start = time.time()
                print("Validate@Epoch:%d, CurrentAccuracyOnTrue/Generated:%.3f/%.3f, Counter:%d/%d" %
                      (epoch_index,
                       current_accuracy_true,current_accuracy_generated,
                       full_counter,len(data_provider.validate_iterator.data_content_path_list)))
                print("Top_K_Accuracies_ForGenerated:")
                print("@", end='')
                for ii in self.accuracy_k:
                    if not ii == self.accuracy_k[len(self.accuracy_k)-1]:
                        print('%d/' % ii, end='')
                    else:
                        print('%d:' % ii, end='')
                tmp_counter=0
                for ii in current_generated_top_k_accuracy_list:
                    if not tmp_counter == len(current_generated_top_k_accuracy_list)-1:
                        print('%.3f/' % ii, end='')
                    else:
                        print('%.3f;' % ii)
                    tmp_counter+=1
                print("Top_K_Accuracies_ForTrue:")
                print("@", end='')
                for ii in self.accuracy_k:
                    if not ii == self.accuracy_k[len(self.accuracy_k) - 1]:
                        print('%d/' % ii, end='')
                    else:
                        print('%d:' % ii, end='')
                tmp_counter=0
                for ii in current_true_top_k_accuracy_list:
                    if not tmp_counter == len(current_generated_top_k_accuracy_list) - 1:
                        print('%.3f/' % ii, end='')
                    else:
                        print('%.3f;' % ii)
                    tmp_counter += 1
                print(self.print_separater)

        final_accuracy_generated = np.float32(full_generated_correct) / np.float32(len(data_provider.validate_iterator.data_content_path_list)) * 100
        final_accuracy_true = np.float32(full_true_correct) / np.float32(len(data_provider.validate_iterator.data_content_path_list)) * 100
        for ii in range(len(self.accuracy_k)):
            final_generated_top_k_accuracy_list[ii] = np.float32(full_generated_top_k_correct_list[ii]) / np.float32(len(data_provider.validate_iterator.data_content_path_list))  * 100
            final_true_top_k_accuracy_list[ii] = np.float32(full_true_top_k_correct_list[ii]) / np.float32(len(data_provider.validate_iterator.data_content_path_list))  * 100

        print(self.print_separater)
        print(self.print_separater)
        print(self.print_separater)
        print("Validate@Epoch:%d, FullAccuracyOnTrue/Generated:%.3f/%.3f" %
              (epoch_index, final_accuracy_true, final_accuracy_generated))
        print("Top_K_Accuracies_ForGenerated:")
        print("@", end='')
        for ii in self.accuracy_k:
            if not ii == self.accuracy_k[len(self.accuracy_k) - 1]:
                print('%d/' % ii, end='')
            else:
                print('%d:' % ii, end='')
        tmp_counter = 0
        for ii in final_generated_top_k_accuracy_list:
            if not tmp_counter == len(final_generated_top_k_accuracy_list) - 1:
                print('%.3f/' % ii, end='')
            else:
                print('%.3f;' % ii)
            tmp_counter += 1
        print("Top_K_Accuracies_ForTrue:")
        print("@", end='')
        for ii in self.accuracy_k:
            if not ii == self.accuracy_k[len(self.accuracy_k) - 1]:
                print('%d/' % ii, end='')
            else:
                print('%d:' % ii, end='')
        tmp_counter = 0
        for ii in final_true_top_k_accuracy_list:
            if not tmp_counter == len(final_true_top_k_accuracy_list) - 1:
                print('%.3f/' % ii, end='')
            else:
                print('%.3f;' % ii)
            tmp_counter += 1
        print(self.print_separater)
        print(self.print_separater)
        print(self.print_separater)


        return final_accuracy_generated, final_generated_top_k_accuracy_list
