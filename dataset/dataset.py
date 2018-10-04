import numpy as np
import tensorflow as tf
import os

import sys
sys.path.append('..')
from utilities.utils import image_show
import random as rnd
import time
import multiprocessing as multi_thread
print_separator = "#################################################################"
from tensorflow.python.client import device_lib
import copy as cpy

GRAYSCALE_AVG = 127.5
class Dataset(object):
    def __init__(self,
                 data_list,
                 label0_list,label1_list,
                 sorted_by_label0,
                 print_marks,
                 info_print_interval):
        self.data_list = data_list
        self.label0_list = label0_list
        self.label1_list = label1_list

        if sorted_by_label0:
            self.sorted_data_by_label0(print_marks=print_marks,
                                       info_print_interval=info_print_interval)



        
    def sorted_data_by_label0(self,print_marks,info_print_interval):
        print(print_separator)
        label0_vec = np.unique(self.label0_list)
        sorted_data_list=list()
        sorted_label0_list = list()
        sorted_label1_list = list()



        sort_start = time.time()
        counter=0
        for label0 in label0_vec:
            found_indices = [ii for ii in range(len(self.label0_list)) if self.label0_list[ii] == label0]
            for ii in found_indices:
                sorted_data_list.append(self.data_list[ii])
                sorted_label0_list.append(self.label0_list[ii])
                sorted_label1_list.append(self.label1_list[ii])

            if time.time()-sort_start > info_print_interval or label0==label0_vec[0] or counter == len(label0_vec)-1:
                print(print_marks+'SortingForLabel0:%d/%d' % (counter+1,len(label0_vec)))
                sort_start=time.time()
            counter+=1
        self.data_list = sorted_data_list
        self.label0_list = sorted_label0_list
        self.label1_list = sorted_label1_list
        print(print_separator)



class Dataset_Iterator(object):
    def __init__(self,thread_num,
                 batch_size,
                 input_width, input_channel,
                 style_reference,
                 content_prototype,
                 info_print_interval,print_marks,
                 augment=False,train_iterator_mark=False,
                 label0_vec=-1,label1_vec=-1,debug_mode=False,
                 ):
        self.batch_size = batch_size
        self.input_width = input_width
        self.input_filters = input_channel
        self.style_reference = style_reference
        self.content_prototype = content_prototype

        self.thread_num = thread_num
        self.augment = augment

        if train_iterator_mark:
            self.label0_vec = np.unique(self.style_reference.label0_list)
            self.label1_vec = np.unique(self.style_reference.label1_list)
            if debug_mode:
                self.label0_vec = np.concatenate([range(176161, 176191),
                                                  range(0,3725)],axis=0)
                self.label1_vec = range(1001, 1051)
                self.label0_vec = map(str, self.label0_vec)
                self.label1_vec = map(str, self.label1_vec)
        else:
            self.label0_vec = label0_vec
            self.label1_vec = label1_vec

        self.content_data_list_alignment_with_true_style_data(print_marks=print_marks,
                                                              info_print_interval=info_print_interval)

        


    def content_data_list_alignment_with_true_style_data(self, print_marks,info_print_interval):

        self.data_style_path_list = list()
        self.data_content_path_list = list()
        self.data_label0_list = list()
        self.data_label1_list = list()
        time_start = time.time()
        label0_vec = np.unique(self.style_reference.label0_list)
        label0_counter = 0
        for label0 in label0_vec:
            current_label0_indices_on_the_style_data = [ii for ii in range(len(self.style_reference.label0_list)) if self.style_reference.label0_list[ii] == label0]
            current_label0_index_on_the_content = self.content_prototype.label0_list.index(label0)
            for index_curt in current_label0_indices_on_the_style_data:
                self.data_style_path_list.append(self.style_reference.data_list[index_curt])
                self.data_content_path_list.append(self.content_prototype.data_list[current_label0_index_on_the_content])
                self.data_label0_list.append(self.style_reference.label0_list[index_curt])
                self.data_label1_list.append(self.style_reference.label1_list[index_curt])

            label0_counter+=1
            if time.time()-time_start>info_print_interval or label0==label0_vec[0] or label0_counter==len(label0_vec):
                time_start=time.time()

                print(print_marks + ' FindingCorrespondendingContentPrototype_BasedOnLabel0:%d/%d' %
                          (label0_counter, len(label0_vec)))




        # self.label0_vec = np.unique(self.data_label0_list)
        # self.label1_vec = np.unique(self.data_label1_list)
        del self.content_prototype
        del self.style_reference

        print(print_separator)




    def reproduce_dataset_lists(self, info, shuffle,info_print_interval):

        if shuffle:
            time_start = time.time()
            full_counter = 0
            old_data_style_path_list = cpy.deepcopy(self.data_style_path_list)
            old_data_content_path_list = cpy.deepcopy(self.data_content_path_list)
            old_data_label0_list = cpy.deepcopy(self.data_label0_list)
            old_data_label1_list = cpy.deepcopy(self.data_label1_list)

            self.data_style_path_list=list()
            self.data_content_path_list=list()
            self.data_label0_list = list()
            self.data_label1_list=list()
            indices_shuffled = np.random.permutation(len(old_data_style_path_list))

            for curt_index in indices_shuffled:
                self.data_style_path_list.append(old_data_style_path_list[curt_index])
                self.data_content_path_list.append(old_data_content_path_list[curt_index])
                self.data_label0_list.append(old_data_label0_list[curt_index])
                self.data_label1_list.append(old_data_label1_list[curt_index])
                if time.time() - time_start > info_print_interval or full_counter == 0:
                    time_start = time.time()
                    print('%s:DatasetReInitialization@CurrentLabel1:%d/%d' % (info, full_counter+1, len(indices_shuffled)))
                full_counter+=1


    def iterator_reset(self,sess):
        sess.run(self.style_iterator.initializer,
                 feed_dict={self.style_data_list_input_op:self.data_style_path_list,
                            self.style_label0_list_input_op:self.data_label0_list,
                            self.style_label1_list_input_op:self.data_label1_list})
        sess.run(self.content_iterator.initializer,
                 feed_dict={self.conetent_data_list_input_op:self.data_content_path_list})



    def create_dataset_op(self):
        def _get_tensor_slice_style():
            data_path_list_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
            label0_list_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
            label1_list_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
            dataset_op = tf.data.Dataset.from_tensor_slices((data_path_list_placeholder,
                                                             label0_list_placeholder,
                                                             label1_list_placeholder))
            return dataset_op, \
                   data_path_list_placeholder, \
                   label0_list_placeholder, label1_list_placeholder

        def _get_tensor_slice_content():
            data_path_list_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
            dataset_op = tf.data.Dataset.from_tensor_slices((data_path_list_placeholder))
            return dataset_op, data_path_list_placeholder



        def _parser_for_style_data(file_list,label0_list,label1_list):
            image_string = tf.read_file(file_list)
            image_decoded = tf.image.decode_image(contents=image_string, channels=1)
            image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, self.input_width, self.input_width)
            img_output = tf.slice(image_resized,
                                  [0, 0, 0],
                                  [self.input_width, self.input_width, self.input_filters])
            # img_output = tf.subtract(tf.divide(tf.cast(img_output, tf.float32), tf.constant(127.5, tf.float32)),
            #                          tf.constant(1, tf.float32))
            return img_output, label0_list, label1_list

        def _parser_for_content_data(file_list):
            image_string = tf.read_file(file_list)
            image_decoded = tf.image.decode_image(contents=image_string, channels=1)
            image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, self.input_width, self.input_width)
            img_output = tf.slice(image_resized,
                                  [0, 0, 0],
                                  [self.input_width, self.input_width, self.input_filters])
            # img_output = tf.subtract(tf.divide(tf.cast(img_output, tf.float32), tf.constant(127.5, tf.float32)),
            #                          tf.constant(1, tf.float32))
            return img_output

        def _convert_label_to_one_hot(dense_label,vocabulary):
            table = tf.contrib.lookup.index_table_from_tensor(mapping=vocabulary, default_value=0)
            encoded = tf.one_hot(table.lookup(dense_label),len(vocabulary), dtype=tf.float32)
            return encoded





        # for true style image
        style_dataset, \
        style_data_list_input_op, \
        style_label0_list_input_op, style_label1_list_input_op = \
            _get_tensor_slice_style()

        style_dataset = \
            style_dataset.map(map_func=_parser_for_style_data,
                              num_parallel_calls=self.thread_num).apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat(-1)
        style_iterator = style_dataset.make_initializable_iterator()
        style_img_tensor, style_label0_tensor_dense, style_label1_tensor_dense = \
            style_iterator.get_next()

        self.style_iterator = style_iterator
        self.style_data_list_input_op = style_data_list_input_op
        self.style_label0_list_input_op = style_label0_list_input_op
        self.style_label1_list_input_op = style_label1_list_input_op


        # for prototype images
        contrent_dataset, \
        conetent_data_list_input_op = _get_tensor_slice_content()
        content_dataset = \
            contrent_dataset.map(map_func=_parser_for_content_data,
                              num_parallel_calls=self.thread_num).apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat(-1)
        content_iterator = content_dataset.make_initializable_iterator()
        content_img_tensor = content_iterator.get_next()

        self.content_iterator = content_iterator
        self.conetent_data_list_input_op = conetent_data_list_input_op

        style_label0_tensor_onehot =_convert_label_to_one_hot(dense_label=style_label0_tensor_dense, vocabulary=self.label0_vec)
        style_label1_tensor_onehot =_convert_label_to_one_hot(dense_label=style_label1_tensor_dense, vocabulary=self.label1_vec)

        img_all = tf.concat([content_img_tensor, style_img_tensor], axis=3)
        if self.augment:
            for ii in range(self.batch_size):
                current_img = img_all[ii,:,:,:]
                crop_size = tf.random_uniform(shape=[],
                                              minval=int(int(img_all.shape[1]) * 0.75),
                                              maxval=int(img_all.shape[1]) + 1, dtype=tf.int32)
                cropped_img = tf.random_crop(value=current_img,
                                             size=[crop_size, crop_size,
                                                   int(img_all.shape[3])])
                cropped_img = tf.image.resize_images(cropped_img, [self.input_width, self.input_width])
                cropped_img = tf.reshape(cropped_img,[self.input_width,self.input_width,self.input_filters*2])
                cropped_img = tf.expand_dims(cropped_img,axis=0)
                if ii ==0:
                    img_all_new = cropped_img
                else:
                    img_all_new = tf.concat([img_all_new,cropped_img],axis=0)
            img_all = img_all_new

        img_all = tf.subtract(tf.divide(tf.cast(img_all, tf.float32), tf.constant(GRAYSCALE_AVG, tf.float32)),
                              tf.constant(1, tf.float32))
        content_img_tensor = tf.expand_dims(img_all[:, :, :, 0], axis=3)
        style_img_tensor = tf.expand_dims(img_all[:, :, :, 1], axis=3)

        self.output_tensor_list = list()
        self.output_tensor_list.append(content_img_tensor) # 0
        self.output_tensor_list.append(style_img_tensor)  # 1
        self.output_tensor_list.append(style_label0_tensor_onehot)  # 2
        self.output_tensor_list.append(style_label1_tensor_onehot)  # 3
        self.output_tensor_list.append(style_label0_tensor_dense)   # 4
        self.output_tensor_list.append(style_label1_tensor_dense)   # 5



    def get_next_batch(self, sess):
        prototype,reference, \
        onehot_label0, onehot_label1, \
        dense_label0, dense_label1 = \
            sess.run([self.output_tensor_list[0],
                      self.output_tensor_list[1],
                      self.output_tensor_list[2],
                      self.output_tensor_list[3],
                      self.output_tensor_list[4],
                      self.output_tensor_list[5]])
        return prototype,reference, \
               onehot_label0, onehot_label1, \
               dense_label0, dense_label1

class DataProvider(object):
    def __init__(self,
                 batch_size,
                 input_width,
                 input_filters,
                 info_print_interval,
                 file_list_txt_content, file_list_txt_style_train, file_list_txt_style_validation,
                 content_data_dir, style_train_data_dir,style_validation_data_dir,
                 augment_train_data=True,
                 debug_mode=False):

        local_device_protos = device_lib.list_local_devices()
        gpu_device = [x.name for x in local_device_protos if x.device_type == 'GPU']
        if len(gpu_device) == 0:
            self.thread_num = multi_thread.cpu_count()
        else:
            self.thread_num = int(multi_thread.cpu_count() / len(gpu_device))

        self.batch_size = batch_size
        self.augment_train_data=augment_train_data
        self.input_width = input_width
        self.input_filters = input_filters
        self.dataset_iterator_create(content_data_dir=content_data_dir,
                                     file_list_txt_content=file_list_txt_content,
                                     style_train_data_dir=style_train_data_dir,
                                     file_list_txt_style_train=file_list_txt_style_train,
                                     style_validation_data_dir=style_validation_data_dir,
                                     file_list_txt_style_validation=file_list_txt_style_validation,
                                     info_print_interval=info_print_interval,
                                     debug_mode=debug_mode)
        # self.content_input_num = self.train_iterator.content_input_num

    def dataset_reinitialization(self, sess, init_for_val, info_interval):
        self.train_iterator.reproduce_dataset_lists(info="TrainData", shuffle=True, info_print_interval=info_interval)
        self.train_iterator.iterator_reset(sess=sess)
        if init_for_val:
            self.validate_iterator.reproduce_dataset_lists(info="ValData", shuffle=False, info_print_interval=info_interval)
            self.validate_iterator.iterator_reset(sess=sess)
        print(print_separator)

    def data_file_list_read(self,file_list_txt,file_data_dir):

        label0_list = list()
        label1_list = list()
        data_list = list()

        for ii in range(len(file_list_txt)):

            file_handle = open(file_list_txt[ii], 'r')
            lines = file_handle.readlines()

            for line in lines:
                curt_line = line.split('@')
                label1_list.append(curt_line[2])
                label0_list.append(curt_line[1])
                curt_data = curt_line[3].split('\n')[0]
                if curt_data[0] == '/':
                    curt_data = curt_data[1:]
                curt_data_path = os.path.join(file_data_dir[ii], curt_data)
                data_list.append(curt_data_path)
            file_handle.close()
        return label1_list, label0_list, data_list



    def dataset_iterator_create(self,info_print_interval,
                                content_data_dir,file_list_txt_content,
                                style_train_data_dir, file_list_txt_style_train,
                                style_validation_data_dir, file_list_txt_style_validation,
                                debug_mode=False):

        def _filter_current_label1_data(current_label1, full_data_list, full_label1_list,full_label0_list):
            selected_indices = [ii for ii in range(len(full_label1_list)) if full_label1_list[ii] == current_label1]
            selected_data_list=list()
            selected_label0_list = list()
            selected_label1_list = list()

            for ii in selected_indices:
                selected_data_list.append(full_data_list[ii])
                selected_label0_list.append(full_label0_list[ii])
                selected_label1_list.append(full_label1_list[ii])
            return selected_data_list, selected_label0_list,selected_label1_list

        # building for content data set
        content_label1_list, content_label0_list, content_data_path_list = \
            self.data_file_list_read(file_list_txt=file_list_txt_content,
                                     file_data_dir=content_data_dir)
        self.content_label0_vec = np.unique(content_label0_list)



        content_dataset = Dataset(data_list=cpy.deepcopy(content_data_path_list),
                                  label0_list=cpy.deepcopy(content_label0_list),
                                  label1_list=cpy.deepcopy(content_label1_list),
                                  sorted_by_label0=False,
                                  print_marks='ForOriginalContentData:',
                                  info_print_interval=info_print_interval)



        # building for style data set for train
        train_style_label1_list, train_style_label0_list, train_style_data_path_list = \
            self.data_file_list_read(file_list_txt=file_list_txt_style_train,
                                     file_data_dir=style_train_data_dir)

        train_style_dataset = Dataset(data_list=cpy.deepcopy(train_style_data_path_list),
                                      label0_list=cpy.deepcopy(train_style_label0_list),
                                      label1_list=cpy.deepcopy(train_style_label1_list),
                                      sorted_by_label0=False,
                                      print_marks='ForStyleReferenceTrainData:',
                                      info_print_interval=info_print_interval)

        # construct the train iterator
        self.train_iterator = Dataset_Iterator(batch_size=self.batch_size,
                                               thread_num=self.thread_num,
                                               input_width=self.input_width,
                                               input_channel=self.input_filters,
                                               style_reference=train_style_dataset,
                                               content_prototype=content_dataset,
                                               augment=self.augment_train_data,
                                               info_print_interval=info_print_interval,
                                               print_marks='ForTrainIterator:',
                                               train_iterator_mark=True,
                                               debug_mode=debug_mode)
        self.style_label0_vec = np.unique(self.train_iterator.label0_vec)
        self.style_label1_vec = np.unique(self.train_iterator.label1_vec)



        # building for style data set for validation
        validation_style_label1_list, validation_style_label0_list, validation_style_data_path_list = \
            self.data_file_list_read(file_list_txt=file_list_txt_style_validation,
                                     file_data_dir=style_validation_data_dir)


        validation_style_dataset = Dataset(data_list=cpy.deepcopy(validation_style_data_path_list),
                                           label0_list=cpy.deepcopy(validation_style_label0_list),
                                           label1_list=cpy.deepcopy(validation_style_label1_list),
                                           sorted_by_label0=False,
                                           print_marks='ForStyleReferenceValidationData:',
                                           info_print_interval=info_print_interval)



        # construct the validation iterator
        self.validate_iterator = Dataset_Iterator(batch_size=self.batch_size,
                                                  thread_num=self.thread_num,
                                                  input_width=self.input_width,
                                                  input_channel=self.input_filters,
                                                  style_reference=validation_style_dataset,
                                                  content_prototype=content_dataset,
                                                  augment=False,
                                                  info_print_interval=info_print_interval,
                                                  print_marks='ForValidationIterator:',
                                                  train_iterator_mark=False,
                                                  label0_vec=self.train_iterator.label0_vec,
                                                  label1_vec=self.train_iterator.label1_vec,
                                                  debug_mode=debug_mode)
        self.train_iterator.create_dataset_op()
        self.validate_iterator.create_dataset_op()

        print(print_separator)

    def get_involved_label_list(self):
        return self.style_label0_vec,self.style_label1_vec

    def compute_total_batch_num(self):
        """Total padded batch num"""
        return int(np.ceil(len(self.train_iterator.data_style_path_list) / float(self.batch_size)))
