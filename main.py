from __future__ import print_function
from six.moves import xrange
import os
import better_exceptions
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from functools import partial

from dataset import CitySpaces
from model import FRRN,_arch_type_a

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

NUM_CLASSES = 19 + 1
def main(config,
         RANDOM_SEED,
         LOG_DIR,
         TRAIN_NUM,
         BATCH_SIZE,
         LEARNING_RATE,
         DECAY_VAL,
         DECAY_STEPS,
         DECAY_STAIRCASE,
         K,
         CROP_SIZE, #Frist cropped,
         IM_SIZE, #And, Resized
         Z_RANGE, # z value for gamma augmentation
         SAVE_PERIOD,
         SUMMARY_PERIOD):
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)

    # >>>>>>> DATASET
    cityspaces = CitySpaces()
    ims,lbs = cityspaces.build_queue(target='train',crop=CROP_SIZE,resize=IM_SIZE,z_range=Z_RANGE,batch_size=BATCH_SIZE,num_threads=4)
    """
    valid_ims,valid_lbs = cityspaces.build_queue(target='val')
    """
    # <<<<<<<

    # >>>>>>> MODEL
    with tf.variable_scope('train'):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS, DECAY_VAL, staircase=DECAY_STAIRCASE)
        tf.summary.scalar('lr',learning_rate)

        with tf.variable_scope('params') as params:
            pass
        net = FRRN(learning_rate,global_step,K,ims,lbs,partial(_arch_type_a,NUM_CLASSES),params,True)

    """
    #Memory Constraint....
    with tf.variable_scope('valid'):
        params.reuse_variables()
        valid+net = FRRN(None,None,K,valid_ims,valid_lbs,partial(_arch_type_a,NUM_CLASSES),params,False)
    """

    with tf.variable_scope('misc'):
        # Summary Operations
        tf.summary.scalar('loss',net.loss)
        tf.summary.scalar('lr',learning_rate)
        # TODO: logliklihood

        summary_op = tf.summary.merge_all()

        # Initialize op
        init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(config.as_matrix()), collections=[])

        # Plot summary
        # TODO: Actually, better way exist. (this method is too slow)
        def _py_draw_plot(label):
            fig, ax = plt.subplots()
            img = ax.imshow(label, interpolation='none', cmap='tab20')
            ax.set_axis_off()
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)
            return buf.getvalue()

        def _tf_draw_plots(label):
            print(label.shape)
            png_str = tf.py_func(_py_draw_plot,
                                 [label],
                                 tf.string,
                                 stateful=False)
            return tf.image.decode_png(png_str, channels=4)

        pred_plots = tf.map_fn(_tf_draw_plots,net.preds[:2],dtype=tf.uint8)
        pred_plots = tf.stack(pred_plots,axis=0)
        gt_plots= tf.map_fn(_tf_draw_plots,lbs[:2],dtype=tf.uint8)
        gt_plots = tf.stack(gt_plots,axis=0)

        extended_summary_op = tf.summary.merge([
            tf.summary.image('image',ims[:2],max_outputs=2),
            tf.summary.image('gt',gt_plots,max_outputs=2),
            tf.summary.image('preds',pred_plots,max_outputs=2),
        ])


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)

    summary_writer = tf.summary.FileWriter(LOG_DIR,sess.graph)
    summary_writer.add_summary(config_summary.eval(session=sess))

    try:
        # Start Queueing
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        for step in tqdm(xrange(TRAIN_NUM),dynamic_ncols=True):
            it,loss,_ = sess.run([global_step,net.loss,net.train_op])
            tqdm.write('[%5d] Loss: %1.3f'%(it,loss))

            if( it % SAVE_PERIOD == 0 ):
                net.save(sess,LOG_DIR,step=it)

            if( it % SUMMARY_PERIOD == 0 ):
                summary = sess.run(summary_op)
                summary_writer.add_summary(summary,it)

            if( it % (SUMMARY_PERIOD*2) == 0 ): #Extended Summary
                summary = sess.run(extended_summary_op)
                summary_writer.add_summary(summary,it)

    except Exception as e:
        coord.request_stop(e)
    finally :
        net.save(sess,LOG_DIR)

        coord.request_stop()
        coord.join(threads)

    net.save(sess,LOG_DIR)

"""
def eval(MODEL,
         CROP,
         IM_SIZE,
         **kwargs):
    # >>>>>>> DATASET
    image = get_image(num_epochs=1)
    images = tf.train.batch(
        [image],
        batch_size=100,
        num_threads=1,
        capacity=100,
        allow_smaller_final_batch=True)
    valid_image = get_image(False,num_epochs=1)
    valid_images = tf.train.batch(
        [valid_image],
        batch_size=100,
        num_threads=1,
        capacity=100,
        allow_smaller_final_batch=True)
    # <<<<<<<

    # >>>>>>> MODEL
    with tf.variable_scope('net'):
        with tf.variable_scope('params') as params:
            pass
        x = tf.placeholder(tf.float32,[None,32,32,3])
        net= VQVAE(None,None,BETA,x,K,D,_cifar10_arch,params,False)

    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)
    net.load(sess,MODEL)


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    try:
        nlls = []
        while not coord.should_stop():
            nlls.append(
                sess.run(net.nll,feed_dict={x:sess.run(valid_images)}))
            print('.', end='', flush=True)
    except tf.errors.OutOfRangeError:
        nlls = np.concatenate(nlls,axis=0)
        print(nlls.shape)
        print('NLL for test set: %f bits/dims'%(np.mean(nlls)))

    try:
        nlls = []
        while not coord.should_stop():
            nlls.append(
                sess.run(net.nll,feed_dict={x:sess.run(images)}))
            print('.', end='', flush=True)
    except tf.errors.OutOfRangeError:
        nlls = np.concatenate(nlls,axis=0)
        print(nlls.shape)
        print('NLL for training set: %f bits/dims'%(np.mean(nlls)))

    coord.request_stop()
    coord.join(threads)
"""

def get_default_param():
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        'LOG_DIR':'./log/%s'%(now),

        'TRAIN_NUM' : 100000, #Size corresponds to one epoch
        'BATCH_SIZE': 3,

        'LEARNING_RATE' : 0.001,
        'DECAY_VAL' : 1.0,
        'DECAY_STEPS' : 20000, # Half of the training procedure.
        'DECAY_STAIRCASE' : False,

        'K':512*64,
        'CROP_SIZE':(512,1024),
        'IM_SIZE' :(256,512), #Prediction is made at 1/2 scale.
        'Z_RANGE':0.05,

        'SUMMARY_PERIOD' : 10,
        'SAVE_PERIOD' : 10000,
        'RANDOM_SEED': 0,
    }

if __name__ == "__main__":
    class MyConfig(dict):
        pass
    params = get_default_param()
    config = MyConfig(params)
    def as_matrix() :
        return [[k, str(w)] for k, w in config.items()]
    config.as_matrix = as_matrix

    main(config=config,**config)
    #test(MODEL='models/cifar10/last.ckpt',**config)
