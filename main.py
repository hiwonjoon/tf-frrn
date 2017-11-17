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

from cityscapesScripts.cityscapesscripts.helpers import labels as L
# For Plotting
color_map = np.zeros((20,3),np.uint8)
for i in range(19):
    color_map[i] = L.trainId2label[i].color


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
    _,ims,lbs = cityspaces.build_queue(target='train',crop=CROP_SIZE,resize=IM_SIZE,z_range=Z_RANGE,batch_size=BATCH_SIZE,num_threads=4)
    """
    _,valid_ims,valid_lbs = cityspaces.build_queue(target='val')
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
            img = ax.imshow(color_map[label])
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

def eval(MODEL,
         TARGET,
         BATCH_SIZE,
         **kwargs):
    if not os.path.exists(os.path.join('results',TARGET)):
        os.makedirs(os.path.join('results',TARGET))
    # >>>>>>> DATASET
    from cityscapesScripts.cityscapesscripts.helpers import labels as L
    from cityscapesScripts.cityscapesscripts.evaluation import evalPixelLevelSemanticLabeling as E
    from PIL import Image

    trainId2id = np.zeros((20,),np.uint8)
    for i in range(19):
        trainId2id[i] = L.trainId2label[i].id
    trainId2id[19] = 0

    cityspaces = CitySpaces()
    scale_factor = np.array(kwargs['CROP_SIZE'])/np.array(kwargs['IM_SIZE'])
    size = tuple(int(l//s) for (l,s) in zip(cityspaces.image_size,scale_factor))
    imnames,ims,lbs = cityspaces.build_queue(target=TARGET,crop=cityspaces.image_size,resize=size,z_range=None,batch_size=BATCH_SIZE,num_threads=2)
    # <<<<<<<

    # >>>>>>> MODEL
    with tf.variable_scope('net'):
        with tf.variable_scope('params') as params:
            pass
        net = FRRN(None,None,kwargs['K'],ims,lbs,partial(_arch_type_a,NUM_CLASSES),params,False)

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
        while not coord.should_stop():
            names,preds = sess.run([imnames,net.preds])
            for name,pred in zip(names,preds):
                name = os.path.basename(str(name,'utf-8'))
                im = Image.fromarray(trainId2id[pred])
                im = im.resize((cityspaces.image_size[1],cityspaces.image_size[0]),
                               Image.NEAREST)
                im.save(os.path.join('results',TARGET,name),"PNG")
            print('.', end='', flush=True)
    except tf.errors.OutOfRangeError:
        print('Complete')

    coord.request_stop()
    coord.join(threads)

    E.main([TARGET])

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
    #eval(MODEL='models/arch_type_a/last.ckpt',TARGET='val',**config)
