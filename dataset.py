from six.moves import xrange
import numpy as np
import tensorflow as tf
import random
import os, glob
from tqdm import tqdm

class CitySpaces():
    def __init__(self,
                 data_dir='datasets/cityspaces',
                 image_size=(1024,2048),
                 seed=0):
        images = {
            'train':None,
            'test':None,
            'val':None
        }
        labels = {
            'train':None,
            'test':None,
            'val':None
        }
        for target in ['train','test','val']:
            searchFine = os.path.join( data_dir, "gtFine", target , "*" , "*labelTrainIds*.png" )
            labels[target] = sorted(glob.glob( searchFine ))

            searchFine = os.path.join( data_dir, "leftImg8bit", target , "*" , "*leftImg8bit*.png" )
            images[target] = sorted(glob.glob( searchFine ))


            for l,i in zip(labels[target],images[target]):
                assert( ''.join(os.path.basename(l).split('_')[:2]) ==
                                ''.join(os.path.basename(i).split('_')[:2])), (l,i)

        assert(len(labels['train']) == 2975 and len(images['train']) == 2975)
        assert(len(labels['val']) == 500 and len(images['val']) == 500)
        assert(len(labels['test']) == 1525 and len(images['test']) == 1525)

        self.images = images
        self.image_size = image_size
        self.labels = labels

    def build_queue(self,target='train',crop=(128,256),resize=(128,256),z_range=0.05,batch_size=2,num_threads=1):
        with tf.device('/cpu'):
            im_name,l_name = tf.train.slice_input_producer([self.images[target],self.labels[target]],num_epochs=None,shuffle=True)
            binary = tf.read_file(im_name)
            image = tf.image.decode_png(binary,channels=3)
            binary = tf.read_file(l_name)
            label = tf.image.decode_png(binary,channels=1)

            # TODO: when validation and test, use different crops.
            cropped = tf.random_crop(tf.concat([image,label],axis=2),list(crop)+[4])
            cropped_im,cropped_label = tf.split(cropped,[3,1],axis=2)

            resized_im = tf.image.resize_images(cropped_im,resize)
            resized_label = tf.image.resize_images(cropped_label,resize,tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            if( target == 'train' ):
                pp = tf.image.random_flip_left_right(resized_im)

                # Gamma augmentation; formula (14)
                z = tf.random_uniform([],minval=-1.*z_range,maxval=z_range)
                gamma = tf.log(0.5+2**(-0.5)*z) / tf.log(0.5-2**(-0.5)*z)
                pp = (tf.cast(pp,tf.float32) / 255.0)**(gamma)
            else :
                pp = (tf.cast(resized_im,tf.float32) / 255.0)

            # convert 255 to label 19.
            mask = tf.cast(tf.equal(resized_label, 255),tf.int32)
            resized_label = mask * 19 + (1-mask) * tf.cast(resized_label,tf.int32)
            resized_label = tf.squeeze(resized_label,axis=2)

            # Build task batch
            x, y = tf.train.batch(
                [pp, resized_label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=10*batch_size)
            return x,y

if __name__ == "__main__":
    cityspaces = CitySpaces()

    images,labels = cityspaces.build_queue(target='train')

    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)

    import itertools
    try:
        # Start Queueing
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        for _it in tqdm(itertools.count()) : # Slice Input producer will throw OutOfRange exception
            if( coord.should_stop() ): break
            ims,las = sess.run([images,labels])
            print(ims.shape,np.min(ims),np.max(ims),las.shape,np.min(las),np.max(las))
    except Exception as e:
        coord.request_stop(e)
    finally :
        coord.request_stop()
        coord.join(threads)

