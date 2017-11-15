from six.moves import xrange
import better_exceptions
import tensorflow as tf
import numpy as np
from commons.ops import *

def _arch_type_a(num_classes):
    def _ru(t,conv3_1,bn_1,conv3_2,bn_2):
        _t = conv3_1(t)
        _t = bn_1(_t)
        _t = tf.nn.relu(_t)
        _t = conv3_2(_t)
        _t = bn_2(_t)
        return t + _t

    def _frru(y_z,conv3_1,bn_1,conv3_2,bn_2,conv1,scale):
        #tf.nn.max_pool(, ksize, strides, padding, data_format='NHWC', name=None)
        y,z = y_z

        _t = tf.concat([y,
                        tf.nn.max_pool(z,[1,scale,scale,1],[1,scale,scale,1],'SAME','NHWC')],axis=3)
        _t = conv3_1(_t)
        _t = bn_1(_t)
        _t = tf.nn.relu(_t)
        _t = conv3_2(_t)
        _t = bn_2(_t)
        y_prime = tf.nn.relu(_t)

        _t = conv1(y_prime)
        _t = tf.image.resize_nearest_neighbor(_t, tf.shape(y_prime)[1:3]*scale)
        z_prime = _t + z

        return y_prime,z_prime

    def _divide_stream(t,conv1):
        z = conv1(t)
        return t,z
    def _concat_stream(y_z,conv1):
        y,z = y_z
        t = tf.concat([tf.image.resize_bilinear(y, tf.shape(y)[1:3]*2), z],axis=3)
        return conv1(t)

    from functools import partial
    # The First Conv
    spec = [
        Conv2d('conv2d_1',3,48,5,5,1,1,data_format='NHWC'),
        BatchNorm('conv2d_1_bn',48,axis=3),
        lambda t,**kwargs : tf.nn.relu(t)]
    # RU Layers
    for i in range(3):
        spec.append(
            partial(_ru,
                    conv3_1=Conv2d('ru48_%d_1'%i,48,48,3,3,1,1,data_format='NHWC'),
                    bn_1 = BatchNorm('ru48_%d_1_bn'%i,48,axis=3),
                    conv3_2=Conv2d('ru48_%d_2'%i,48,48,3,3,1,1,data_format='NHWC'),
                    bn_2 = BatchNorm('ru48_%d_2_bn'%i,48,axis=3))
        )
    # Split Streams
    spec.append(
        partial(_divide_stream,
                conv1 = Conv2d('conv32',48,32,1,1,1,1,data_format='NHWC'))
    )
    # FFRU Layers (Encoding)
    prev_ch = 48
    for it,ch,scale in [(3,96,2),(4,192,4),(2,384,8),(2,384,16)] :
        spec.append(
            lambda y_z : (tf.nn.max_pool(y_z[0],[1,2,2,1],[1,2,2,1],'SAME','NHWC'),y_z[1]) #maxpooling y only.
        )
        for i in range(it):
            spec.append(
                partial(_frru,
                        conv3_1=Conv2d('encode_frru%d_%d_%d_1'%(ch,scale,i),prev_ch+32,ch,3,3,1,1,data_format='NHWC'),
                        bn_1 = BatchNorm('encode_frru%d_%d_%d_1_bn'%(ch,scale,i),ch,axis=3),
                        conv3_2=Conv2d('encode_frru%d_%d_%d_2'%(ch,scale,i),ch,ch,3,3,1,1,data_format='NHWC'),
                        bn_2 = BatchNorm('encode_frru%d_%d_%d_2_bn'%(ch,scale,i),ch,axis=3),
                        conv1 = Conv2d('encode_frru%d_%d_%d_3'%(ch,scale,i),ch,32,1,1,1,1,data_format='NHWC'),
                        scale=scale)
            )
            prev_ch = ch
    # FRRU Layers (Decoding)
    for it,ch,scale in [(2,192,8),(2,192,4),(2,96,2)] :
        spec.append(
            lambda y_z : (tf.image.resize_bilinear(y_z[0], tf.shape(y_z[0])[1:3]*2), y_z[1])
        )
        for i in range(it):
            spec.append(
                partial(_frru,
                        conv3_1=Conv2d('decode_frru%d_%d_%d_1'%(ch,scale,i),prev_ch+32,ch,3,3,1,1,data_format='NHWC'),
                        bn_1 = BatchNorm('decode_frru%d_%d_%d_1_bn'%(ch,scale,i),ch,axis=3),
                        conv3_2=Conv2d('decode_frru%d_%d_%d_2'%(ch,scale,i),ch,ch,3,3,1,1,data_format='NHWC'),
                        bn_2 = BatchNorm('decode_frru%d_%d_%d_2_bn'%(ch,scale,i),ch,axis=3),
                        conv1 = Conv2d('decode_frru%d_%d_%d_3'%(ch,scale,i),ch,32,1,1,1,1,data_format='NHWC'),
                        scale=scale)
            )
            prev_ch = ch
    # Concat Streams
    spec.append(
        partial(_concat_stream,
                conv1 = Conv2d('conv48',prev_ch+32,48,1,1,1,1,data_format='NHWC')))
    # RU Layers
    for i in range(3,6):
        spec.append(
            partial(_ru,
                    conv3_1=Conv2d('ru48_%d_1'%i,48,48,3,3,1,1,data_format='NHWC'),
                    bn_1 = BatchNorm('ru48_%d_1_bn'%i,48,axis=3),
                    conv3_2=Conv2d('ru48_%d_2'%i,48,48,3,3,1,1,data_format='NHWC'),
                    bn_2 = BatchNorm('ru48_%d_2_bn'%i,48,axis=3))
        )
    # Final Classification Layer
    spec.append(
        Conv2d('conv_c',48,num_classes,1,1,1,1,data_format='NHWC'))

    return spec

class FRRN():
    def __init__(self,lr,global_step,K,
                 im,gt,arch_fn,
                 param_scope,is_training=False):
        with tf.variable_scope(param_scope):
             net_spec = arch_fn()

        with tf.variable_scope('forward'):
            _t = im
            for block in net_spec:
                print(_t)
                _t = block(_t)
            self.logits = _t
            self.preds = tf.argmax(self.logits,axis=3)

            # Loss
            naive_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=gt)
            # TODO: ignore pixels labed as void? is it requried?
            # mask = tf.logical_not(tf.equal(gt,0))
            # naive_loss = naive_loss * mask
            boot_loss,_ = tf.nn.top_k(tf.reshape(naive_loss,[tf.shape(im)[0],tf.shape(im)[1]*tf.shape(im)[2]]),k=K,sorted=False)
            self.loss = tf.reduce_mean(tf.reduce_sum(boot_loss,axis=1))
        if( is_training ):
            with tf.variable_scope('backward'):
                optimizer = tf.train.AdamOptimizer(lr)
                self.train_op= optimizer.minimize(self.loss,global_step=global_step)

        save_vars = {('train/'+'/'.join(var.name.split('/')[1:])).split(':')[0] : var for var in
                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,param_scope.name) }
        #for name,var in save_vars.items():
        #    print(name,var)

        self.saver = tf.train.Saver(var_list=save_vars,max_to_keep = 3)

    def save(self,sess,dir,step=None):
        if(step is not None):
            self.saver.save(sess,dir+'/model.ckpt',global_step=step)
        else :
            self.saver.save(sess,dir+'/last.ckpt')

    def load(self,sess,model):
        self.saver.restore(sess,model)


if __name__ == "__main__":
    with tf.variable_scope('params') as params:
        pass

    im = tf.placeholder(tf.float32,[None,256,512,3])
    gt = tf.placeholder(tf.int32,[None,256,512]) #19 + unlabeled area(plus ignored labels)
    global_step = tf.Variable(0, trainable=False)

    from functools import partial
    net = FRRN(0.1,global_step,512*64,im,gt,partial(_arch_type_a,20),params,True)
    print(net.logits)

    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)

    for _ in range(30):
        _t,preds,_ = (sess.run([net.logits,net.preds,net.train_op],
                               feed_dict={im:np.random.random((1,256,512,3)),
                                          gt:np.zeros((1,256,512))}))
        print(preds.shape)

