import tensorflow as tf

class Conv2d(object) :
    def __init__(self,name,input_dim,output_dim,k_h=4,k_w=4,d_h=2,d_w=2,
                 stddev=0.02, data_format='NCHW') :
        with tf.variable_scope(name) :
            assert(data_format == 'NCHW' or data_format == 'NHWC')
            self.w = tf.get_variable('w', [k_h, k_w, input_dim, output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[output_dim], initializer=tf.constant_initializer(0.0))
            if( data_format == 'NCHW' ) :
                self.strides = [1, 1, d_h, d_w]
            else :
                self.strides = [1, d_h, d_w, 1]
            self.data_format = data_format
    def __call__(self,input_var,name=None,w=None,b=None,**kwargs) :
        w = w if w is not None else self.w
        b = b if b is not None else self.b

        if( self.data_format =='NCHW' ) :
            return tf.nn.bias_add(
                        tf.nn.conv2d(input_var, w,
                                    use_cudnn_on_gpu=True,data_format='NCHW',
                                    strides=self.strides, padding='SAME'),
                        b,data_format='NCHW',name=name)
        else :
            return tf.nn.bias_add(
                        tf.nn.conv2d(input_var, w,data_format='NHWC',
                                    strides=self.strides, padding='SAME'),
                        b,data_format='NHWC',name=name)
    def get_variables(self):
        return {'w':self.w,'b':self.b}

class TransposedConv2d(object):
    def __init__(self,name,input_dim,out_dim,
                 k_h=4,k_w=4,d_h=2,d_w=2,stddev=0.02,data_format='NCHW') :
        with tf.variable_scope(name) :
            self.w = tf.get_variable('w', [k_h, k_w, out_dim, input_dim],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[out_dim], initializer=tf.constant_initializer(0.0))

        self.data_format = data_format
        if( data_format =='NCHW' ):
            self.strides = [1, 1, d_h, d_w]
        else:
            self.strides = [1, d_h, d_w, 1]

    def __call__(self,input_var,name=None,**xargs):
        shapes = tf.shape(input_var)
        if( self.data_format == 'NCHW' ):
            shapes = tf.stack([shapes[0],tf.shape(self.b)[0],shapes[2]*2,shapes[3]*2])
        else:
            shapes = tf.stack([shapes[0],shapes[1]*2,shapes[2]*2,tf.shape(self.b)[0]])

        return tf.nn.bias_add(
            tf.nn.conv2d_transpose(input_var,self.w,output_shape=shapes,
                                data_format=self.data_format,
                                strides=self.strides,padding='SAME'),
            self.b,data_format=self.data_format,name=name)

class BatchNorm(object):
    def __init__(self,name,dims,axis=1,epsilon=1e-3,momentum=0.999,center=True,scale=True) :
        self.momentum = momentum
        self.epsilon = epsilon
        self.axis = axis
        self.center=center
        self.scale=scale
        with tf.variable_scope(name) as scope:
            with tf.variable_scope('bn') :
                self.gamma= tf.get_variable('gamma',[dims], initializer=tf.constant_initializer(1.0))
                self.beta = tf.get_variable('beta',[dims], initializer=tf.constant_initializer(0.0))
                self.moving_mean = tf.get_variable('moving_mean',[dims], initializer=tf.constant_initializer(0.0), trainable=False)
                self.moving_variance = tf.get_variable('moving_variance',[dims], initializer=tf.constant_initializer(1.0), trainable=False)
        self.scope = scope

    def __call__(self,input_var,is_training=True,**xargs) :
        with tf.variable_scope(self.scope) :
            return tf.layers.batch_normalization(
                input_var,
                axis=self.axis,
                momentum=self.momentum,
                epsilon=self.epsilon,
                center=self.center,
                scale=self.scale,
                training=is_training,
                reuse=True,
                name='bn')
        """
        ---Do NOT forget to add update_ops dependencies for your loss function.---
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,tf.get_default_graph().get_name_scope())
        #And, do not make any scope inside map_fn, since scope.name will not work...(it is corrupted by map_fn.)
        print(update_ops)
        with tf.control_dependencies(update_ops):
        """
    def get_variables(self):
        return {}
