from numpy import ndarray
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

def cos(x,y):
    """
    x,y are two-dim matixe
    """
    normx = tf.sqrt(tf.reduce_sum(tf.square(x),axis=-1 , keepdims = True))
    normy = tf.sqrt(tf.reduce_sum(tf.square(y),axis=-1 , keepdims = True))
    norm  = tf.matmul(normx,normy,transpose_b = True)
    prd   = tf.matmul(x,y,transpose_b = True)
    return prd /(norm+1e-8)

def get_shape(tensor):
    s_shape = tensor.get_shape().as_list()
    d_shape = tf.unstack(tf.shape(tensor))
    shape = []
    for i,s in enumerate(s_shape):
        if s is None:
            shape.append(d_shape[i])
        else:
            shape.append(s)
    return shape

def build_embedding_table(vec_or_shape,scope = ''):
    if isinstance(vec_or_shape,ndarray):
        table = tf.get_variable(name = scope+'/embedding_table',
                                shape = vec_or_shape.shape,
                                initializer = tf.constant_initializer(value=vec_or_shape,
                                                                      dtype= tf.float32,
                                                                      verify_shape = False))
    elif isinstance(vec_or_shape,(list,tuple)):
        table = tf.get_variable(name = scope+'/embedding_table',
                                shape = vec_or_shape,
                                initializer = tf.random_uniform_initializer(-1e-2,1e-2))
    return table

def lookup(table,ids):
    table_ = tf.concat([tf.zeros_like(table[0:1,:]),
                        table[1:,:]],axis = 0,name='build_lookup_table')
    return tf.nn.embedding_lookup(table_,ids,name='lookup_op')

def sequen_mask(tensor):
    indices = tf.reduce_sum(tf.abs(tensor),axis=-1)
    ones    = tf.ones_like(indices,dtype = tf.float32)
    zeros   = tf.zeros_like(indices,dtype = tf.float32)
    lengths = tf.where(tf.equal(indices,0.),zeros,ones)
    return lengths

def actual_length_test():
    x = tf.constant([[[1.,2.,3.,0.,0.],[4. , 5. , 6. , 0. , 0.]],
                     [[1.,2.,3.,4.,0.],[4. , 5. , 6. , 7. , 0.]]])
    y = sequen_mask(x)
    sess = tf.Session()
    ret = sess.run(y)
    return ret


def cnn_extractor(inputs,units,kernels=[2,3,4,5],training=True,keep_prob=.5,scope='cnn_extractor'):
    cnns = []
    with tf.variable_scope(scope,reuse = tf.get_variable_scope().reuse):
        for i,ks in enumerate(kernels):
            cnv = tf.layers.conv1d(inputs      = inputs , 
                                   filters     = 2 * units//len(kernels),
                                   kernel_size = ks,
                                   strides     = 1,
                                   activation  = tf.nn.elu,
                                   name        = 'conv_size%d'%ks,
                                   padding     = 'same',
                                   kernel_initializer = xavier_initializer())
            cnv = tf.layers.dropout(cnv,training = training, rate = keep_prob)
            cnns.append(cnv)
        cnn_out = tf.concat(cnns,axis=-1,name = 'concate')
        res_cnn = tf.layers.dense(inputs,units = units * 2 , activation = None,
                                 name = 'res_inputTocnn')
        cnn_states = cnn_out + res_cnn
    return cnn_states

def _bilstm(inputs,lens,units,training = True,keep_prob=.5,scope='bilstm_extractor'):
    with tf.variable_scope(scope,reuse = tf.get_variable_scope().reuse):
        if training:
            forget_bias = 1.
        else:
            forget_bias = 0.
        
        fcell = tf.nn.rnn_cell.LSTMCell(num_units = units  , activation = tf.nn.tanh,
                                        name = 'forecell',forget_bias = forget_bias)
        bcell = tf.nn.rnn_cell.LSTMCell(num_units = units  , activation = tf.nn.tanh,
                                        name = 'backcell',forget_bias = forget_bias)
        if training:
            
            fcell = tf.nn.rnn_cell.DropoutWrapper(fcell,output_keep_prob = keep_prob)
            bcell = tf.nn.rnn_cell.DropoutWrapper(bcell,output_keep_prob = keep_prob)
        """
            new change , inputs changed to cnn_states , oringal is inputs
        """
        states,finals = tf.nn.bidirectional_dynamic_rnn(cell_bw         = bcell,
                                                        cell_fw         = fcell,
                                                        inputs          = inputs,
                                                        sequence_length = lens,
                                                        dtype           = tf.float32,
                                                        scope           = 'recurrent')
        states = tf.concat(states,axis=-1,name = 'concat')
    return states


def _encoder(inputs,lens,units,training=True,keep_prob=.5,scope='encoding'):
    
    states    = _bilstm(inputs,lens,units,training = training,
                        keep_prob=keep_prob,scope=scope+'/bilstm1')
    inp_units = inputs.get_shape().as_list()[-1]
    out_units = states.get_shape().as_list()[-1]
    if inp_units != out_units:
        res   = tf.layers.conv1d(inputs,filters = out_units,kernel_size=1,strides=1,
                                name = 'resconnection',padding='valid',
                                activation = None,kernel_initializer=xavier_initializer())
    else:
        res   = inputs
        
    states1   = states+res
    states2   = _bilstm(states1,lens,units,training=training,
                        keep_prob=keep_prob,scope = scope+'/bilstm2')
    output    = tf.concat(states2,axis=-1,name='concat')
    return output



def _mlp(inputs,hidden_size,output_size,hidden_activation,name):
    with tf.variable_scope(name,reuse = tf.AUTO_REUSE):
        hidden = tf.layers.dense(inputs,units=hidden_size,
                                 activation=hidden_activation,
                                 name = name+'/hidden',
                                 kernel_initializer=xavier_initializer())
        output = tf.layers.dense(hidden,units=output_size,activation=None,
                                 name = name+'/output',
                                 kernel_initializer=xavier_initializer())
    return output

def channel_attention(inputs,ratio=4,scope='channel_wise'):
    """
        inputs is a 3-d tensor with shape [batch_size,max_len,units]
    """
    output_size = inputs.get_shape().as_list()[-1]
    hidden_size =  output_size // ratio
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        avg_pool  = tf.reduce_mean(inputs,axis=1,keepdims=True,name='avg_pool')
        max_pool  = tf.reduce_max(inputs,axis =1,keepdims=True,name='max_pool')
        avg_score = _mlp(avg_pool,hidden_size,output_size,tf.nn.relu,name='shared_mlp')
        max_score = _mlp(max_pool,hidden_size,output_size,tf.nn.relu,name='shared_mlp')
        weights   = tf.nn.sigmoid(avg_score+max_score,name='activation')
    return weights

def spatial_attentoin(inputs,window=3,scope='spatial_attention'):
    with tf.variable_scope(scope,reuse = tf.AUTO_REUSE):
        avg_pool   = tf.reduce_mean(inputs,axis=-1,keepdims=True,name='avg_pool')
        max_pool   = tf.reduce_max(inputs,axis=-1,keepdims =True,name='max_pool')
        descriptor = tf.concat([avg_pool,max_pool],axis=-1,name='get_descriptor')
        score      = tf.layers.conv1d(descriptor,filters=1,
                                      kernel_size=window,strides=1,
                                      kernel_initializer=xavier_initializer(),
                                      name = scope+'/scoring',
                                      padding='same',activation=None)
        weights    = tf.nn.sigmoid(score,name='activation')
    return weights

def attention(inputs,ratio=4,window=3,scope='attention'):
    """
        combien the channle attention and spatial attention
    """
    with tf.variable_scope(scope,reuse = tf.AUTO_REUSE):
        channel_weights = channel_attention(inputs,ratio,scope='channel_attention')
        inputs_dash     = tf.multiply(channel_weights , inputs,name='apply_channel_attention')
        spatial_weights = spatial_attentoin(inputs_dash,window,scope='spatial_attention')
        output          = tf.matmul(spatial_weights, inputs_dash, transpose_a = True, name = 'apply_spatial_attention')
        weights         = tf.squeeze(spatial_weights,axis=-1,name='reshape_weights')
        output          = tf.squeeze(output,axis=1,name='reshape_output')
    return weights,output
