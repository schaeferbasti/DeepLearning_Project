import tensorflow as tf
import math

class CNN_ByteNet:
    def __init__(self, options):
        self.options = options
        embedding_channels = 2 * options['residual_channels']

        self.w_source_embedding = tf.get_variable('w_source_embedding',
                                                  [options['source_vocab_size'], embedding_channels],
                                                  initializer=tf.truncated_normal_initializer(stddev=0.02))

        self.w_target_embedding = tf.get_variable('w_target_embedding',
                                                  [options['target_vocab_size'], embedding_channels],
                                                  initializer=tf.truncated_normal_initializer(stddev=0.02))

    def build_options(self):
        options = self.options
        self.source_sentence = tf.placeholder('int32',
                                              [None, None], name='source_sentence')
        self.target_sentence = tf.placeholder('int32',
                                              [None, None], name='target_sentence')

        target_1 = self.target_sentence[:, 0:-1]
        target_2 = self.target_sentence[:, 1:]

        source_embedding = tf.nn.embedding_lookup(self.w_source_embedding,
                                                  self.source_sentence, name="source_embedding")
        target_1_embedding = tf.nn.embedding_lookup(self.w_target_embedding,
                                                    target_1, name="target_1_embedding")

        curr_input = source_embedding
        for layer_no, dilation in enumerate(options['encoder_dilations']):
            curr_input = byetenet_residual_block(curr_input, dilation,
                                                     layer_no, options['residual_channels'],
                                                     options['encoder_filter_width'], causal=False, train=True)

        encoder_output = curr_input
        combined_embedding = target_1_embedding + encoder_output
        curr_input = combined_embedding
        for layer_no, dilation in enumerate(options['decoder_dilations']):
            curr_input = byetenet_residual_block(curr_input, dilation,
                                                     layer_no, options['residual_channels'],
                                                     options['decoder_filter_width'], causal=True, train=True)

        logits = conv1d(tf.nn.relu(curr_input),
                            options['target_vocab_size'], name='logits')
        print
        "logits", logits
        logits_flat = tf.reshape(logits, [-1, options['target_vocab_size']])
        target_flat = tf.reshape(target_2, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_flat, logits=logits_flat)

        self.loss = tf.reduce_mean(loss)
        self.arg_max_prediction = tf.argmax(logits_flat, 1)
        tf.summary.scalar('loss', self.loss)

    def build_translator(self, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        options = self.options
        self.t_source_sentence = tf.placeholder('int32',
                                                [None, None], name='source_sentence')
        self.t_target_sentence = tf.placeholder('int32',
                                                [None, None], name='target_sentence')

        source_embedding = tf.nn.embedding_lookup(self.w_source_embedding,
                                                  self.t_source_sentence, name="source_embedding")
        target_embedding = tf.nn.embedding_lookup(self.w_target_embedding,
                                                  self.t_target_sentence, name="target_embedding")

        curr_input = source_embedding
        for layer_no, dilation in enumerate(options['encoder_dilations']):
            curr_input = byetenet_residual_block(curr_input, dilation,
                                                     layer_no, options['residual_channels'],
                                                     options['encoder_filter_width'], causal=False, train=False)

        encoder_output = curr_input[:, 0:tf.shape(self.t_target_sentence)[1], :]

        combined_embedding = target_embedding + encoder_output
        curr_input = combined_embedding
        for layer_no, dilation in enumerate(options['decoder_dilations']):
            curr_input = byetenet_residual_block(curr_input, dilation,
                                                     layer_no, options['residual_channels'],
                                                     options['decoder_filter_width'], causal=True, train=False)

        logits = conv1d(tf.nn.relu(curr_input),
                            options['target_vocab_size'], name='logits')
        logits_flat = tf.reshape(logits, [-1, options['target_vocab_size']])
        probs_flat = tf.nn.softmax(logits_flat)

        self.t_probs = tf.reshape(probs_flat,
                                  [-1, tf.shape(logits)[1], options['target_vocab_size']])


    def build_model(self):
        options = {
            'source_vocab_size': 250,
            'target_vocab_size': 250,
            'residual_channels': 512,
            'encoder_dilations': [1, 2, 4, 8, 16,
                                  1, 2, 4, 8, 16
                                  ],
            'decoder_dilations': [1, 2, 4, 8, 16,
                                  1, 2, 4, 8, 16
                                  ],
            'encoder_filter_width': 3,
            'decoder_filter_width': 3
        }
        model = ByteNet_Translator(options)
        model.build_options()
        model.build_translator(reuse=True)
        return model


# OPS Functions
def fully_connected(input_, output_nodes, name, stddev=0.02):
    with tf.variable_scope(name):
        input_shape = input_.get_shape()
        input_nodes = input_shape[-1]
        w = tf.get_variable('w', [input_nodes, output_nodes],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable('b', [output_nodes],
                                 initializer=tf.constant_initializer(0.0))
        res = tf.matmul(input_, w) + biases
        return res


# 1d CONVOLUTION WITH DILATION
def conv1d(input_, output_channels,
           dilation=1, filter_width=1, causal=False,
           name="dilated_conv"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [1, filter_width, input_.get_shape()[-1], output_channels],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [output_channels],
                            initializer=tf.constant_initializer(0.0))

        if causal:
            padding = [[0, 0], [(filter_width - 1) * dilation, 0], [0, 0]]
            padded = tf.pad(input_, padding)
            input_expanded = tf.expand_dims(padded, dim=1)
            out = tf.nn.atrous_conv2d(input_expanded, w, rate=dilation, padding='VALID') + b
        else:
            input_expanded = tf.expand_dims(input_, dim=1)
            out = tf.nn.atrous_conv2d(input_expanded, w, rate=dilation, padding='SAME') + b

        return tf.squeeze(out, [1])


def layer_normalization(x, name, epsilon=1e-8, trainable=True):
    with tf.variable_scope(name):
        shape = x.get_shape()
        beta = tf.get_variable('beta', [int(shape[-1])],
                               initializer=tf.constant_initializer(0), trainable=trainable)
        gamma = tf.get_variable('gamma', [int(shape[-1])],
                                initializer=tf.constant_initializer(1), trainable=trainable)

        mean, variance = tf.nn.moments(x, axes=[len(shape) - 1], keep_dims=True)

        x = (x - mean) / tf.sqrt(variance + epsilon)

        return gamma * x + beta


def byetenet_residual_block(input_, dilation, layer_no,
                            residual_channels, filter_width,
                            causal=True, train=True):
    block_type = "decoder" if causal else "encoder"
    block_name = "bytenet_{}_layer_{}_{}".format(block_type, layer_no, dilation)
    with tf.variable_scope(block_name):
        input_ln = layer_normalization(input_, name="ln1", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        conv1 = conv1d(relu1, residual_channels, name="conv1d_1")
        conv1 = layer_normalization(conv1, name="ln2", trainable=train)
        relu2 = tf.nn.relu(conv1)

        dilated_conv = conv1d(relu2, residual_channels,
                              dilation, filter_width,
                              causal=causal,
                              name="dilated_conv"
                              )
        print
        dilated_conv
        dilated_conv = layer_normalization(dilated_conv, name="ln3", trainable=train)
        relu3 = tf.nn.relu(dilated_conv)
        conv2 = conv1d(relu3, 2 * residual_channels, name='conv1d_2')
        return input_ + conv2


def init_weight(dim_in, dim_out, name=None, stddev=1.0):
    return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)


def init_bias(dim_out, name=None):
    return tf.Variable(tf.zeros([dim_out]), name=name)