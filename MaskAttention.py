from tensorflow.keras import layers
import tensorflow as tf


class SimpleVectorMask(layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleVectorMask, self).__init__(**kwargs)
        self.mask_weight = None

    def build(self, input_shape):
        self.mask_weight = self.add_weight(name='mask_weights',
                                           shape=[input_shape[2], ],
                                           initializer=tf.initializers.Constant(0.1),
                                           regularizer='l1_l2',
                                           trainable=True)
        self.build = True

    def call(self, x, trainable=None):
        w = self.mask_weight
        m = tf.matmul(x, tf.reshape(w, shape=(w.shape[0], 1)))
        m = layers.Flatten()(m)
        return m


class MiddlePointMaskWithoutBias(layers.Layer):
    def __init__(self, power=2., **kwargs):
        super(MiddlePointMaskWithoutBias, self).__init__(**kwargs)
        self.power = power
        self.stdWordEmb = None
        self.mask_weight = None

    def build(self, input_shape):
        self.stdWordEmb = self.add_weight(name='standard_word_embedding',
                                          shape=[1, input_shape[2]],
                                          initializer=tf.initializers.Constant(0.5),
                                          # regularizer='l2',
                                          trainable=True
                                          )
        self.mask_weight = self.add_weight(name='word_embedding_weight',
                                           shape=[input_shape[2], ],
                                           initializer=tf.initializers.Constant(0.1),
                                           regularizer='l1',
                                           trainable=True
                                           )
        self.built = True

    def call(self, x, trainable=None):
        x2 = (x - self.stdWordEmb)
        x2 = tf.pow(x2, self.power)
        ebw = self.mask_weight
        x2 = ebw * x2
        out = (1 - tf.reduce_sum(x2, axis=-1) / tf.reduce_sum(self.Embweight))
        return out


class MiddlePointMaskWithBias(layers.Layer):
    def __init__(self, power=2., **kwargs):
        super(MiddlePointMaskWithBias, self).__init__(**kwargs)
        self.power = power
        self.stdWordEmb = None
        self.mask_weight = None
        self.mask_bias = None

    def build(self, input_shape):
        # 保存标准词向量
        self.mask_weight = self.add_weight(name='word_embedding_weight',
                                           shape=[1, input_shape[2]],
                                           initializer=tf.initializers.Constant(0.1),
                                           regularizer='l1_l2',
                                           trainable=True
                                           )
        self.stdWordEmb = self.add_weight(name='standard_word_embedding',
                                          shape=[input_shape[2], ],
                                          initializer=tf.initializers.Constant(0.1),
                                          #regularizer='l1',
                                          trainable=True
                                          )
        self.mask_bias = self.add_weight(name='embedding_bias',
                                         shape=[1, input_shape[2]],
                                         initializer=tf.initializers.Constant(0.),
                                         #regularizer='l2',
                                         trainable=True
                                         )
        self.built = True

    def call(self, x, trainable=None):
        d = tf.abs(x - self.stdWordEmb)
        d = tf.pow(d, self.power)
        s = self.mask_weight * (self.mask_bias - d)
        out = tf.reduce_sum(s, axis=-1)
        return out


class AreaRangeMaskWithoutBias(layers.Layer):
    def __init__(self, power=0.5, **kwargs):
        super(AreaRangeMaskWithoutBias, self).__init__(**kwargs)
        self.power = power
        self.maxStdWordEmb = None
        self.minStdWordEmb = None
        self.mask_weight = None

    def build(self, input_shape):
        self.maxStdWordEmb = self.add_weight(name='max_standard_word_embedding',
                                             shape=[1, input_shape[2]],
                                             initializer=tf.initializers.Constant(1.),
                                             regularizer='l2',
                                             trainable=True
                                             )
        self.minStdWordEmb = self.add_weight(name='min_standard_word_embedding',
                                             shape=[1, input_shape[2]],
                                             initializer=tf.initializers.Constant(-1.),
                                             regularizer='l2',
                                             trainable=True
                                             )
        self.mask_weight = self.add_weight(name='word_embeding_weight',
                                           shape=[1, input_shape[2]],
                                           initializer=tf.initializers.Constant(0.),
                                           #regularizer='l1_l2',
                                           trainable=True
                                           )
        self.built = True

    def call(self, x, trainable=None):
        Mx = tf.abs(self.maxStdWordEmb - x)
        mx = tf.abs(x - self.minStdWordEmb)
        Mm = tf.pow(tf.abs(self.maxWE - self.minWE), self.power)
        wx = tf.exp(1.) / tf.exp(Mm * (Mx + mx))
        ebw = self.Embweight
        x2 = ebw * wx
        out = tf.reduce_sum(x2, axis=-1)
        return out


class AreaRangeMaskWithBias(layers.Layer):
    def __init__(self, power=0.5, **kwargs):
        super(AreaRangeMaskWithBias, self).__init__(**kwargs)
        self.power = power
        self.maxStdWordEmb = None
        self.minStdWordEmb = None
        self.mask_weight = None
        self.mask_bias = None

    def build(self, input_shape):
        self.maxStdWordEmb = self.add_weight(name='max_standard_word_embeding',
                                             shape=[1, input_shape[2]],
                                             initializer=tf.initializers.Constant(1.),
                                             regularizer='l2',
                                             trainable=True
                                             )
        self.minStdWordEmb = self.add_weight(name='min_standard_word_embeding',
                                             shape=[1, input_shape[2]],
                                             initializer=tf.initializers.Constant(-1.),
                                             regularizer='l2',
                                             trainable=True
                                             )
        self.mask_weight = self.add_weight(name='word_embeding_weight',
                                           shape=[1, input_shape[2]],
                                           initializer=tf.initializers.Constant(0.),
                                           #regularizer='l1_l2',
                                           trainable=True
                                           )
        self.mask_bias = self.add_weight(name='embedding_bias',
                                         shape=[1, input_shape[2]],
                                         initializer=tf.initializers.Constant(0.),
                                         #regularizer='l2',
                                         trainable=True
                                         )
        self.built = True

    def call(self, x, trainable=None):
        d = tf.abs(x - self.minStdWordEmb) + tf.abs(self.maxStdWordEmb - x)
        d = tf.pow(d, self.power)
        s = self.mask_weight * (self.mask_bias - d)
        out = tf.reduce_sum(s, axis=-1)
        return out


class WordsQueryMask(layers.Layer):
    def __init__(self, usedk=True, **kwargs):
        super(WordsQueryMask, self).__init__(**kwargs)
        self.usedk = usedk
        self.wq = None
        self.wk = None
        self.dk = None

    def build(self, input_shape):
        self.wq = self.add_weight(name='query_weight',
                                  shape=[input_shape[1], input_shape[1]],
                                  #                                   initializer=tf.initializers.zeros(),
                                  regularizer='l2',
                                  trainable=True)
        self.wk = self.add_weight(name='key_weight',
                                  shape=[1, input_shape[1]],
                                  #                                   initializer=tf.initializers.zeros(),
                                  regularizer='l2',
                                  trainable=True)
        self.dk = input_shape[2]
        self.build = True

    def call(self, x, trainable=None):
        q = tf.matmul(self.wq, x)
        k = tf.matmul(self.wk, x)
        qkt = tf.matmul(q, tf.transpose(k, [0, 2, 1]))
        if self.usedk:
            qkt = qkt / tf.pow(float(self.dk), 0.5)
        qkt = layers.Flatten()(qkt)
        return qkt


class EmbeddingQueryMask(layers.Layer):
    def __init__(self, usedk=True, **kwargs):
        super(EmbeddingQueryMask, self).__init__(**kwargs)
        self.usedk = usedk
        self.wq = None
        self.wk = None
        self.dk = None

    def build(self, input_shape):
        self.wq = self.add_weight(name='query_weight',
                                  shape=[input_shape[2], input_shape[1]],
#                                   initializer=tf.initializers.zeros(),
                                  regularizer='l2',
                                  trainable=True)
        self.wk = self.add_weight(name='key_weight',
                                  shape=[input_shape[2], 1],
#                                   initializer=tf.initializers.zeros(),
                                  regularizer='l2',
                                  trainable=True)
        self.dk = input_shape[2]
        self.build = True

    def call(self, x, trainable=None):
        q = tf.matmul(x, self.wq)
        k = tf.matmul(x, self.wk)
        qkt = tf.matmul(q, k)
        if self.usedk:
            qkt = qkt/tf.pow(float(self.dk), 0.5)
        qkt = layers.Flatten()(qkt)
        return qkt