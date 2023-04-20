# -*-coding:utf-8-*-
# @author: Yiqin Qiu
# @email: barryxxz@stu.hqu.edu.cn

from tensorflow.keras.layers import *
import tensorflow.keras.backend as K


class OurLayer(Layer):
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self.trainable_weights:
                self.trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self.non_trainable_weights:
                self.non_trainable_weights.append(w)
        for u in layer.updates:
            if not hasattr(self, '_updates'):
                self._updates = []
            if u not in self._updates:
                self._updates.append(u)
        return outputs


class TrainablePositionEmbedding(OurLayer):
    def __init__(self, maxlen, v_dim, merge_mode='add', **kwargs):
        super(TrainablePositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.v_dim = v_dim
        self.merge_mode = merge_mode

    def build(self, input_shape):
        super(TrainablePositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.maxlen, self.v_dim),
            initializer='zeros'
        )

    def call(self, inputs):
        if isinstance(inputs, list):
            x, r = inputs
        else:
            x, r = inputs, 0
        pid = K.arange(start=0, stop=K.shape(x)[1])
        pid = K.expand_dims(pid, 0)
        pid = K.tile(pid, [K.shape(x)[0], 1])
        pid = K.abs(pid - K.cast(r, 'int32'))
        pv = K.gather(self.embeddings, pid)
        if self.merge_mode == 'add':
            return pv + x
        else:
            return K.concatenate([x, pv])

    def compute_output_shape(self, input_shape):
        if self.merge_mode == 'add':
            return input_shape
        else:
            return (input_shape[0], input_shape[1], input_shape[2] + self.v_dim)
