# -*-coding:utf-8-*-
# @author: Yiqin Qiu
# @email: barryxxz@stu.hqu.edu.cn

import tensorflow as tf
from utils import TrainablePositionEmbedding
from tensorflow.keras.layers import *
from tensorflow.keras import layers, initializers
from tensorflow.keras.models import Model


class ConvStem(layers.Layer):
    def __init__(self, in_channel, **kwargs):
        super(ConvStem, self).__init__(**kwargs)
        self.conv_pool = Conv2D(in_channel, 2, 2, kernel_initializer=initializers.he_normal)
        self.ln = LayerNormalization()
        self.gelu = Activation(tf.nn.gelu)

    def call(self, inputs, **kwargs):
        down_inputs = self.conv_pool(inputs)
        return self.gelu(self.ln(down_inputs))


class PAEBlock(layers.Layer):
    def __init__(self, in_channel, **kwargs):
        super(PAEBlock, self).__init__(**kwargs)
        self.pool = GlobalAveragePooling2D()
        self.fc1 = Dense(in_channel // 2, use_bias=False, kernel_initializer=initializers.he_normal)
        self.relu = Activation(tf.nn.relu)
        self.fc2 = Dense(in_channel, use_bias=False, kernel_initializer=initializers.he_normal)
        self.sigmoid1 = Activation(tf.nn.sigmoid)
        self.conv = Conv2D(1, 1, use_bias=False, kernel_initializer=initializers.he_normal)
        self.sigmoid2 = Activation(tf.nn.sigmoid)
        self.multi = Multiply()

    def call(self, inputs, **kwargs):
        spatial_scale = self.sigmoid1(self.conv(inputs))
        inputs = self.multi([inputs, spatial_scale])
        channel_squeeze = tf.reshape(self.pool(inputs), (-1, 1, 1, inputs.shape[-1]))
        channel_scale = self.sigmoid2(self.fc2(self.relu(self.fc1(channel_squeeze))))
        return self.multi([inputs, channel_scale])


def residual_block(inputs, filters, stride=1, re_sample=False, with_weight=False):
    x = LayerNormalization()(inputs)
    x = Activation(tf.nn.gelu)(x)
    x = SeparableConv2D(filters, 3, strides=stride, padding="same", kernel_initializer=initializers.he_normal)(x)

    x = LayerNormalization()(x)
    x = Activation(tf.nn.gelu)(x)
    x = SeparableConv2D(filters, 3, padding="same", kernel_initializer=initializers.he_normal)(x)

    if with_weight:
        x = PAEBlock(filters)(x)
    if re_sample:
        short_cut = Conv2D(filters, 1, strides=stride, kernel_initializer=initializers.he_normal)(inputs)
        out = Add()([x, short_cut])
    else:
        out = Add()([x, inputs])
    return out


class GatedFFN(layers.Layer):
    def __init__(self, hidden_units, **kwargs):
        super(GatedFFN, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.linear1 = Dense(hidden_units * 2, use_bias=False, kernel_initializer=initializers.he_normal)
        self.gelu1 = Activation(tf.nn.gelu)
        self.linear2 = Dense(hidden_units * 2, use_bias=False, kernel_initializer=initializers.he_normal)
        self.linear3 = Dense(hidden_units, use_bias=False, kernel_initializer=initializers.he_normal)

    def call(self, inputs, **kwargs):
        x = self.gelu1(self.linear1(inputs)) * self.linear2(inputs)
        return self.linear3(x)


def SepSteNet(input_shape):
    original_x = Input(shape=input_shape)
    x = Lambda(lambda tensor: tf.reshape(tensor, (-1, tensor.shape[1] * 4, 10)))(original_x)

    x = Embedding(40, 16)(x)

    x = residual_block(x, filters=128, re_sample=True, with_weight=True)
    x = residual_block(x, filters=128, with_weight=True)
    x = residual_block(x, filters=128, with_weight=True)

    x = residual_block(x, filters=256, re_sample=True, with_weight=True)
    x = residual_block(x, filters=256, with_weight=True)
    x = residual_block(x, filters=256, with_weight=True)

    x = Lambda(lambda tensor: tf.reshape(tensor, (-1, tensor.shape[1], tensor.shape[2] * tensor.shape[3])))(x)

    x = Conv1D(512, 1, kernel_initializer=initializers.he_normal)(x)
    x = LayerNormalization()(x)
    x = Activation(tf.nn.gelu)(x)

    x1 = TrainablePositionEmbedding(x.shape[1], x.shape[2])(x)

    for _ in range(1):
        x = LayerNormalization()(x1)
        x = MultiHeadAttention(12, 64)(x, x, x)

        x = LayerNormalization()(x)
        x = GatedFFN(x.shape[-1])(x)
        x = Dropout(0.2)(x)

    x = GlobalAveragePooling1D()(x)

    logit = Dense(2, activation='softmax')(x)

    return Model(inputs=original_x, outputs=logit)


def backbone(input_shape):
    inputs = Input(shape=input_shape)
    x = Lambda(lambda tensor: tf.reshape(tensor, (-1, tensor.shape[1] * 4, 10)))(inputs)

    x = Embedding(40, 16)(x)

    x = residual_block(x, filters=128, re_sample=True, with_weight=True)
    x = residual_block(x, filters=128, with_weight=True)
    x1 = residual_block(x, filters=128, with_weight=True)

    x = residual_block(x, filters=256, re_sample=True, with_weight=True)
    x = residual_block(x, filters=256, with_weight=True)
    x2 = residual_block(x, filters=256, with_weight=True)

    x = Lambda(lambda tensor: tf.reshape(tensor, (-1, tensor.shape[1], tensor.shape[2] * tensor.shape[3])))(x)

    x = Conv1D(512, 1, kernel_initializer=initializers.he_normal)(x)
    x = LayerNormalization()(x)
    x = Activation(tf.nn.gelu)(x)

    x = TrainablePositionEmbedding(x.shape[1], x.shape[2])(x)

    x = LayerNormalization()(x)
    x3 = MultiHeadAttention(12, 64)(x, x, x)

    x = LayerNormalization()(x)
    x = GatedFFN(x.shape[-1])(x)
    x4 = Dropout(0.2)(x)

    return Model(inputs=inputs, outputs=[x1, x2, x3, x4])


def DPES(input_shape):
    original_x = Input(shape=input_shape)
    calibrate_x = Input(shape=input_shape)

    """
    modifying the obtained output list from backbone(), and replace it with your backbone network
    """
    os_branch = backbone(input_shape=input_shape)
    cs_branch = backbone(input_shape=input_shape)
    x1, x2, x3, x4 = os_branch(original_x)
    y1, y2, y3, y4 = cs_branch(calibrate_x)

    x1 = GlobalAveragePooling2D()(x1)
    x2 = GlobalAveragePooling2D()(x2)
    x3 = GlobalAveragePooling1D()(x3)
    x4 = GlobalAveragePooling1D()(x4)
    y1 = GlobalAveragePooling2D()(y1)
    y2 = GlobalAveragePooling2D()(y2)
    y3 = GlobalAveragePooling1D()(y3)
    y4 = GlobalAveragePooling1D()(y4)

    dis1 = Concatenate()([x1, y4])
    dis2 = Concatenate()([x2, y3])
    dis3 = Concatenate()([x3, y2])
    dis4 = Concatenate()([x4, y1])

    dis1 = Dense(128, kernel_initializer='he_normal')(dis1)
    dis2 = Dense(128, kernel_initializer='he_normal')(dis2)
    dis3 = Dense(128, kernel_initializer='he_normal')(dis3)
    dis4 = Dense(128, kernel_initializer='he_normal')(dis4)

    out = Concatenate()([dis1, dis2, dis3, dis4])

    logit = Dense(2, activation='softmax', name='out_1')(out)

    return Model(inputs=[original_x, calibrate_x], outputs=logit)


def SepSteNet_with_DPES(input_shape):
    """
    our implementation of SepSteNet with DPES in a crude way.
    When applying it to other backbone, please see DPES().
    """

    original_x = Input(shape=input_shape)
    calibrate_x = Input(shape=input_shape)

    x = Lambda(lambda tensor: tf.reshape(tensor, (-1, tensor.shape[1] * 4, 10)))(original_x)
    y = Lambda(lambda tensor: tf.reshape(tensor, (-1, tensor.shape[1] * 4, 10)))(calibrate_x)

    x = Embedding(40, 16)(x)
    y = Embedding(40, 16)(y)

    x = residual_block(x, filters=128, re_sample=True, with_weight=True)
    x = residual_block(x, filters=128, with_weight=True)
    x1 = residual_block(x, filters=128, with_weight=True)

    x = residual_block(x1, filters=256, re_sample=True, with_weight=True)
    x = residual_block(x, filters=256, with_weight=True)
    x2 = residual_block(x, filters=256, with_weight=True)

    x = Lambda(lambda tensor: tf.reshape(tensor, (-1, tensor.shape[1], tensor.shape[2] * tensor.shape[3])))(x2)

    x = Conv1D(512, 1, kernel_initializer=initializers.he_normal)(x)
    x = LayerNormalization()(x)
    x = Activation(tf.nn.gelu)(x)

    x = TrainablePositionEmbedding(x.shape[1], x.shape[2])(x)

    x = LayerNormalization()(x)
    x3 = MultiHeadAttention(12, 64)(x, x, x)

    x = LayerNormalization()(x3)
    x = GatedFFN(512)(x)
    x4 = Dropout(0.2)(x)

    #######################################################

    y = residual_block(y, filters=128, re_sample=True, with_weight=True)
    y = residual_block(y, filters=128, with_weight=True)
    y1 = residual_block(y, filters=128, with_weight=True)

    y = residual_block(y1, filters=256, re_sample=True, with_weight=True)
    y = residual_block(y, filters=256, with_weight=True)
    y2 = residual_block(y, filters=256, with_weight=True)

    y = Lambda(lambda tensor: tf.reshape(tensor, (-1, tensor.shape[1], tensor.shape[2] * tensor.shape[3])))(y2)

    y = Conv1D(512, 1, kernel_initializer=initializers.he_normal)(y)
    y = LayerNormalization()(y)
    y = Activation(tf.nn.gelu)(y)

    y = TrainablePositionEmbedding(y.shape[1], y.shape[2])(y)

    y = LayerNormalization()(y)
    y3 = MultiHeadAttention(12, 64)(y, y, y)

    y = LayerNormalization()(y3)
    y = GatedFFN(512)(y)
    y4 = Dropout(0.2)(y)

    #######################################################

    x1 = GlobalAveragePooling2D()(x1)
    x2 = GlobalAveragePooling2D()(x2)
    x3 = GlobalAveragePooling1D()(x3)
    x4 = GlobalAveragePooling1D()(x4)
    y1 = GlobalAveragePooling2D()(y1)
    y2 = GlobalAveragePooling2D()(y2)
    y3 = GlobalAveragePooling1D()(y3)
    y4 = GlobalAveragePooling1D()(y4)

    dis1 = Concatenate()([x1, y4])
    dis2 = Concatenate()([x2, y3])
    dis3 = Concatenate()([x3, y2])
    dis4 = Concatenate()([x4, y1])

    dis1 = Dense(128, kernel_initializer='he_normal')(dis1)
    dis2 = Dense(128, kernel_initializer='he_normal')(dis2)
    dis3 = Dense(128, kernel_initializer='he_normal')(dis3)
    dis4 = Dense(128, kernel_initializer='he_normal')(dis4)

    out = Concatenate()([dis1, dis2, dis3, dis4])

    logit = Dense(2, activation='softmax', name='out_1')(out)
    return Model(inputs=[original_x, calibrate_x], outputs=logit)
