# -*-coding:utf-8-*-
# @author: Yiqin Qiu
# @email: barryxxz@stu.hqu.edu.cn

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn.metrics import recall_score, accuracy_score
from tensorflow_addons.optimizers import AdamW
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras import backend
from model import SepSteNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class MyReduceLROnPlateau(Callback):
    def __init__(self,
                 monitor='val_loss',
                 factor=0.1,
                 patience=10,
                 verbose=0,
                 mode='auto',
                 min_delta=1e-4,
                 cooldown=0,
                 min_lr=0,
                 **kwargs):
        super(MyReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            logging.warning('`epsilon` argument is deprecated and '
                            'will be removed, use `min_delta` instead.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        if self.mode not in ['auto', 'min', 'max']:
            logging.warning('Learning rate reduction mode %s is unknown, '
                            'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Learning rate reduction is conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = backend.get_value(self.model.optimizer.lr)
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        backend.set_value(self.model.optimizer.lr, new_lr)
                        backend.set_value(self.model.optimizer.weight_decay, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                  'rate to %s.' % (epoch + 1, new_lr))
                            print('\nEpoch %05d: ReduceLROnPlateau reducing weight '
                                  'decay to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


def train(x_train_data, y_train_data, x_val_data, y_val_data, weight_path, model):
    y_train_data = to_categorical(y_train_data, num_classes=2)
    y_val_data = to_categorical(y_val_data, num_classes=2)

    checkpoint = ModelCheckpoint(weight_path, monitor='val_accuracy', verbose=0, save_best_only=True,
                                 mode='max', save_weights_only=True)
    reduce_lr = MyReduceLROnPlateau(monitor='val_loss', factor=0.2, min_lr=8e-7, patience=5, verbose=1)

    callbacks_list = [checkpoint, reduce_lr]

    optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-4)
    loss = CategoricalCrossentropy()
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    hist = model.fit(x_train_data, y_train_data, batch_size=args.batch_size, epochs=args.epoch,
                     validation_data=(x_val_data, y_val_data), callbacks=callbacks_list, verbose=2, shuffle=True)


def test(x_test_data, y_test_data, weight_path):
    in_shape = x_test_data.shape[1:]
    net = SepSteNet(in_shape)
    net.load_weights(weight_path)
    y_predict = net.predict(x_test_data)

    y_predict = np.argmax(y_predict, axis=1)
    print('[INFO] accuracy on test set: %0.2f%%' % (accuracy_score(y_test_data, y_predict) * 100))

    tpr = recall_score(y_test_data, y_predict)
    tnr = recall_score(y_test_data, y_predict, pos_label=0)
    fpr = 1 - tnr
    fnr = 1 - tpr
    print('[INFO] FPR on test set: %0.2f' % (fpr * 100))
    print('[INFO] FNR on test set: %0.2f' % (fnr * 100))


def main(arg):
    print('[INFO] Loading dataset')

    tmp = np.load('./dataset/data_{}_{}s_{}_train.npy'.format(method, arg.length, arg.em_rate), allow_pickle=True)
    x_train, y_train = np.asarray([item[0] for item in tmp]), np.asarray([item[2] for item in tmp])
    x_train = x_train.astype(np.float)

    tmp = np.load('./dataset/data_{}_{}s_{}_val.npy'.format(method, arg.length, arg.em_rate), allow_pickle=True)
    x_val, y_val = np.asarray([item[0] for item in tmp]), np.asarray([item[2] for item in tmp])
    x_val = x_val.astype(np.float)

    tmp = np.load('./dataset/data_{}_{}s_{}_test.npy'.format(method, arg.length, arg.em_rate), allow_pickle=True)
    x_test, y_test = np.asarray([item[0] for item in tmp]), np.asarray([item[2] for item in tmp])
    x_test = x_test.astype(np.float)

    print('[INFO] Loading finished')

    print('[INFO] The property of the dataset')
    print("train num: %d" % len(x_train))
    print("val num: %d" % len(x_val))
    print("test num: %d" % len(x_test))

    print('[INFO] Loading model')
    in_shape = x_train.shape[1:]
    model = SepSteNet(in_shape)
    model.summary()

    if arg.train:
        print('[INFO] Training started')
        train(x_train, y_train, x_val, y_val, arg.model_path, model)
        print('[INFO] Finished')
    if arg.test:
        print('[INFO] Testing the detection performance')
        test(x_test, y_test, arg.model_path)
        print('[INFO] Finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SepSteNet')

    parser.add_argument('--method', type=str, default='Geiser',
                        help='Target steganography method, option: Geiser, Miao/enta1, Miao/enta2, Miao/enta4.')
    parser.add_argument('--length', type=str, default='0.9',
                        help='Sample length, option: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0.')
    parser.add_argument('--em_rate', type=str, default='RAND',
                        help='Embeeding rate, option: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, RAND.')
    parser.add_argument('--model_path', type=str, default='',
                        help='Path of the model weight, you do not have to set it when using our trained weights.')

    parser.add_argument('--seed', type=int, default=777,
                        help='Value of random seed.')
    parser.add_argument('--epoch', type=int, default=40,
                        help='Epoch of training the model.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size during training and testing.')

    parser.add_argument('--train', type=bool, default=False,
                        help='Whether to train the model.')
    parser.add_argument('--test', type=bool, default=True,
                        help='Whether to test the model.')

    args = parser.parse_args()

    set_seed(args.seed)

    if args.method[:4] == 'Miao':
        method = args.method[:4] + '_' + args.method[-5:]
    else:
        method = args.method
    args.model_path = './model_weights/SepSteNet/SepSteNet_{}_{}s_{}.h5'.format(method, args.length, args.em_rate)
    print('[INFO] The path of model weight:')
    print(args.model_path)

    main(args)
