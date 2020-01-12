from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import tensorlayer as tl
import time
import numpy as np
from rawdata.dataset import make_train_dataset_from_raw
from model import get_model

n_epoch = 20

net = get_model()
net.load_weights('weights/net_weights_e15.h5') # modify this to load trained model
train_weights = net.trainable_weights
optimizer = tf.optimizers.Adam(learning_rate=0.00001)

train_ds = make_train_dataset_from_raw()

for epoch in range(16, n_epoch):
    start_time = time.time()

    train_loss, train_acc, n_iter = 0, 0, 0
    for img, aud, out in train_ds:
        net.train()

        # compute outputs, loss and update model
        with tf.GradientTape() as tape:
            _logits = net([img, aud])
            _loss = tl.cost.cross_entropy(_logits, out, name='train_loss')
        grad = tape.gradient(_loss, train_weights)
        optimizer.apply_gradients(zip(grad, train_weights))
        del tape

        _acc = np.mean(np.equal(np.argmax(_logits, 1), out))
        train_loss += _loss
        train_acc += _acc
        n_iter += 1

        if n_iter % 421 == 0:
            net.save_weights('./net_weights_e{}i{}.h5'.format(epoch, n_iter))

        if n_iter % 42 == 0:
            print("[epoch{} n_iter{}] took: {:3f}, train_loss: {:5f}, train_acc: {:5f}"
                   .format(epoch, n_iter, time.time() - start_time, train_loss / n_iter, train_acc / n_iter))

    net.save_weights('./net_weights_e{}.h5'.format(epoch))
    print("[Epoch{} done!] n_iter: {}, took: {:3f}, train_loss: {:5f}, train_acc: {:5f}"
           .format(epoch, n_iter, time.time() - start_time, train_loss / n_iter, train_acc / n_iter))
