from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import tensorlayer as tl
import time
import os
import numpy as np
from dataset import make_val_dataset
from model import get_model

net = get_model()
val_ds = make_val_dataset()

model = 'weights/net_weights_e8.h5' # modify this to load trained model
net.load_weights(model)
net.eval()

model = os.path.basename(model)
step_time = time.time()
val_loss, val_acc, vn_iter = 0, 0, 0

for img, aud, out in val_ds:
    _logits = net([img, aud])

    val_loss += tl.cost.cross_entropy(_logits, out, name='eval_loss')
    val_acc += np.mean(np.equal(np.argmax(_logits, 1), out))
    vn_iter += 1

    print("[model {}] took: {:3f}, vn_iter: {}, val_loss: {:5f}, val_acc: {:5f}"
       .format(model, time.time() - step_time, vn_iter, val_loss / vn_iter, val_acc / vn_iter))
