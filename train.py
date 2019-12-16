import tensorflow as tf
import tensorlayer as tl
from dataset import make_train_dataset,make_val_dataset
from model import get_model
import time
import numpy as np
# lr=0.25e-4, EPOCHS=100, save_checkpoint=500, batch_size=64

n_epoch=100
train_ds=make_train_dataset()
test_ds=make_val_dataset()
net=get_model()
#net.load_weights('./net_weights.h5')#一开始训练不要有这个东西，之后才有
train_weights=net.trainable_weights
optimizer = tf.optimizers.SGD(train_weights, lr=0.25e-4)
#n_step_epoch = 训练一次之后会知道unknown step epoch是多少

for epoch in range(n_epoch):
    start_time = time.time()

    train_loss, train_acc, n_iter = 0, 0, 0
    for img, aud, out in train_ds:
        net.train()

        step_time=time.time()
        with tf.GradientTape() as tape:
            # compute outputs
            _logits = net([img, aud])#慌。。
            # compute loss and update model
            _loss = tl.cost.cross_entropy(_logits, out, name='train_loss')
            #loss_L2 = 0
            # for p in tl.layers.get_variables_with_name('relu/W', True, True):
            #      _loss_L2 += tl.cost.lo_regularizer(1.0)(p)
            #_loss = _loss_ce + _loss_L2

        grad = tape.gradient(_loss, train_weights)
        optimizer.apply_gradients(zip(grad, train_weights))

        train_loss += _loss
        _acc=np.mean(np.equal(np.argmax(_logits, 1), out))
        train_acc += _acc
        n_iter += 1

        del tape

        if n_iter%10==0:
            net.eval()
            val_loss, val_acc, vn_iter = 0, 0, 0
            for img,aud, out in test_ds:
                _logits = net([img,aud])  # is_train=False, disable dropout
                val_loss += tl.cost.cross_entropy(_logits, out, name='eval_loss')
                val_acc += np.mean(np.equal(np.argmax(_logits, 1), out))
                vn_iter += 1
            print("   val loss: {}".format(val_loss / vn_iter))
            print("   val acc:  {}".format(val_acc / vn_iter))
            net.train()
            print("Epoch: [{}/{}] [{}/unknown_step_epoch] took: {:3f}, train_loss: {:5f}, train_acc: {:5f}".format(epoch, n_epoch, n_iter,time.time() - step_time,_loss, _acc))

    print("Epoch: [{}/{}] took: {:3f}, train_loss: {:5f}, train_acc: {:5f}".format(epoch, n_epoch,time.time() - start_time,train_loss / n_iter,train_acc / n_iter))
    net.save_weights('./net_weights.h5')  # every epoch

    # use training and evaluation sets to evaluate the model every print_freq epoch


'''
    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        net.eval()
        val_loss, val_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in test_ds:
            _logits = net(X_batch)  # is_train=False, disable dropout
            val_loss += tl.cost.cross_entropy(_logits, y_batch, name='eval_loss')
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   val loss: {}".format(val_loss / n_iter))
        print("   val acc:  {}".format(val_acc / n_iter))
        net.train()
'''
