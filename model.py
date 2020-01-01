import tensorflow as tf
import tensorlayer as tl
from tensorlayer.models import Model
from tensorlayer.layers import Layer, Input, Dense, Conv2d, MaxPool2d, BatchNorm2d, Flatten

# Vision ConvNet
def get_vision_net():
    ni = Input([None, 224, 224, 3])

    nn = Conv2d(n_filter=64, filter_size=(3, 3), strides=(2, 2), padding='SAME')(ni)
    nn = BatchNorm2d(act=tf.nn.relu)(nn)
    nn = Conv2d(n_filter=64, filter_size=(3, 3), strides=(1, 1), padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu)(nn)
    nn = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME')(nn)
    assert(nn.shape[1:] == (56, 56, 64))

    nn = Conv2d(n_filter=128, filter_size=(3, 3), strides=(1, 1), padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu)(nn)
    nn = Conv2d(n_filter=128, filter_size=(3, 3), strides=(1, 1), padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu)(nn)
    nn = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME')(nn)
    assert(nn.shape[1:] == (28, 28, 128))

    nn = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu)(nn)
    nn = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu)(nn)
    nn = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME')(nn)
    assert(nn.shape[1:] == (14, 14, 256))

    nn = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu)(nn)
    nn = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu)(nn)
    assert(nn.shape[1:] == (14, 14, 512))

    return Model(inputs=ni, outputs=nn, name='Vision ConvNet')

# Audio ConvNet
def get_audio_net():
    ni = Input([None, 257, 200, 1])

    nn = Conv2d(n_filter=64, filter_size=(3, 3), strides=(2, 2), padding=[[0, 0], [0, 0], [0, 2], [0, 0]])(ni)
    nn = BatchNorm2d(act=tf.nn.relu)(nn)
    nn = Conv2d(n_filter=64, filter_size=(3, 3), strides=(1, 1), padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu)(nn)
    nn = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME')(nn)
    assert(nn.shape[1:] == (64, 50, 64))

    nn = Conv2d(n_filter=128, filter_size=(3, 3), strides=(1, 1), padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu)(nn)
    nn = Conv2d(n_filter=128, filter_size=(3, 3), strides=(1, 1), padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu)(nn)
    nn = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME')(nn)
    assert(nn.shape[1:] == (32, 25, 128))

    nn = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu)(nn)
    nn = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu)(nn)
    nn = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='VALID')(nn)
    assert(nn.shape[1:] == (16, 12, 256))

    nn = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu)(nn)
    nn = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), padding='SAME')(nn)
    nn = BatchNorm2d(act=tf.nn.relu)(nn)
    assert(nn.shape[1:] == (16, 12, 512))

    return Model(inputs=ni, outputs=nn, name='Audio ConvNet')

class NormDistance(Layer):
    # Normalize the image embedding vector and the audio embedding vector,
    # and calculate the Euclidean distance between them.
    #
    # Represent "L2 normalization" and "Euclidean distance" layers which
    # are written in the paper.

    def __init__(self, name=None):
        super(NormDistance, self).__init__(name)

    def build(self, input_shapes):
        pass

    def forward(self, inputs):
        tensor1 = inputs[0]
        tensor2 = inputs[1]

        tensor1 = tf.nn.l2_normalize(tensor1, axis=1)
        tensor2 = tf.nn.l2_normalize(tensor2, axis=1)

        output = tf.norm(tensor1 - tensor2, axis=1, keepdims=True)
        return output

    def __repr__(self):
        s = '{classname}('
        if self.name is not None:
            s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

# AVE Net
def get_model():
    vision_net = get_vision_net()
    audio_net = get_audio_net()

    ni1 = vision_net.inputs
    nn1 = vision_net.outputs
    ni2 = audio_net.inputs
    nn2 = audio_net.outputs

    nn1 = MaxPool2d(filter_size=(14, 14), padding='VALID')(nn1)
    nn1 = Flatten()(nn1)
    assert(nn1.shape[1:] == (512, ))
    nn1 = Dense(n_units=128, act=tf.nn.relu)(nn1)
    nn1 = Dense(n_units=128, act=None)(nn1)

    nn2 = MaxPool2d(filter_size=(16, 12), padding='VALID')(nn2)
    nn2 = Flatten()(nn2)
    assert(nn2.shape[1:] == (512, ))
    nn2 = Dense(n_units=128, act=tf.nn.relu)(nn2)
    nn2 = Dense(n_units=128, act=None)(nn2)

    nn = NormDistance()([nn1, nn2])
    assert(nn.shape[1:] == (1, ))
    nn = Dense(n_units=2, act=tf.nn.softmax)(nn)

    return Model(inputs=[ni1, ni2], outputs=nn)

if __name__ == "__main__":
    M = get_model()
    print(M)

    # Try to run some sample epochs...
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    for epoch in range(5):
        images = tf.random.uniform((10, 224, 224, 3), maxval=256.)
        audios = tf.random.uniform((10, 257, 200, 1), maxval=256.)

        #labels = tf.random.uniform((10, ), minval=0, maxval=2, dtype=tf.int32)
        #labels = tf.zeros((10, ), dtype=tf.int32)
        labels = tf.ones((10, ), dtype=tf.int32)
        print("labels={}".format(labels))

        M.train()
        weights = M.trainable_weights
        with tf.GradientTape() as tape:
            logits = M([images, audios])
            loss = tl.cost.cross_entropy(logits, labels)

        print("logits={}\n".format(logits))
        grad = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grad, weights))

    print("Good! It seems that everything works well!")
