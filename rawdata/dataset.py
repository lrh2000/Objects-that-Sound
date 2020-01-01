import tensorflow as tf
import numpy as np

class RawDataGenerator:
    def __init__(self):
        self.part = 0
        self.sub = 0
        self.loaded_data = None
        self.curr_pos = 0

    def __iter__(self):
        return self

    def __call__(self):
        return self

    def __next__(self):
       while True:
           if not(self.loaded_data is None or self.curr_pos >= len(self.loaded_data)):
                data = self.loaded_data[self.curr_pos]
                self.curr_pos += 1
                return tuple(data)

           self.loaded_data = None
           self.curr_pos = 0
           try:
               self.loaded_data = np.load('Data/data_part{}sub{}.npy'.format(self.part, self.sub),
                                            allow_pickle=True)
               self.sub += 1
           except FileNotFoundError:
               if self.sub == 0:
                   raise StopIteration
               self.part += 1
               self.sub = 0

def image_normalize(img, aud, tag):
    img = img / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return (img, aud, tag)

def make_train_dataset_from_raw():
    train_ds = tf.data.Dataset.from_generator(
            RawDataGenerator(),
            output_types=(tf.float32, tf.float32, tf.int32)
        )
    train_ds = train_ds.map(image_normalize, num_parallel_calls=4)
    train_ds = train_ds.shuffle(188888)
    train_ds = train_ds.batch(80)
    return train_ds

def make_val_dataset_from_raw():
    raise NotImplementedError
    # It's not used by us.

if __name__ == '__main__':

    train_ds = make_train_dataset_from_raw()
    for images, audios, labels in train_ds:
        print(images.shape)
        print(audios.shape)
        print(labels)
        break
