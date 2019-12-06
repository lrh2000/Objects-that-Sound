import tensorflow as tf
import scipy.signal
import scipy.io.wavfile
import random
import cv2
import os
import logging

class DataGenerator:
    def __init__(self, csv_file, video_dir, audio_dir):
        with open(csv_file) as f:
            self.segments = f.readlines()
            self.segments = [s[:-1].split(',') for s in self.segments]
            self.segments = [s[:1] + [s[3:]] for s in self.segments]
            self.segments = dict(self.segments)

        self.video_dir = video_dir
        self.audio_dir = audio_dir

        self.local_video = sorted(os.listdir(video_dir))
        self.local_audio = sorted(os.listdir(audio_dir))
        assert(len(self.local_audio) == len(self.local_video))

        self.frames_per_second = 30
        self.frame0_time = 0.5

        self.files_count = len(self.local_video)
        self.frames_per_file = 30 * 9
        self.overall_count = self.frames_per_file * self.files_count

        self.current = 0
        self.last_video_file = ""

        self.saved_audios = dict()

    def __iter__(self):
        return self

    def __call__(self):
        return self

    def __next__(self):
        if self.current >= self.overall_count:
            raise StopIteration
        idx = self.current // self.frames_per_file
        frame = self.current % self.frames_per_file
        self.current += 1

        video_file = self.local_video[idx]
        video_time = self.frame0_time + frame / self.frames_per_second
        label = random.randint(0, 1)
        if label:
            video_id = video_file.split("video_")[1].split(".mp4")[0]
            video_tags = set(self.segments[video_id])

            while True:
                audio_file = random.choice(self.local_audio)
                audio_id = audio_file.split("audio_")[1].split(".wav")[0]
                audio_tags = set(self.segments[audio_id])

                if not(audio_tags & video_tags):
                    break

            logging.debug('audio_tags %s video_tags %s', audio_tags, video_tags)
            frame = random.randint(0, self.frames_per_file - 1)
        else:
            audio_file = self.local_audio[idx]
        audio_time = self.frame0_time + frame / self.frames_per_second

        logging.debug('audio_file %s audio_time %s', audio_file, audio_time)
        logging.debug('video_file %s video_time %s', video_file, video_time)
        logging.debug('label %s', label)

        if self.last_video_file != video_file:
            self.last_video_file = video_file
            self.video = cv2.VideoCapture(os.path.join(self.video_dir, video_file))
        self.video.set(cv2.CAP_PROP_POS_MSEC, video_time * 1000)
        success, image = self.video.read()
        if not success:
            logging.warning('Failed to read video file. (file %s time %s)',
                                video_file, video_time)
            return self.__next__()
        image = cv2.resize(image, (224, 224))

        if self.saved_audios.get(audio_file, None) is None:
            audio = scipy.io.wavfile.read(os.path.join(self.audio_dir, audio_file))
            rate, samples = audio
            if rate != 48000:
                logging.warning('Wrong wav rate %s. (file %s time %s)',
                                    rate, audio_file, audio_time)
                return self.__next__()
            self.saved_audios[audio_file] = samples
        else:
            rate = 48000
            samples = self.saved_audios[audio_file]
        samples = samples[int(rate * (audio_time - 0.5)):int(rate * (audio_time + 0.5))]

        spectrogram = scipy.signal.spectrogram(samples, rate, nperseg=512, noverlap=274)[2]
        if spectrogram.shape != (257, 200):
            logging.warning('Wrong spectrogram.shape %s. (file %s time %s)',
                                spectrogram.shape, audio_file, audio_time)
            return self.__next__()
        spectrogram = scipy.log(spectrogram + 1e-7)
        spectrogram = spectrogram.reshape(tuple(list(spectrogram.shape) + [1]))

        assert(image.shape == (224, 224, 3))
        assert(spectrogram.shape == (257, 200, 1))

        return image, spectrogram, label

def make_train_dataset():
    train_ds = tf.data.Dataset.from_generator(
            DataGenerator('csv/unbalanced_train_segments_filtered.csv', 'Video/', 'Audio/'),
            output_types=(tf.float32, tf.float32, tf.int32)
        )
    train_ds = train_ds.shuffle(2700)
    train_ds = train_ds.batch(64)
    train_ds = train_ds.prefetch(10)
    return train_ds

def make_val_dataset():
    val_ds = tf.data.Dataset.from_generator(
            DataGenerator('csv/balanced_train_segments_filtered.csv', 'VideoVal/', 'AudioVal/'),
            output_types=(tf.float32, tf.float32, tf.int32)
        )
    val_ds = val_ds.shuffle(2700)
    val_ds = val_ds.batch(64)
    val_ds = val_ds.prefetch(10)
    return val_ds

if __name__ == '__main__':
    #logging.basicConfig(level=logging.DEBUG)

    train_ds = make_train_dataset()
    for images, audios, labels in train_ds:
        print(images.shape)
        print(audios.shape)
        print(labels)
        break
