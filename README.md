# Objects-that-Sound
An implementation of the paper "[Objects that Sound](https://arxiv.org/abs/1712.06651)".

Refered to the project "[
Learning intra-modal and cross-modal embeddings](https://github.com/rohitrango/objects-that-sound)".

## Objective

> Our objective is to create a group of networks that can embed audio and visual inputs into a common space that is suitable for cross-modal retrieval. We achieve this by training from unlabelled video using only audio-visual correspondence (AVC) as the objective function. This is a form of crossmodal self-supervision from video.

The paper shows that audio and visual embeddings that enable both within-mode (e.g. audio-to-audio) and between-mode retrieval can be learnt. Yet we evaluate our network using only the latter, due to limited time.

## Dataset

[Audioset dataset](https://research.google.com/audioset/download.html)

## Architecture
![architecture](https://github.com/lrh2000/Objects-that-Sound/blob/master/img/arc.png)

## Problems encountered and solved
- **Data preprocessing**: we originally get images from mp4 by downloading videos from youtube, choosing 10s-long fractions to use and then select images from these segments. The whole process was quite slow: it seemed that the bottleneck was data processing rather than  network training. By using 'resize' function and 'spectrogram' function, we preprocessed data into the input format of our network, so that multiprocessing became possible and training was accelerated.
- **Preventing shortcuts**: having been trained with one epoch, we found that the network was able to find a shortcut to exploit the not-so-randomized data to increase its correctness in training. It misused the latest data to rapidly adjust itself so as to pretend to be learned because we didn't shuffle our data in a range that was large enough. We therefore increased shuffle parameter so that data pairs would be well scattered.

## Results
![Results](https://github.com/lrh2000/Objects-that-Sound/blob/master/img/results.png)

## To-Do
- **Visualizing the results**: display the retrieved results that matches each query best.
- **Localizing objects that sound**: implement the latter part of the paper.
- **Extension to other evaluations**: include other three types of queries (image-2-audio, image-2-image and audio-2-audio)
