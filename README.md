# Objects-that-Sound
An implementation of the paper "[Objects that Sound](https://arxiv.org/abs/1712.06651)".

Refered to the project "[
Learning intra-modal and cross-modal embeddings](https://github.com/rohitrango/objects-that-sound)".

## Objective

> Our objective is networks that can embed audio and visual inputs into a common space that is suitable for cross-modal retrieva. We achieve this objective by training from unlabelled video using only audio-visual correspondence (AVC) as the objective function. This is a form of crossmodal self-supervision from video.

The paper shows that audio and visual embeddings can be learnt that enable both within-mode (e.g. audio-to-audio) and between-mode retrieval. Yet we evaluate our network using only the latter, due to the limited time.

## Dataset

[Audioset dataset](https://research.google.com/audioset/download.html)

## Architecture
![architecture](https://github.com/lrh2000/Objects-that-Sound/blob/master/img/arc.png)

## Problems encountered and solved
- **Data preprocessing**: we originally read images from mp4, which was slow to decode. The bottleneck appeared to be data processing rather than  network training. By using 'resize' function and 'spectrogram' function, we preprocessed data into the input format of our network, so that multiprocessing became possible and training was accelerated.
- **Preventing shortcuts**: having been trained with one epoch, we found the network ﬁnding subtle data shortcuts to exploit. It misused the latest data to rapidly adjust itself so as to pretend to be learned. We therefore increased shuffle parameter so that data pairs would be well scattered.

## Results
![Results](https://github.com/lrh2000/Objects-that-Sound/blob/master/img/results.png)

## To-Do
- **Visualizing the results**: dispaly the retrieved results that matches each query best.
- **Localizing objects that sound**: implement the latter part of the paper.
- **Extension to other evaluations**: include other three types of queries (image-2-audio, image-2-image and audio-2-audio)
