
# Aquire and Build Datasets
The datasets used in this project were mainly the ones provided with the DCASE 2021 Challenge, specifically Task 4: Sound Event Detection and Separation in Domestic Environments[^1].

[^1]: https://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments.

## Datasets
DESED is one of the collections of datasets from DCASE. It includes 4 public datasets with strong annotations (see [DCASE 2019](https://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments#reference-labels) for a definition). 

From DESED we used the following:

- DESED Synthetic Training \
    Synthetically created soundscapes. \
    Number of audio clips: 10000. \
    Speech-labels in annotations: 18740 / 40673 = 46.0%. \
    Duration of Speech (hours): 541.8 / 1666.7 = 32.5%. \
    Clip resolution: 44.1 kHz. \

- DESED Synthetic Validation \
    Synthetically created soundscapes. \
    Number of audio clips: 1500. \
    Speech-labels in annotations: 2577 / 6051 = 42.6%. \
    Duration of Speech (hours): 128.6 / 250.0 = 51.5%. \
    Clip resolution: 44.1 kHz. \

- DESED Real Validation (NOTE: secondary dataset used for validation) \
    Based on real data from recorded soundscapes. \
    Number of audio clips: 1167. *\** \
    Speech-labels in annotations: 1749 / 4240 = 41.3%. \
    Duration of Speech (hours): 43.6 / 194.5 = 22.4%. \
    Clip resolution: 44.1 kHz. \


- DESED Public Evaluation \
    Based on real data from recorded soundscapes. \
    Number of audio clips: 692. *\** \
    Speech-labels in annotations: 913 / 2765 = 33.0%. \
    Duration of Speech (hours): 20.8 / 115.3 = 18.0%. \
    Clip resolution: 44.1 kHz. \

*\** *Numbers might differ from original dataset because of missing downloads.*

## Download
The instructions for how to download these datasets can be found at [Domestic Environment Sound Event Detection Dataset](https://project.inria.fr/desed/).
