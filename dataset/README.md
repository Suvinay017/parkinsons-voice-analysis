# Dataset — Italian Parkinson's Voice and Speech (IPVS)

This project uses the **Italian Parkinson's Voice and Speech (IPVS)** dataset for voice-based Parkinson's disease classification.

## Original dataset

- **Dataset:** Italian Parkinson's Voice and Speech (IPVS)
- **Original source:** [IEEE DataPort — Italian Parkinson's Voice and Speech](https://ieee-dataport.org/open-access/italian-parkinsons-voice-and-speech)
- **DOI:** [10.21227/aw6b-tg17](https://doi.org/10.21227/aw6b-tg17)
- **Dataset citation:** Dimauro, G. and Girardi, F., "Italian Parkinson's Voice and Speech," IEEE DataPort, 2019.

Please obtain the dataset from its original source and follow the access conditions and license displayed by the provider. This repository does not redistribute the original audio recordings.

## Dataset organization used in this project

The experiment uses the following participant-aware partitions:

| Partition | Participants | Recordings | Class distribution |
|---|---:|---:|---|
| Development set | 48 | 605 | 27 HC, 21 PD |
| CNN training subset | 38 | 474 | Participant-level subset of development data |
| CNN validation subset | 10 | 131 | Participant-level subset of development data |
| Held-out test set | 17 | 226 | 10 HC, 7 PD |

**Important:** The CNN training and validation subsets are subdivisions of the development set; they are not additional participants. The held-out test participants are separate from the development participants.

- **HC:** Healthy Control
- **PD:** Parkinson's Disease

## Reproducing the experiment

1. Obtain the IPVS dataset from the original source linked above.
2. Arrange the audio files according to the structure and file paths expected by the notebook.
3. Run `notebooks/parkinsons_voice_classification_pipeline.ipynb` and follow its participant-aware split and preprocessing steps.

The notebook is the source of truth for the exact preprocessing, feature extraction, and model-evaluation pipeline. Do not create a random recording-level split: recordings from the same participant must remain in the same partition.

## Citation

If you use this dataset, cite the original dataset:

> Dimauro, G. and Girardi, F., "Italian Parkinson's Voice and Speech," IEEE DataPort, 2019. https://doi.org/10.21227/aw6b-tg17

Also follow any citation requirements stated on the original dataset page.
