# Parkinson's Voice Analysis

Research project comparing handcrafted acoustic features with a Log-Mel CNN for voice-based Parkinson's disease classification.

## Repository contents

- `notebooks/`: complete executed Google Colab/Jupyter research pipeline.
- `papers/ieee/`: IEEE two-column paper PDF and LaTeX source.
- `papers/springer/`: Springer one-column paper PDF and LaTeX source (alternate-format copy; verify it matches the final IEEE revision before treating it as a final submission).
- `presentation/`: research presentation PDF, Beamer source, and image/code assets.
- `results/`: confusion matrices and permutation-importance figure.
- `dataset/`: directory layout for authorized local dataset placement; raw recordings are intentionally not included.

## Dataset note — important

The experiment used the Italian Parkinson's Voice and Speech (IPVS) collection: 831 WAV recordings from 65 participants (15 young healthy controls, 22 elderly healthy controls, 28 participants with Parkinson's disease). The original audio files were not present in the files supplied for packaging, so this repository does **not** contain the raw dataset. The `dataset/train`, `dataset/validation`, and `dataset/test` folders are placeholders, not populated datasets. Obtain the IPVS data from its authorized provider and check its license/terms before adding or redistributing any recordings. Do not upload identifiable or restricted participant audio to a public repository without permission.

The saved experiment split is participant-disjoint:

| Partition | Participants | Recordings | HC participants | PD participants |
|---|---:|---:|---:|---:|
| Classical development / CNN development | 48 | 605 | 27 | 21 |
| CNN training subset | 38 | 474 | — | — |
| CNN validation subset | 10 | 131 | — | — |
| Held-out test | 17 | 226 | 10 | 7 |

The 38/10 CNN split is within the 48 development participants. Do not create recording-level splits; keep all recordings from a participant in one partition.

## Reproducing the experiment

1. Download the IPVS dataset from its authorized source and place it under `dataset/` using the cohort/participant folder structure expected by the notebook. The notebook's scanner handles the extra range-folder level in the PD cohort.
2. Open `notebooks/parkinsons_voice_classification_pipeline.ipynb` in Google Colab or Jupyter.
3. Install the dependencies listed in `requirements.txt` (TensorFlow is additionally required for the CNN; install a version compatible with your runtime).
4. Update the dataset root and output/cache paths in the notebook for your environment.
5. Run cells in order. The notebook audits participants, extracts features, creates participant-level splits, trains/evaluates classical models and the CNN, and saves metrics/artifacts.

## Key saved results

- 831 WAV recordings; 65 participants.
- RBF-SVM: recording-level accuracy 0.9558; participant-level accuracy 0.9412.
- Log-Mel CNN: recording-level accuracy 0.9690; participant-level accuracy 0.9412.
- Participant-level held-out test cohort: 17 participants.

These are results from one saved split on one dataset. They are not external clinical validation and must not be presented as clinical diagnostic performance.

## Citation / attribution

See the reference lists in the included papers for the cited literature and dataset-related publication. Follow the IPVS provider's citation and data-use requirements when using the recordings.
