## Pytorch implementation of [Neural Sequence Decoder](https://github.com/fwillett/speechBCI/tree/main/NeuralDecoder)

## Requirements
- python >= 3.9

## Installation

pip install -e .

## How to run

1. Convert the speech BCI dataset using [formatCompetitionData.ipynb](./notebooks/formatCompetitionData.ipynb)
2. Train model: `python ./scripts/train_model.py`



## Samantha's Notes
I used Shreeram's parameters from bruinlearn to train his (lightweight) version of the baseline. If you're curious what the original baseline parameters were, they're still on the original repo from the assignment (https://github.com/cffan/neural_seq_decoder)



