## Pytorch implementation of [Neural Sequence Decoder](https://github.com/fwillett/speechBCI/tree/main/NeuralDecoder)

## Requirements
- python >= 3.9

## Installation

pip install -e .

## How to run

1. Convert the speech BCI dataset using [formatCompetitionData.ipynb](./notebooks/formatCompetitionData.ipynb)
2. Train model: `python ./scripts/train_model.py`




#####################################################
Samantha's Notes: 

decoder_dataset is the processed version of the matlab data (meaning you don't have to download the matlab data from the dryad page nor do you need to run formatCompetitionData.ipynb)


