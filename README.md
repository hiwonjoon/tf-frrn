# FRRN

## Hello World?

Foo bar is always our friend

## Training from scratch

### Dataset preprocessing

1. Donwload datasets(`gtFine_trainvaltest.zip`, `leftImg8bit_trainvaltest.zip`) and unzip under the directory dataset
  - Make sure that path look like `./datasets/cityspaces/gtFine` and `./datasets/cityspaces/leftImg8bit`.
2. Run script to convert TrainId format.
  - `python cityscapesScripts/cityscapesscripts/preparation/createTrainIdLabelImgs.py`

### Run main training script

Run,
`
python main.py
`

## Acknowledgement


