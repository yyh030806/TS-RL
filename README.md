# React-OT: Optimal Transport for Generating Transition State in Chemical Reactions
==============================

## Environment set-up
```
conda env create -f env.yaml
conda activate reactot && pip install -e .
```

## Download data
The processed data is uploaded on zenodo, [download link](https://zenodo.org/records/13131875). You need to put both pickle files under the data directory.

```
mkdir reactot/data
mkdir reactot/data/transition1x
mv PATH_TO_PKL_FILES reactot/data/transition1x/
```

## Evaluation using a pre-trained model
The pre-trained model can be downloaded through the [download link](https://zenodo.org/records/13131875).
```
python evaluation.py --checkpoint PATH_TO_CHECKPOINT --solver ode --nfe 10
``` 

## Training
```
python -m reactot.trainer.train_rpsb_ts1x
```
Note that the default parameters and model types are used in the current command. More detailed instructions on model training will follow in the open-source github repo.

## Data used in this work
1. Transition1x: https://gitlab.com/matschreiner/Transition1x
2. RGD1: https://figshare.com/articles/dataset/model_reaction_database/21066901
3. Berkholz-15: https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.23910
4. DA-41: Private communication. Will be released upon publication
