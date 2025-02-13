# React-OT:Optimal Transport for Generating Transition State in Chemical Reactions
==============================

OA-ReactDiff is the first diffusion-based generative model for generating 3D chemical reactions, which not only accelerates the search for 3D transition state in chemical reactions by a factor of 1000, but also generates and explores new and unknown chemical reactions. 
In this work, we developed React-OT, an optimal transport approach to generate TSs of an elementary reaction in a fully deterministic manner.Compared to OA-ReactDiff, React-OT eliminates the need of training an additional ranking model and reduces the number of inference evaluations of the denoising model from 40,000 to 50, achieving a near 1000-fold acceleration. With React-OT, highly accurate TS structures can be deterministically generated in 0.4 seconds.

![image](https://github.com/deepprinciple/react-ot/blob/main/reactot/Figures/figure1.jpg)
Fig. 1 | Overview of the diffusion model and optimal transport framework for generating TS.

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
Note that the default parameters and model types are used in the current command. More detailed instructions on model training will be updated soon.

## Data used in this work
1. Transition1x: https://gitlab.com/matschreiner/Transition1x
2. RGD1: https://figshare.com/articles/dataset/model_reaction_database/21066901
3. Berkholz-15: https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.23910
