# SUCPA Convergence

This is the code to analyze the convergence of the SUCPA algorithm.

## Requirements

Check that you have the packages listed in `requirements.txt` installed. If not, run:

```bash
conda create --name sucpa python=3.10
conda activate sucpa
pip install -r requirements.txt
```

## Usage

### Running SUCPA on NLP datasets

To run the SUCPA algorithm for a specific value of $\beta^{[0]}$ (for instance, $\beta^{[0]}=\begin{bmatrix} 0.1 & 0.2 \end{bmatrix}$) run:

```bash
python run_sucpa.py \
    --dataset=sst2 \
    --steps=10 \
    --beta_init=0.1,0.2
```

You can also run the SUCPA algorithm for a range of values of $\beta^{[0]}$ by defining a number of repetitions and an initial seed:

```bash
python run_sucpa.py \
    --dataset=sst2 \
    --steps=10 \
    --repetitions=10 \
    --random_state=3892
```

In all cases, the results will be saved in the `results/dataset={dataset}` folder. Supported datasets for these experiments are `sst2` (two classes) and `mnli` (three classes). Logits for these datasets are contained in the `data` directory and were obtained with the code in [this repository](https://github.com/LautaroEst/llmcal).


### Running SUCPA on image dataset

The procedure tu run SUCPA on image datasets is the same as for NLP datasets, so you just need to run (for instance):

```bash
python run_sucpa.py \
    --dataset=cat-dog \
    --steps=10 \
    --beta_init=0.1,0.2
```

The logits for the image datasets are contained in the `data` directory and were obtained fine-tuning a ResNet18 model on the Cat-Dog dataset of [this](https://www.kaggle.com/competitions/dogs-vs-cats/data) Kaggle competition. To obtain the logits and labels contained in the `data/dogs-vs-cats/` directory you can download the dataset, open it on the `./data` directory and run the `run_resnet.py` script.


## Results



## Citation
