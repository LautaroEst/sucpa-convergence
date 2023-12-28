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


## Results



## Citation
