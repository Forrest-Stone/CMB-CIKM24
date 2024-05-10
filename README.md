# CMB-CIKM24

## Overall
Pytorch implementation for paper "Counterfactual Multi-player Bandits for Explainable Recommendation Diversification".

## Requirements
- Python 3.9
- pytorch 1.13.0
- cuda 11

## Instruction
1. You may download Amazon Review dataset from https://nijianmo.github.io/amazon/index.html.

2. We provide an example on "CDs and Vinyl" datasets. The pre-processing data within subtopic information are already in the "dataset/CDs" folder.

4. To set the python path, under the project root folder, run:
    ```
    source setup.sh
    ```
5. To train the base recommender: run:
    ```
    python scripts/train_base_amazon.py
    ```
6. To run cmb method, run:
    ```
    python scripts/generate_div_amazon.py
    ```
