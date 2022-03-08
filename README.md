# DL-batch-correction

A deep learning model for removing batch effects in microarray data. 

## Requirements

- matplotlib == 3.4.2
- numpy == 1.20.3
- pandas == 1.2.4
- torch == 1.8.1

## Usage

### Input

The data is assumed to be in matrix form with columns as samples and features as rows, and `\t` delimiters.

Expression matrix:
```
feature   Sample_1 Sample_2 ... Sample_n
Feature_1 -0.1661  -1.2160  ... -0.4630
Feature_2 -1.7001  -0.06567 ...  0.8384
...
Feature_m -0.4630   0.8384  ... -0.8774
```

Sample labels:
```
sample   group batch
Sample_1 1     2
Sample_2 1     1
Sample_3 1     2
...
Sample_n 2     2
```

## Notebooks

* [Transformer Simple Correction (NEW)](https://colab.research.google.com/drive/1OpS4fzwI_v09rUapaXo7Ki0_kSxZVN-0?usp=sharing)
* [NormAE demo](https://colab.research.google.com/drive/1RLyh5yqNW8DD9vyjBL0pQoXOaqnTBAXO?usp=sharing)
* [Peptide transformer](https://colab.research.google.com/drive/1GAh0lC1hI-DEFqzLxjaRbO4eRKshXwL3?usp=sharing)
