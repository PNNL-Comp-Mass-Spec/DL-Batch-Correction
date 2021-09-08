# DL-batch-correction

A deep learning model for removing batch effects in microarray data. 

## Requirements

- matplotlib == 3.4.2
- numpy == 1.20.3
- pandas == 1.2.4
- torch == 1.8.1

## Usage

### Input

The data is assumed to be in matrix form with columns as samples and features as rows.

```
python main.py --path_to_data .\data\emory\emory_data_clean.txt /
  --path_to_labels .\data\emory\emory_batch_labels.txt /
  --output_dir .\experiments\emory_normae_pipeline /
  --num_epochs 1000 --batch_size 8
```

### Output

The above command will place two files in the output directory:

- `loss.png`: a plot of loss vs. number of epochs
- `output.cvs`: reconstructed input data
