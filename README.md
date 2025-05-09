# Daily Sales Forecasting

## Aim

To forecast the daily sales of products with the holiday and promotional sales information. This information includes:

1. Holiday Features (Christmas, is_weekend, is_holiday, days_to_christmas)

2. Promotional Sales (Easter sale, Back to School sale, Black Friday sales, and Boxing Day sale)

3. Time-series Features (7-day lag, 7-day rolling mean)

## Environment Setup

- Clone this repository.

```bash

git clone https://github.com/Gaurav0502/daily-sales-forecasting.git

```

- Install all packages in the ```requirements.txt``` file.

```bash

pip install -r requirements.txt

```

- Download and store all the three datasets from following sources:

1. Online retail dataset: https://archive.ics.uci.edu/dataset/352/online+retail

2. Clusters and Event dataset: https://www.kaggle.com/datasets/gauravpendharkar/cluster-and-events-data


## Results

| Model    | Overall MAPE     | Model Complexity    | Dashboard     |
|--------------|--------------|--------------|--------------|
| TFT | 21.70 | 0.2 | [wandb](https://wandb.ai/gauravpendharkar/TFT%20Window-based%20Evaluation?nw=nwusermitugaurav15) |
| NHiTS | 18.70 | 0.2 | [wandb](https://wandb.ai/gauravpendharkar/NHiTS%20Window-based%20Evaluation/workspace?nw=nwusermitugaurav15) |
| DeepAR | TBA | 0.2 | wandb |

## References

Online Retail dataset: https://archive.ics.uci.edu/dataset/502/online+retail+ii

