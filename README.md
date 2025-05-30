<div align="center">
<h1>Daily Sales Forecasting</h1>
</div>

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

- The following directory structure is required for the code in this repository to work properly:

```bash
.
├── data
│   ├── clusters
│   │   ├── cluster_0.csv
│   │   ├── cluster_1.csv
│   │   ├── cluster_2.csv
│   │   ├── cluster_3.csv
│   │   ├── cluster_4.csv
│   │   ├── cluster_5.csv
│   │   ├── cluster_6.csv
│   │   ├── cluster_7.csv
│   │   ├── cluster_8.csv
│   │   └── cluster_9.csv
│   ├── events.json
│   └── online_retail_II.xlsx
├── dataprocessor.py
├── deepar
│   ├── config.py
│   └── deepar.py
├── evaluator.py
├── modelling.ipynb
├── nhits
│   ├── configs.py
│   └── nhits.py
├── README.md
├── requirements.txt
├── sales_holidays.py
├── setup.sh
├── tft
│   ├── configs.py
│   ├── tft.py
│   └── tuner.py
```
- To use the code in this repository, use the following code snippet:
  
```python

# custom data processor (can be modified as required)
from dataprocessor import DataProcessor

# custom model building and training
# (works only if .py or .ipynb is in root directory of the project)

# DeepAR
from deepar.config import CrossValConfigs, DataSetConfigs, DeepARConfigs
from deepar.deepar import DeepAR_model

# NHiTS
from nhits.config import CrossValConfigs, DataSetConfigs, OptimizerConfigs, NHiTSConfigs
from nhits.nhits import NHiTS_model

# TFT
from tft.config import CrossValConfigs, DataSetConfigs, TFTConfigs
from tft.tft import TFT

# custom model evaluator
from evaluator import Evaluator

# data pre-processing
# and feature engineering
dp = DataProcessor(data_dir = <csv-dataset-file-path>, clusters = <list-of-clusters>)
dp.process()
data = dp.data

# model training
# (update accordingly for TFT or NHiTS)
deepar = DeepAR_model(
    data = data,
    configs = {
        "cross_validation": CrossValConfigs(),
        "dataset": DataSetConfigs(),
        "deepar": DeepARConfigs()
    }
)

deepar.cross_validate(
    MIN_TIME_IDX = data["time_idx"].max() - 90,
    MAX_TIME_IDX = data["time_idx"].max()
)

# model evaluation
evaluator = Evaluator(deepar.cv_results)
evaluator.print_mape_summary()
evaluator.forecast_error_by_region()
evaluator.mape_by_forecast()

```
- Feel free to raise an issue if there any issues with the repository!
  
## Results

<div align="center">

| Model    | Overall MAPE     | Model Complexity    | Dashboard     |
|--------------|--------------|--------------|--------------|
| TFT | 21.70 | 0.2 | [wandb](https://wandb.ai/gauravpendharkar/TFT%20Window-based%20Evaluation?nw=nwusermitugaurav15) |
| NHiTS | 18.70 | 0.2 | [wandb](https://wandb.ai/gauravpendharkar/NHiTS%20Window-based%20Evaluation/workspace?nw=nwusermitugaurav15) |
| DeepAR | 19.51 | 0.2 | [wandb](https://wandb.ai/gauravpendharkar/DeepAR%20Window%20based%20evaluation/overview) |
</div>

where:

$$ \text{Model Complexity} = \frac{\text{Number of Models}}{\text{Number of Clusters}} $$

## References

Online Retail dataset: https://archive.ics.uci.edu/dataset/502/online+retail+ii

Demand Forecasting using TFT: https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html

Autoregressive modelling with DeepAR and DeepVAR: https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/deepar.html

