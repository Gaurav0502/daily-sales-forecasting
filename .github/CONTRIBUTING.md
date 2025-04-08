# Contribution guidelines

(This applies only to the collaborators added to this repository)

- Work only in your respective branches and NOT in main.

```bash
git checkout <branch-name>
```

- If you use any new module from Python, please add it inside requirements.txt.

- To get the aggregated daily sales data, use the ```cluster.py```.

```python

from cluster import get_daily_sales

df = get_daily_sales(<integer between 0 and 9 for cluster ID>)

```


# Guidelines for pull request
Before making a pull request, ensure the following requirements are fulfilled:

- Pull the code from main to the respective branch.

```bash
git pull origin main
```

- The dataset(s) are populated under the data directory (will be ignored by Git). Due to the size of the dataset, no data files must be pushed into your respective branch.

- No documents must be pushed to Github (.pptx, .pdf, .docx, etc). If you have any documents locally, populate them into the documents directory (will ignored by Git). The documents will be added as google drive links in the README.md file.

- The overall directory structure must be as follows:

```bash
.
├── README.md
├── cluster.py
├── data
│   ├── clusters
│   │   ├── cluster_0.csv
│   │   ├── cluster_1.csv
│   │   ├── cluster_2.csv
│   │   ├── cluster_3.csv
│   │   ├── cluster_4.csv
│   │   ├── cluster_5.csv
│   │   ├── cluster_6.csv
│   │   ├── cluster_7.csv
│   │   ├── cluster_8.csv
│   │   └── cluster_9.csv
│   └── online_retail_II.xlsx
├── preprocess.py
├── requirements.txt
└── rough.ipynb

4 directories, 17 files

```
