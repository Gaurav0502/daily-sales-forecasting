# Contribution guidelines

(This applies only to the collaborators added to this repository)

- Work only in your respective branches and NOT in main.

```bash
git checkout <branch-name>
```

- If you use any new module from Python, please add it inside requirements.txt.


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
├── clustering.ipynb
├── clusters.json
├── data
│   ├── online_retail_II.xlsx
├── eda.ipynb
├── model.py
├── modelling.ipynb
├── preprocess.py
└── requirements.txt

```
