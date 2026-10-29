# Dataset attribution and reuse notice

MOLT uses the **Credit Card Fraud Detection** dataset by the Machine Learning Group at ULB and Worldline, distributed through Kaggle. The Kaggle dataset page identifies the database license as Open Database License (ODbL) 1.0 and the contents license as Database Contents License (DbCL) 1.0. Review the source and full license texts before reusing or redistributing the data:

- Dataset page: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Open Database License 1.0: https://opendatacommons.org/licenses/odbl/1-0/
- Database Contents License 1.0: https://opendatacommons.org/licenses/dbcl/1-0/

The small `app_data/creditcard_sample.parquet` file is a derived sample containing every fraud row plus a seeded sample of 5,000 legitimate rows. Numeric features are converted to float32 and the class label to int8. It is included with this notice and the transformation code in `experiments/run_all.py`. The source listing states that the database is under ODbL and its contents under DbCL; this project does not claim ownership of the dataset or make a legal determination for every use case. Preserve attribution and comply with applicable attribution/share-alike requirements when using or redistributing the database or derived sample.

Suggested attribution: “Credit Card Fraud Detection dataset, collected and analysed in a research collaboration of Worldline and the Machine Learning Group, Université Libre de Bruxelles (ULB); distributed via Kaggle. Dataset page: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.”

The dataset contains anonymized records from two days in September 2013. The `Amount` field is used only as a dataset amount-unit cost proxy in experiments; reported amount units are not realized financial savings.
