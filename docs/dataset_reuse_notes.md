# Credit-card dataset source and reuse notes

The Kaggle page for the ULB/Worldline **Credit Card Fraud Detection** dataset describes 284,807 anonymized transactions over two days, including 492 frauds; it says V1–V28 are PCA-transformed features, with Time and Amount untransformed. Its license metadata lists **Database: Open Database** and **Contents: Database Contents**.

Source page: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The official ODbL 1.0 text permits database use and derivative databases subject to conditions. For public distribution/use, relevant terms include attribution/notices, share-alike for derivative databases, and access to the derivative database or the changes/method that produced it. The official DbCL 1.0 grants copyright permissions over contents but says those permissions are subject to compliance with ODbL.

ODbL 1.0: https://opendatacommons.org/licenses/odbl/1-0/
DbCL 1.0: https://opendatacommons.org/licenses/dbcl/1-0/

MOLT's included Parquet file is a small derived sample, not the full source data. It contains all fraud rows plus a seeded subset of 5,000 legitimate rows, with numeric features cast to float32 and Class cast to int8. The sampling method is included in `experiments/run_all.py`. This is a factual source summary for project documentation, not legal advice; users should independently review current source terms and applicable law before redistribution or use.
