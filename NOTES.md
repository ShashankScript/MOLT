# Notes

These are my working notes for MOLT and some things I still want to try.

## Design decisions

 **Score before labels**  
  The model checks a window before its labels become available. Even with zero delay the labels are only available from the next window.

 **ADWIN for retraining**  
  I used fraud misses to trigger retraining. PSI and KS are only used to check changes in the data.

 **Champion vs challenger**  
  I calibrated both models to keep the false positive rate near 1%. Then I tested them on a separate window.

 **Sampling**  
  I kept all fraud cases and 25% of the normal transactions. This makes the experiments easier to run but the results may not match real transactions.

 **Oracle strategy**  
  The oracle is just a reference with zero label delay. It is not the best possible result.

 **Cost calculation**  
  I used the transaction amount as the cost of a missed fraud. Each alert adds 5 cost units. These are only estimates and not actual money.

## Known issues

 The drift is created for the experiment. I have not tested natural drift yet.
 I used 5 seeds to check the results.
 Some windows have very few fraud cases so the recall can change a lot.
 Drift-triggered retraining did not perform as well as periodic retraining in my experiments.

## Things to try next

 Try different ADWIN settings to detect drift faster.
 Check if PSI or KS can help trigger retraining.
 Try periodic retraining with a drift detector to avoid unnecessary training.
 Test the system on another dataset with natural drift.

## My own notes

I will update this section as I continue working on the project.

 What I changed and what happened
 What errors I faced and how I fixed them
 What I learned from the results
 What I would improve next time