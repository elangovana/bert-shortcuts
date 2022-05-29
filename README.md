# shortcuts

## To create word shuffles

``` bash
 python src/utils/convert_shuffle.py --csv_sep ',' --output temp/outdata_shuffle.csv  temp/data/dev_matched_mini.csv "sentence1,sentence2"
```

## Helper to convert tsv to csv

```bash
 python src/utils/convert_to_csv.py --output temp/dev_matched.csv temp/data/dev_matched.tsv gold_label entailment,neutral,contradiction

```

## Differences

What factors lead to seemingly improved performance?

1. Unknown words
2. Base Architecture - LSTM vs Transformers
3. Architecture size
4. Pretraining Training procedure - None vs LM vs NSP
5. Training time / epochs

What shortcuts has the model learnt?

1. Bag of words similarity?

- Superficial lexical similarity between train and test?
- Difference in FP vs FN
- High superficial similarity, lead to similar labels?

2. Impact on generalisability
    1. Differentiate between False positives and False negatives, application specific
    2. Assume that
        1. false positives are likely due to lexical / superficial similarity
        2. False negative are most likely due to lexical / superficial dissimilarity
    3. Is Label confidence more telling that the label??
        1. High confidence associated with superficial sim to training data, regardless of the label
    4. Top key words (per class)
       Evidence
    5. PPI PTM paper shows high confidence high similarity to commonly occurring train words,
    6. McCoy shows that test accuracy is consistent across several runs , while new dataset changes
    7. Shuffling words result in majority of the labels remaining the same.
    8. NSP is what causes increase in BERT over previous SOTA

- Large scale, short cuts become apparent
- Large scale bow ?
- Confidence calibration
    - high confidence indicate high lexical similarity , low confidence show lower lexical similarity
    - it has learnt some short cuts to work on test set, random chance?
    - in distribution vs out distribution
        - What is out of distribution in NLP?
            - We know minor changes affect performance. E.g. McCoy and word shuffle experiments.
            - High confidence generally represents high lexical similarity to training data
            - In practice, say you have training data of 100 records and you are able to identify 20% addiional record
              of interest at low cost/effort we can say that we have recooperated
            - Question is can we reliably detect the 20 records, cos not all predictions are correct and we do not know
              ground truth.
                - High confidence is an indicator that the data is likely to be lexically similar to training
                - But lexical similarity does not mean than the prediction is correct., e.g. word shuffle in mlni not
                  changing output
                - if we are interested in reducing in FP, and we model has predicted with high confidence that it is a
                  positive class we are intersted in..
                    - Shuffle doesnt change
            - Question is in a pool of 100, 80 are correct, we shortlist pretty effectively. If a person spends 1 minute
              per verification, and annotation time is 1
                - v = a/n and hit ratio is 1/h
                - baseline = h * n * v per loi
                - 1 in h * a images are correct , then the model break even
                - 1 in h * a/n images are correct , then the model is 200% efficient
                - at essentially 2/n % accuracy
                - More high similarity samples train
            - high confidence errors are more effective?

        - Low resource- https://aclanthology.org/2020.coling-main.100.pdf
   
