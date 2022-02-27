# shortcuts

## To create word shuffles

``` bash
 python src/utils/convert_shuffle.py --csv_sep ',' --output temp/outdata_shuffle.csv  temp/data/dev_matched_mini.csv "sentence1,sentence2"
```

## Helper to convert tsv to csv

```bash
 python src/utils/convert_to_csv.py --output temp/dev_matched.csv temp/data/dev_matched.tsv gold_label entailment,neutral,contradiction

```