# Commands

## Create virtual env (with UV if you have aura)

```
cd backend
```

## Run project
Run with:
```

pixi run run
```

This targets `app.py` by default.

## Vegetation Data

To run the model, you need vegetation data which is sourced from the [Forest Service in the US Department of Agriculture](https://research.fs.usda.gov/products/dataandtools/fia-datamart).

1) Go to the site.

2) Select "NFI Data". 

3) Select "CSV" as the Data Type.

4) Select State(s) as "California" only.

5) Inside of "Data Available for Download", there should be a series of CSV files with the name scheme "CA_<some-text>.csv"; download the following of these files:

- CA_COND.csv
- CA_GRND_CVR.csv
- CA_PLOT.csv
- CA_P2VEG_SUBPLOT_SPP.csv
- CA_SEEDLING.csv

6) Put all of these files into `backend/Datasets/vegetation`. 