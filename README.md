# Income Prediction

A notebook that fits three classifiers on the adult census table, pickles gradient boosting, and a Flask form that returns `>50K` or `<=50K`.

Live page: [maadhiniportfolio.vercel.app/projects/income-pred-webapp](https://maadhiniportfolio.vercel.app/projects/income-pred-webapp)

## Files

- `incomeprediction.ipynb` — cleaning, three models, ROC curves, and the pickle export
- `adult.csv` — census-style rows labeled `<=50K` or `>50K`
- `model.pkl` — the gradient boosting classifier, saved with scikit-learn 1.0.2
- `app.py` — Flask form. `GET /` renders the page. `POST /predict` scores one row
- `index.html` — the form
- `procfile` — `web: gunicorn app:app`
- `req.txt` — the Python packages the app was written against

## Notebook

`fnlwgt` and `educational-num` are dropped. `?` is filled with the most common value in that column. `LabelEncoder` turns workclass, education, marital status, occupation, relationship, race, gender, country, and income into integers. Income `0` is `<=50K`. Income `1` is `>50K`.

Random forest, gradient boosting, and AdaBoost each train on a 70/30 split with `random_state=100`. Reported accuracy is about 0.85 to 0.86. `model.pkl` is the gradient boosting fit (`n_estimators=100`).

## Flask app

The form posts 12 fields, in this order: age, workclass, education, marital status, occupation, relationship, race, gender, capital gain, capital loss, hours per week, native country. The app casts them to ints, reshapes them to one row, and calls `predict`.

`model.pkl` loads on scikit-learn 1.0.2. A newer scikit-learn install fails because `sklearn.ensemble._gb_losses` is no longer there. `req.txt` pins Flask 1.1.1 and leaves scikit-learn as `>=0.18`, so a fresh install needs scikit-learn pinned to 1.0.2.

`render_template` looks for `templates/index.html`. The form file is `index.html` at the repository root, so the app does not find the template until that file is moved.

Workclass, occupation, and country options in the form start at 1. The notebook's encoder starts at 0, and the country list skips Holand-Netherlands, so those three fields do not match the trained codes. Education, marital status, relationship, race, and gender do match. Numeric ranges in the data are age 17–90, capital gain 0–99999, capital loss 0–4356, and hours per week 1–99.
