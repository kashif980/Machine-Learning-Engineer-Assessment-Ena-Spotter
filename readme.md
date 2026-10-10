# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions "december-chart-inputs.csv"
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

# Freight Rate Prediction Challenge

See `freight-rate-ml-assessment.pdf` for the assessment instructions.

## What to Do

1. Train and validate the model using `train-test.csv`.
2. Predict every load in `validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `validation-predictions-template.csv` and save the final file as `validation_predictions.csv`.
4. Predict every row in `december-chart-inputs.csv` by filling its `predicted_rate` column.
5. Install dependencies and run the scorer using the commands below.

## Running the Scorer

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Validate Prediction Files

```bash
python score.py --predictions validation_predictions.csv --december-predictions "december-chart-inputs.csv"
```

### 3. Expected Output

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

### 4. Generated Chart

The December chart is saved to:

`scorer_results/candidate_december.png`

## Important Notes

- Run the commands from the project root directory.
- Ensure the required files are present in the project root.
- The scorer validates the prediction outputs and generates the December chart. It does not calculate Spotter's official model evaluation score.

## Submit

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link
