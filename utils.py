import pandas as pd

def write_predictions_to_csv(predictions, output_file):
    """Write the result to a csv file.
    
    The `predictions` is a dictionary mapping from id to output.
    """

    predictions_df = pd.DataFrame([
        {"id": k, "prediction": v}
        for k,v in predictions.items()
    ])
    predictions_df.to_csv(output_file, index=False)

def load_predictions_from_csv(input_file):
    """Read the predictions from a csv file."""
    pred_df = pd.read_csv(input_file)
    predictions = {}
    for _, row in pred_df.iterrows():
        predictions[row["id"]] = row["prediction"]
    return predictions
    