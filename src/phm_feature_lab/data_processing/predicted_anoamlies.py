import os
import json
import numpy as np
import pandas as pd

class PredictedAnomalies:
    def __init__(self, saving_path, group, df_test):
        """
        Initialize the etestuator with the path to save results.
        
        Parameters:
        - saving_path (str): Directory where the JSON results will be saved.
        """
        self.saving_path = saving_path
        self.group = group
        self.__df_test = df_test

    def evaluate(self, X_test, test_codes, y_pred, y_test):
        """
        Etestuate anomalies and prepare test results.
        
        Parameters:
        - X_test (pd.DataFrame or np.ndarray): Validation feature set.
        - test_codes (list): Unique codes for the test set.
        - y_pred (list): Predicted anomaly labels.
        - y_test (list): Ground truth labels.
        - df_test (pd.DataFrame): DataFrame containing additional metadata like 'Operation'.
        
        Returns:
        - test (pd.DataFrame): Validation DataFrame with scores and labels.
        - anomaly_scores (pd.DataFrame): Grouped anomaly scores by 'Unique_Code'.
        """
        # Ensure X_test is a DataFrame
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)

        # Prepare test DataFrame
        test = X_test.copy()
        test['Unique_Code'] = test_codes
        test['Anomaly'] = y_pred
        test['Label'] = y_test

        # Ensure 'Operation' exists in df_test
        if 'Operation' in self.__df_test:
            test['Operation'] = self.__df_test['Operation'].values
        else:
            test['Operation'] = "Unknown"

        # Group by 'Unique_Code' and calculate the anomaly percentage
        anomaly_scores = (
            test.groupby('Unique_Code')
            .apply(lambda group: (group['Anomaly'] == 1).sum() / len(group) * 100)
            .reset_index(name='Anomaly_Score')
        )

        # Merge the calculated scores back into the original DataFrame
        test = test.merge(anomaly_scores, on='Unique_Code', how='left')

        return test, anomaly_scores


    def save_results(self, test, anomaly_scores):
        """
        Save the results to a JSON file.

        Parameters:
        - test (pd.DataFrame): Validation DataFrame with scores and labels.
        - anomaly_scores (pd.DataFrame): Grouped anomaly scores by 'Unique_Code'.
        """
        # Ensure 'Operation' exists in anomaly_scores
        if 'Operation' not in anomaly_scores.columns:
            anomaly_scores['Operation'] = test['Operation']

        # Group results by operation
        grouped_results = anomaly_scores.groupby("Operation").apply(
            lambda group: [
                {"unique_code": row["Unique_Code"], "predicted": np.round(row["Anomaly_Score"], 2)}
                for _, row in group.iterrows()
            ]
        ).to_dict()

        # Include true anomaly labels
        true_anomalies = self.__df_test[self.__df_test["Label"] == 1]["Unique_Code"].unique().tolist()

        # Prepare the final results
        results = {
            "operations": {
                operation: {"anomalies": grouped_results[operation]}
                for operation in grouped_results
            },
            "label": true_anomalies
        }

        # Define the path to save the JSON
        output_path = os.path.join(self.saving_path, f"group{self.group}_results.json")

        # Ensure the saving directory exists
        os.makedirs(self.saving_path, exist_ok=True)

        # Save results to JSON
        with open(output_path, "w") as json_file:
            json.dump(results, json_file, indent=4)

        print(f"Results saved to {output_path}")