from itertools import combinations

class JaccardSimilarity:
    def __init__(self, y_pred):
        """
        Initializes the class with a dictionary of binary predictions.

        Parameters:
        y_pred (dict): Dictionary where keys are operation names and values are binary lists.
        """
        self._y_pred = y_pred

    @staticmethod
    def get_for_lists(y1, y2):
        """
        Calculates the Jaccard similarity between two binary lists.

        Parameters:
        y1 (list): Binary list (0s and 1s) representing anomalies from Model 1.
        y2 (list): Binary list (0s and 1s) representing anomalies from Model 2.

        Returns:
        similarity (float): Jaccard similarity between the lists.
        """
        # Convert the lists to sets of indices where anomalies (1s) occur
        set1 = set(i for i, val in enumerate(y1) if val == 1)
        set2 = set(i for i, val in enumerate(y2) if val == 1)

        # Calculate the intersection and union of the sets
        intersection = set1.intersection(set2)
        union = set1.union(set2)

        # Calculate the Jaccard similarity
        similarity = len(intersection) / len(union) if len(union) > 0 else 0

        return similarity

    def get_between_operations(self):
        """
        Calculates the Jaccard similarity between all combinations of operations in y_pred.

        Returns:
        jaccard_results (dict): Dictionary with Jaccard similarities for each pair of operations.
        """
        # Extract the operation names
        operations = list(self._y_pred.keys())

        # Dictionary to store the results
        jaccard_results = {}

        # Generate all combinations of operation pairs
        for op1, op2 in combinations(operations, 2):
            # Calculate the Jaccard similarity
            similarity = self.jaccard_similarity_binary(self._y_pred[op1], self._y_pred[op2])

            # Store the result
            jaccard_results[f"{op1} vs {op2}"] = similarity

        return jaccard_results