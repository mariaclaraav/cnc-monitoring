import pandas as pd
import numpy as np

class DataScaler:
    """
    Uma classe para normalizar colunas numéricas em um DataFrame usando um scaler especificado,
    com opções para excluir colunas específicas e reutilizar o scaler para consistência.
    """
    
    def __init__(self, scaler, exclude_columns=None):
        """
        Inicializa a instância do DataScaler.

        Parâmetros:
            scaler (object): Um objeto scaler instanciado (por exemplo, StandardScaler, MinMaxScaler).
                             Deve ser fornecido obrigatoriamente.
            exclude_columns (list, opcional): Lista de colunas a serem excluídas do escalonamento.
                                              Padrão é None (sem exclusões).
        """
        if scaler is None:
            raise ValueError("O parâmetro 'scaler' é obrigatório e deve ser um objeto scaler instanciado.")
        
        self.scaler = scaler
        self.exclude_columns = exclude_columns or []
        self.numeric_cols = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajusta o scaler ao DataFrame e escala as colunas numéricas.

        Parâmetros:
            df (pd.DataFrame): O DataFrame de entrada.

        Retorna:
            pd.DataFrame: Um DataFrame com colunas numéricas normalizadas.
        """
        # Identificar colunas numéricas, excluindo as especificadas
        self.numeric_cols = df.select_dtypes(include=['number']).columns.difference(self.exclude_columns)
        
        # Ajustar e transformar as colunas numéricas
        scaled_values = self.scaler.fit_transform(df[self.numeric_cols])
        scaled_df = pd.DataFrame(scaled_values, columns=self.numeric_cols, index=df.index)

        # Combinar colunas escaladas com o restante do DataFrame
        df_scaled = df.copy()
        df_scaled[self.numeric_cols] = scaled_df

        return df_scaled

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma o DataFrame usando o scaler já ajustado.

        Parâmetros:
            df (pd.DataFrame): O DataFrame de entrada.

        Retorna:
            pd.DataFrame: Um DataFrame com colunas numéricas normalizadas.
        """
        if self.numeric_cols is None:
            raise ValueError("O scaler não foi ajustado. Chame 'fit_transform' primeiro.")

        # Transformar as colunas numéricas
        scaled_values = self.scaler.transform(df[self.numeric_cols])
        scaled_df = pd.DataFrame(scaled_values, columns=self.numeric_cols, index=df.index)

        # Combinar colunas escaladas com o restante do DataFrame
        df_scaled = df.copy()
        df_scaled[self.numeric_cols] = scaled_df

        return df_scaled
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reverts the scaling operation, transforming the scaled data back to its original values.

        Parameters:
            df (pd.DataFrame): The scaled DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with the numeric columns reverted to their original values.
        """
        if self.numeric_cols is None:
            raise ValueError("The scaler has not been fitted. Call 'fit_transform' first.")

        # Handle both DataFrame and numpy array inputs
        if isinstance(df, pd.DataFrame):
            # Ensure only numeric columns are transformed
            original_values = self.scaler.inverse_transform(df[self.numeric_cols])
            original_df = pd.DataFrame(original_values, columns=self.numeric_cols, index=df.index)

            # Combine reverted columns with the rest of the DataFrame
            df_reverted = df.copy()
            df_reverted[self.numeric_cols] = original_df
            return df_reverted

        elif isinstance(df, np.ndarray):
            # If df is a numpy array, assume it matches the numeric column structure
            return self.scaler.inverse_transform(df)

        else:
            raise TypeError("Input must be a pandas DataFrame or a numpy array.")
