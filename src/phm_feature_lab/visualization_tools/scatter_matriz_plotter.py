import pandas as pd
import plotly.express as px

class ScatterMatrixPlotter:
    def __init__(self, df):
        """
        Inicializa a classe com o DataFrame.

        Parâmetros:
        - df: DataFrame contendo os dados.
        """
        self.df = df

    def plot_scatter_matrix(self, highlight_column=None, machine=None, process=None, cols=None, sample_frac=0.05, random_state=0):
        """
        Plota uma matriz de dispersão (scatter matrix) para as colunas especificadas,
        destacando uma coluna específica (Unique_Code, Machine, Period, etc.).

        Parâmetros:
        - highlight_column: Coluna a ser destacada (ex: 'Unique_Code', 'Machine', 'Period').
        - machine: Filtro para a coluna 'Machine' (opcional).
        - process: Filtro para a coluna 'Process' (opcional).
        - cols: Lista de colunas a serem incluídas na matriz de dispersão.
        - sample_frac: Fração do DataFrame a ser amostrada (padrão 0.05).
        - random_state: Seed para geração de números aleatórios (padrão 0).
        """
        # Filtrar por máquina e processo, se especificado
        df_filtered = self.df.copy()
        if machine:
            df_filtered = df_filtered[df_filtered["Machine"] == machine]
        if process:
            df_filtered = df_filtered[df_filtered["Process"] == process]

        # Garantir que apenas colunas existentes no DataFrame sejam usadas
        if cols:
            cols = [col for col in cols if col in df_filtered.columns]
        else:
            cols = df_filtered.columns.tolist()

        # Verificar se a coluna de destaque existe no DataFrame
        if highlight_column and highlight_column not in df_filtered.columns:
            raise ValueError(f"A coluna '{highlight_column}' não existe no DataFrame.")

        # Ordem dos valores únicos para a coluna de destaque
        if highlight_column:
            highlight_order = df_filtered[highlight_column].unique()
        else:
            highlight_order = None

        # Criar a matriz de dispersão
        fig = px.scatter_matrix(
            df_filtered.sample(frac=sample_frac, random_state=random_state),
            dimensions=cols,
            color=highlight_column,
            category_orders={highlight_column: list(highlight_order)} if highlight_column else None,
        )

        # Atualizar layout
        fig.update_layout(
            width=1400,
            height=1000,
            legend_title_font_size=18,
            legend_title_text=highlight_column if highlight_column else "Nenhum destaque",
        )

        # Atualizar características dos traços
        fig.update_traces(marker=dict(size=2), diagonal_visible=False, showupperhalf=False)

        # Exibir a figura
        fig.show()