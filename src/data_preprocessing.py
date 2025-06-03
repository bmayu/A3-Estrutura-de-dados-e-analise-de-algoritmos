import pandas as pd
import os

RAW_PATH = "./data/raw/creditcard.csv"
PROCESSED_PATH = "./data/processed/creditcard_clean.csv"

def preprocess_data():
    df = pd.read_csv(RAW_PATH)

    # Verifica dados nulos
    print("Verificando dados nulos:")
    print(df.isnull().sum())

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Dados limpos salvos em: {PROCESSED_PATH}")

if __name__ == "__main__":
    preprocess_data()
