import os
import pandas as pd
import numpy as np


def process_excel_data(file_path="../../data/clinical_data.xlsx", debug=False) -> pd.DataFrame:
    """
    Process the Excel file and return a DataFrame.

    Args:
        file_path (str): Path to the Excel file.
        debug (bool): If True, prints additional debug information.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    pd.set_option('future.no_silent_downcasting', True)

    df_raw = pd.read_excel(file_path)
    print(f"Successfully loaded the excel file")
    admissible_features = [
        "WIEK",  # Age
        "PLEC",  # Gender
        "male sex",
        "GLASGOW",
        "Operowany przed przyjęciem (0/1)",  # Surgery before admission (0/1)
        "CH_PRZEW_Brak",  # Lack of chronic diseases
        "marskość wątroby i nadciśnienie wrotne",  # Liver cirrhosis and portal hypertension
        "niewydolność krążenia_NYHA IV",  # Heart failure NYHA IV
        "przewlekłe ciężkie choroby układu oddechowego",  # Chronic severe respiratory diseases
        "niewydolność nerek wymagajaca dializ",  # Kidney failure requiring dialysis
        "obniżenie odporności",  # Immunosuppression
        # ...existing feature list...
    ] # Thanks to Karol and Witek

    # Filter the DataFrame to keep only admissible features
    df = df_raw.loc[:, admissible_features + ["ZGON wewnątrzszpitalnie"]]  # Hospital mortality

    # Convert 'ZGON wewnątrzszpitalnie' to binary values
    df = df.replace({'ZGON wewnątrzszpitalnie': {'TAK': 1, 'NIE': 0}})  # YES/NO to 1/0

    # Change name of 'ZGON wewnątrzszpitalnie' to 'ZGON'
    df = df.rename(columns={'ZGON wewnątrzszpitalnie': 'ZGON'})  # Rename to "Mortality"

    # Check if 'KG' column exists and filter it
    df.index = df_raw['KG']  # Medical record number as index
    df = df.rename_axis("id")

    # Check if 'PLEC' column exists and filter it
    df = df.drop('PLEC', axis=1)  # Drop gender column (duplicate with 'male sex')

    print("Successfully processed the Excel file")

    if debug:
        # ...existing debug code...

    return df

if __name__ == "__main__":
    # Set the path to your Excel file
    excel_file_path = "../../data/clinical_data.xlsx"

    try:
        df = process_excel_data(file_path=excel_file_path, debug=True)
        df.to_csv("../../data/processed_tabular.csv")
        print("Processed DataFrame:\n", df.head())
    except Exception as e:
        print(f"An error occurred: {e}")
