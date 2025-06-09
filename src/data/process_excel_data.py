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
        "WIEK",
        "PLEC",
        "male sex",
        "GLASGOW",
        "Operowany przed przyjęciem (0/1)",
        "CH_PRZEW_Brak",
        "marskość wątroby i nadciśnienie wrotne",
        "niewydolność krążenia_NYHA IV",
        "przewlekłe ciężkie choroby układu oddechowego",
        "niewydolność nerek wymagajaca dializ",
        "obniżenie odporności",
        "GLASGOW.1",
        "MAP 1sza doba",
        "pao2/fio2 1sza doba",
        "BILI TISS 1",
        "dopamina przyjęcie tak=1, nie = 0 (TISS nr 1)",
        "dopamina  dawka (TISS nr 1)",
        "noradrenalina przyjęcie: tak=1, nie =0 (TISS nr 1)",
        "noradrenalina przyjęcie dawka ug/kg/min (TISS nr 1)",
        "dobutamina przyjęcie tak=1, nie =0 (TISS nr 1)",
        "CTK skurczowe przyjęcie (TISS nr 1)",
        "CTK rozkurczowe przyjęcie (TISS nr 1)",
        "akcja serca przyjęcie (TISS nr 1)",
        "temperatura ciała przy przyjęciu (TISS nr 1)",
        "FIO2 (TISS nr 1)",
        "Antybiotyki Tak=1, NIE =0)",
        "argipresyna/empesin TAK=1, NIE=0 (TISS nr 1 godz. 0.00)",
        "argipresyna dawka  (TISS nr 1 godz. 0.00)",
        "diureza w ml w ciągu pierwszych 6 godzin od przyjęcia ",
        "podaż płynów w dobie przyjęcia (ml) - okienko \"RAZEM\" po prawej stronie pod wierszem \"woda endogenna\" (TISS nr 1)",
        "bilans płynów w dobie przyjęcia (ml) (TISS 1)",
        "pH (1. gaz. 1 TISS)",
        "pCO2 (1. gaz. 1 TISS)",
        "pO2 (1. gaz. 1 TISS)",
        "Hb (1. gaz. 1 TISS)",
        "K (1. gaz. 1 TISS)",
        "Na (1. gaz. 1 TISS)",
        "Ca2+ (1. gaz. 1 TISS)",
        "Cl (1. gaz. 1 TISS)",
        "sodium chloride difference",
        "Glukoza (1. gaz. 1sza doba)",
        "Lac (1. gaz. 1sza doba)",
        "Crea (1. gaz. 1 TISS)",
        "Bil (1. gaz. 1 TISS)",
        "BE (1. gaz. 1sza doba)",
        "HCO3 (1. gaz. 1 TISS)",
        "mOsm (1. gaz. 1 TISS)",
        "COVID-19",
        "cukrzyca",
        "nadciśnienie",
        "niewydolność nerek przewlekła",
        "POCHP",
        "astma",
        "niewydolność wątroby przewlekła",
        "przewlekła choroba niedokrwienna serca",
        "stan po przebyciu zawału serca",
        "niewydolność serca przewlekła",
        "czynna choroba nowotworowa",
        "migotanie przedsionków -jakiekolwiek",
        "choroby psychiczne (depresja, schizofrenia, otępienie)",
        "choroby neurologiczne (Ch. Parkinsona, choroby nerwowo-mięśniowe)",
        "Sepsa (0/1)",
        "sodium chloride difference tiss 1",
        "sodium chloride difference tiss 2",
        "sodium chloride difference tiss 3"
    ]  # Thanks to Karol and Witek

    # Filter the DataFrame to keep only admissible features
    df = df_raw.loc[:, admissible_features + ["ZGON wewnątrzszpitalnie"]]  # Hospital mortality

    # Convert 'ZGON wewnątrzszpitalnie' to binary values
    df = df.replace({'ZGON wewnątrzszpitalnie': {'TAK': 1, 'NIE': 0}})  # YES/NO to 1/0

    # Change name of 'ZGON wewnątrzszpitalnie' to 'ZGON'
    df = df.rename(columns={'ZGON wewnątrzszpitalnie': 'ZGON'})  # Rename to "Mortality"

    # Check if 'KG' column exists and filter it
    df.index = df_raw['KG']  # Medical record number as index
    df = df.rename_axis("id")
    #remove backslashes from index
    df.index = df.index.str.replace('/', '', regex=False)  # Normalize index to avoid issues with slashes

    # Check if 'PLEC' column exists and filter it
    df = df.drop('PLEC', axis=1)  # Drop gender column (duplicate with 'male sex')

    print("Successfully processed the Excel file")

    if debug:
        print(f"DataFrame index type: {type(df.index[0]) if len(df.index) > 0 else 'empty'}")
        print(f"First 5 IDs: {df.index[:5].tolist() if len(df.index) >= 5 else df.index.tolist()}")
        print(f"First 5 rows of the DataFrame:\n{df.head()}")

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
