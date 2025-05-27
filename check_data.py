import pandas as pd

# Load the tabular data file
tabular_data_file = "data/clinical_data.xlsx"
try:
    df = pd.read_excel(tabular_data_file)
    print(f"Successfully loaded {tabular_data_file}")

    # Print the column names and data types
    print("\nColumn names and data types:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")
        # Print a sample value
        print(f"    Sample value: {df[col].iloc[0]}")

    # Print the first few rows
    print("\nFirst 5 rows:")
    print(df.head())

    # Check columns we're planning to use
    columns_to_use = ['ZGON wewnątrzszpitalnie', 'PLEC', 'male sex', 'WIEK', 'KG']

    print("\nChecking columns we plan to use:")
    for col in columns_to_use:
        if col in df.columns:
            print(f"  - {col}: Found, dtype: {df[col].dtype}")
            # Print unique values for categorical columns
            if df[col].dtype == 'object' or df[col].nunique() < 10:
                print(f"    Unique values: {df[col].unique()}")
        else:
            print(f"  - {col}: Missing")

except Exception as e:
    print(f"Error loading {tabular_data_file}: {e}")
