# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_and_clean_data(csv_path):
    
    """
    Load, clean, and normalize the Pima Diabetes dataset.

    Steps performed:
    1. Load CSV into pandas DataFrame.
    2. Insert 'subject_id' column for traceability.
    3. Set plausible ranges of each variable, to remove physiologically impossible or extreme values.
    4. Remove samples that contain values outside plausible ranges.
    5. Normalize all feature columns using Min-Max scaling.

    Input:
    csv_path (str): Path to the CSV dataset.

    Returns:
    pd.DataFrame: Cleaned and normalized dataset with 'subject_id'.
    """
    
    
    # Step 1: Load data
    df = pd.read_csv(csv_path)

    
    # Step 2: Insert subject_id for traceability
    df.insert(0, 'subject_id', df.index)

    
    # Step 3: Set plausible ranges 

    # The ranges are based on references listed, and expanded for pathological values in diabetic patients.
    # You can change them in a more reasonable range if it is necessary. But don't forget to save a new version of cleaned data if you change the range（Just uncomment the line that saves to csv）
    
    plausible_ranges = {
        "Pregnancies": (0, np.inf),
        "Glucose": (40, 600),                # Mayo Clinic Laboratories Critical Values / Critical Results List, https://www.ncbi.nlm.nih.gov/books/NBK555976/
        "BloodPressure": (30, 120),          # https://www.nhlbi.nih.gov/health/low-blood-pressure, https://www.nhlbi.nih.gov/health/high-blood-pressure
        "SkinThickness": (5, 60),            # Anthropometric Reference Data for Children and Adults: United States,2007–2010
        "Insulin": (0, 600),                 # Roche Diagnostics. (2023). Elecsys Insulin: Method Sheet (V 4.0). 
        "BMI": (16, 70),                     # https://www.who.int/data/nutrition/nlis/info/malnutrition-in-women，Corpodean F, Kachmar M, Popiv I, LaPenna KB, Lenhart D, Cook M, Albaugh VL, Schauer PR. BMI ≥ 70: A Multi-Center Institutional Experience of the Safety and Efficacy of Metabolic and Bariatric Surgery Intervention. Obes Surg. 2024 Sep;34(9):3165-3172. doi: 10.1007/s11695-024-07419-7. Epub 2024 Jul 24. PMID: 39046626.
        "DiabetesPedigreeFunction": (0, np.inf),   # According to the definition of Diabetes Pedigree FunctionUsing in the paper ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus
        "Age": (21, np.inf)
    }


    # Step 4: Remove rows outside plausible ranges
    for feature, (lower, upper) in plausible_ranges.items():
        df = df[(df[feature] >= lower) & (df[feature] <= upper)]
       
    return df


if __name__ == "__main__":
    import os
    raw_data_path = "data/diabetes.csv"

    if os.path.exists(raw_data_path):
        print("Loading data from ", raw_data_path)
        df_clean = load_and_clean_data(raw_data_path)
        df_clean.to_csv("data/diabetes_cleaned.csv", index=False)
        print("Data cleaned successfully, saved to data/diabetes_cleaned.csv")
    else:
        print(f"File {raw_data_path} does not exist.")
