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

    # The ranges are adviced by Chatgpt, you can change them in a more reasonable range if it is necessary.
    # !!! Don't forget to save a new version of cleaned data if you change the range （Just uncomment the line that saves to csv）!!!
    
    plausible_ranges = {
        "Pregnancies": (0, np.inf),
        "Glucose": (40, 600),
        "BloodPressure": (30, 150),
        "SkinThickness": (5, 80),
        "Insulin": (2, 900),
        "BMI": (10, 80),
        "DiabetesPedigreeFunction": (0, 3.0),
        "Age": (21, np.inf)
    }


    # Step 4: Remove rows outside plausible ranges
    for feature, (lower, upper) in plausible_ranges.items():
        df = df[(df[feature] >= lower) & (df[feature] <= upper)]
        

    # !!! Uncomment this line to save the cleaned dataset !!!
    # df.to_csv("diabetes_cleaned.csv", index=False)
    


    # Step 5: Data normalization
    
    # Set the features to be normalized （2 methods）
    
    ## Method_1: Set features that going to be normalized.
    features_to_normalize = ["Pregnancies", "Glucose", "BloodPressure", 
                           "SkinThickness", "Insulin", "BMI", 
                           "DiabetesPedigreeFunction", "Age"]
    
#     ## Method_2: Drop columns that not need to be normalized.
#     features_to_normalize = df.columns.drop(["subject_id", "Outcome"])
    
    # Normalization
    scaler = MinMaxScaler() 
    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])
    
    # Reset index after all cleaning
    df.reset_index(drop=True, inplace=True)

    return df