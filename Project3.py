import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_excel('cleaningdata.xlsx') 

def handle_missing_data(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

def remove_duplicates(df):
    initial_count = df.shape[0]
    df.drop_duplicates(inplace=True)
    final_count = df.shape[0]
    print(f"Removed {initial_count - final_count} duplicate rows.")
    return df

def fix_data_types(df):
    if 'date_column' in df.columns: 
        df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    return df

def handle_outliers(df):
    if 'price' in df.columns:
        upper_limit = df['price'].quantile(0.95)
        df = df[df['price'] < upper_limit]
        print(f"Removed outliers from 'price' column.")
    
    return df

def standardize_and_normalize(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

# Apply the cleaning functions
df = handle_missing_data(df)
df = remove_duplicates(df)
df = fix_data_types(df)
df = handle_outliers(df)
df = standardize_and_normalize(df)

# Save the cleaned dataset to a new Excel file
df.to_excel('cleaned_dataset.xlsx', index=False)  # Save cleaned data to a new Excel file
print("Cleaned data saved to 'cleaned_dataset.xlsx'")