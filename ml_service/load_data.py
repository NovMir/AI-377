import pandas as pd
import os
import numpy as np

# Define the paths to the data directories
raw_data_path = os.path.join('data', 'raw')
processed_data_path = os.path.join('data', 'processed')

# Create processed directory if it doesn't exist
os.makedirs(processed_data_path, exist_ok=True)

try:
    # Load the CSV files
    df_oakville = pd.read_csv(os.path.join(raw_data_path, 'OakvillePrice.csv'))
    df_guelph = pd.read_csv(os.path.join(raw_data_path, 'GuelphPrice.csv'))
    df_mississauga = pd.read_csv(os.path.join(raw_data_path, 'MissisaugaPrice.csv'))
    df_milton = pd.read_csv(os.path.join(raw_data_path, 'MiltonPrice.csv'))

    # Convert price column to float64 in all dataframes for consistency
    df_oakville['price'] = df_oakville['price'].astype('float64')
    df_guelph['price'] = df_guelph['price'].astype('float64')
    df_mississauga['price'] = df_mississauga['price'].astype('float64')
    df_milton['price'] = df_milton['price'].astype('float64')

    # Combine all dataframes
    combined_df = pd.concat([df_oakville, df_guelph, df_mississauga, df_milton], ignore_index=True)

    # Add city column using the mapping
    city_mapping = {
        0: len(df_oakville),
        1: len(df_guelph),
        2: len(df_mississauga),
        3: len(df_milton)
    }
    
    combined_df['city'] = np.where(combined_df.index < city_mapping[0], 'Oakville',
                                  np.where(combined_df.index < city_mapping[0] + city_mapping[1], 'Guelph',
                                           np.where(combined_df.index < city_mapping[0] + city_mapping[1] + city_mapping[2], 'Mississauga', 'Milton')))

    # Fill missing 'price' values with city mean
    for city in combined_df['city'].unique():
        city_mean = combined_df[combined_df['city'] == city]['price'].mean()
        combined_df.loc[(combined_df['city'] == city) & (combined_df['price'].isnull()), 'price'] = city_mean

    # Drop columns that are completely empty (all values are NaN)
    columns_to_drop = combined_df.columns[combined_df.isna().all()].tolist()
    if columns_to_drop:
        print(f"\nDropping completely empty columns: {columns_to_drop}")
        combined_df = combined_df.drop(columns=columns_to_drop)

    # Save the combined dataframe to CSV
    output_file = os.path.join(processed_data_path, 'combined_properties.csv')
    combined_df.to_csv(output_file, index=False)
    print(f"\nCombined dataframe saved to: {output_file}")

    # Display basic information about the combined dataframe
    print("\nCombined Dataframe Information:")
    print("="*50)
    print(f"Total number of rows: {len(combined_df)}")
    print(f"Total number of columns: {len(combined_df.columns)}")
    print("\nFirst 5 rows of the combined dataframe:")
    print(combined_df.head())
    
    print("\nMissing values in each column:")
    print(combined_df.isnull().sum())
    
    print("\nData types of columns:")
    print(combined_df.dtypes)

except FileNotFoundError as e:
    print(f"Error: One or more CSV files not found. {str(e)}")
except Exception as e:
    print(f"An error occurred: {str(e)}") 