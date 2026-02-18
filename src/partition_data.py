import pandas as pd
import os

def create_hospital_silos(csv_path, num_clients=3):
    """Partitions the dataset into N separate hospital CSVs."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Place the dataset in the data/ folder.")
        return

    # Load and clean
    df = pd.read_csv(csv_path)
    df.drop(['education'], axis=1, inplace=True, errors='ignore')
    df.dropna(inplace=True)

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate partition sizes
    total_rows = len(df)
    step = total_rows // num_clients

    for i in range(num_clients):
        # Slice the DataFrame
        start_idx = i * step
        # For the last hospital, take all remaining rows
        end_idx = (i + 1) * step if i < num_clients - 1 else total_rows
        
        partition = df.iloc[start_idx:end_idx]
        
        file_path = f'data/hospital_{i+1}.csv'
        partition.to_csv(file_path, index=False)
        print(f"Hospital {i+1} silo created with {len(partition)} records.")

if __name__ == "__main__":
    create_hospital_silos('data/framingham.csv')