import pandas as pd

def adjust_repeated_ids(csv_path, output_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Check if 'ID' column exists
    if 'ID' not in df.columns:
        raise ValueError("The CSV file must contain an 'ID' column.")
    
    # Initialize a set to keep track of unique IDs
    unique_ids = set()
    
    # Adjust IDs to ensure all are unique
    for i in range(len(df)):
        original_id = df.at[i, 'ID']
        new_id = original_id
        
        # If the ID is already in the set, keep adding 100000000 until it's unique
        while new_id in unique_ids:
            new_id += 100000000
        
        # Update the ID in the DataFrame and add the new ID to the set
        df.at[i, 'ID'] = new_id
        unique_ids.add(new_id)
    
    # Save the modified DataFrame back to the CSV
    df.to_csv(output_path, index=False)
    print('dataset saved!')

# Example usage:
# adjust_repeated_ids('path_to_your_csv.csv')

if __name__ == "__main__":

    splits = ['train', 'valid', 'test']
    # splits = ['total']
    for split in splits:
        adjust_repeated_ids(f'/mnt1/user_forbes/datasets/tat_asr_channel/{split}/metadata.csv', f'/mnt1/user_forbes/datasets/tat_asr_channel/{split}/metadata.csv')