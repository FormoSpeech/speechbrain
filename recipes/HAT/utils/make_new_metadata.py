import pandas as pd

# List of specific speaker IDs
specific_spk_ids = [
    "XM006", "XM007", "XM008", "XM009", "XM010", "XM011", "XM012", "XM013", "XM014", "XM015",
    "XM016", "XM017", "XM038", "XM021", "XM022", "XM023", "XM025", "XM027", "XM028", "XM029",
    "XM030", "XM031", "XM032", "XM033", "XF005", "XF006","XF007", "XF008", "XF010", "XF011",
    "XF012", "XF013", "XF014", "XF015", "XF016", "XF018","XF019", "XF020", "XF021", "XF022",
    "XF023", "XF024", "XF025", "XF027", "XF028", "XF029","XF030", "XF031"
]

# List of files to process
files = [
    "_metadata_android.csv", "_metadata_H8y.csv", "_metadata_no_condenser.csv", "_metadata_no_lavalier.csv", "_metadata_sample.csv",
    "_metadata_condenser.csv", "_metadata_iOS.csv", "_metadata_no_H8x.csv", "_metadata_no_PCmic.csv", "_metadata_webcam.csv",
    "_metadata.csv", "_metadata_lavalier.csv", "_metadata_no_H8y.csv", "_metadata_no_webcam.csv", "_metadata_H8x.csv",
    "_metadata_no_android.csv", "_metadata_no_iOS.csv", "_metadata_PCmic.csv"
]

# Process each file
for file in files:
    df = pd.read_csv(f"/mnt/user_forbes/datasets/sixian_reading/train/{file}")
    filtered_df = df[df['spk_id'].isin(specific_spk_ids)]
    new_file_name = file[1:]  # Remove the leading underscore
    filtered_df.to_csv(f"/mnt/user_forbes/datasets/sixian_reading/train/{new_file_name}", index=False)
    print(f"Processed and saved {new_file_name}")
