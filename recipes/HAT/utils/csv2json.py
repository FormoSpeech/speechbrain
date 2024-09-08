import csv
import json

# Function to convert CSV data to desired JSON format
def convert_csv_to_json(csv_file_path, json_file_path):
    json_data = []
    
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        
        for row in reader:
            # Construct the required JSON structure
            json_object = {
                "id": row["ID"],
                "stem": row["stem"],
                "speaker": row["spk_id"],
                "channel": row["channel"],
                "audio_path": row["wav_path"],
                "duration": float(row["duration"]),
                "text": row["text"],
            }
            
            # Append to the list of JSON objects
            json_data.append(json_object)
    
    # Write the JSON data to a file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        for item in json_data:
            json.dump(item, json_file, ensure_ascii=False)
            json_file.write('\n')
    print(f"File saved at {json_file_path}!")

if __name__ == "__main__":
    
    path = [
        ('/mnt1/user_forbes/datasets/tat_asr_channel/train/metadata.csv', '/mnt1/user_forbes/datasets/tat_asr_channel/train/metadata.json'),
        ('/mnt1/user_forbes/datasets/tat_asr_channel/valid/metadata.csv', '/mnt1/user_forbes/datasets/tat_asr_channel/valid/metadata.json'),
        ('/mnt1/user_forbes/datasets/tat_asr_channel/test/metadata.csv', '/mnt1/user_forbes/datasets/tat_asr_channel/test/metadata.json')
    ]
    
    for csv_file_path, json_file_path in path:
        convert_csv_to_json(csv_file_path, json_file_path)
