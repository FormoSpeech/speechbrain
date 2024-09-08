import csv
import json

def convert_json_to_csv(json_file_path, csv_file_path):
    # Read the JSON data from the file
    json_data = []
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        for line in json_file:
            json_data.append(json.loads(line))

    # Open the CSV file for writing
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["ID", "stem", "spk_id", "spk_gender", "channel", "text", "dialect", "duration", "wav_path"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write the header row
        writer.writeheader()

        for item in json_data:
            # Derive spk_gender from the second letter of the speaker
            spk_gender = 'female' if item["speaker"][1] == 'F' else 'male'
            
            # Construct the CSV row
            csv_row = {
                "ID": item["id"],
                "stem": item["stem"],
                "spk_id": item["speaker"],
                "spk_gender": spk_gender,
                "channel": item["channel"],
                "text": item["text"],
                "dialect": item["dialect"],
                "duration": item["duration"],
                "wav_path": item["audio_path"]
            }
            
            # Write the row to the CSV file
            writer.writerow(csv_row)
    print(f"File saved at {csv_file_path}.")

# Example usage
json_file_path = 'your_json_file_path.json'
csv_file_path = 'your_csv_file_path.csv'
convert_json_to_csv(json_file_path, csv_file_path)
