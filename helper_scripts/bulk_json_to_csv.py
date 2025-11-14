import os
import json
import csv

def batch_process_json_to_csv(json_folder, output_csv):
    # List to store all data from JSON files
    data = []
    
    # Loop through each file in the directory
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            json_file_path = os.path.join(json_folder, filename)
            
            # Read and load the JSON data
            with open(json_file_path, 'r') as json_file:
                try:
                    row = {}
                    
                    json_data = json.load(json_file)
                    row['filename'] = filename
                    row["CER"] = json_data["CER"]
                    row["WER"] = json_data["WER"]
                    row["prompt_token_count"] = json_data["prompt_token_count"]
                    row["completion_token_count"] = json_data["completion_token_count"]
                    row["total_tokens"] = json_data["total_tokens"]
                    row["num_gt_features"] = json_data["num_gt_features"]
                    row["num_ocr_features"] = json_data["num_ocr_features"]
                    row["num_matches"] = len(json_data["matches"])
                    row["num_of_extras"] = len(json_data["extras"])
                    row["match_percentage"] = len(json_data["matches"]) / json_data["num_gt_features"]
                    # Append the data to the list
                    data.append(row)
                except json.JSONDecodeError:
                    print(f"Error decoding {filename}, skipping this file.")
    
    # Get the keys for the CSV header (from the first file's keys)
    if data:
        headers = list(data[0].keys())
        
        # Write to CSV
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            
            # Write the header and rows
            writer.writeheader()
            for row in data:
                writer.writerow(row)
            
        print(f"CSV file has been written to {output_csv}")
    else:
        print("No data was processed. Check the JSON files.")

# Usage example:
json_folder = 'test_data/eval'  # Replace with the path to your JSON files
output_csv = 'combined_eval.csv'        # Output CSV file path

batch_process_json_to_csv(json_folder, output_csv)