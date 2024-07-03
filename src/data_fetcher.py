import requests
import json

class DataFetcher:
    def __init__(self, source_url, output_file):
        self.source_url = source_url
        self.output_file = output_file

    def fetch_data(self):
        headers = {
            'Accept': 'application/json'
        }
        print("Fetching data from:", self.source_url)
        response = requests.get(self.source_url, headers=headers)
        if response.status_code == 200:
            print("Data fetched successfully.")
            return response.json()
        else:
            print("Failed to fetch data. Status code:", response.status_code)
            response.raise_for_status()

    def preprocess_data(self, data):
        # Implement preprocessing logic here
        # This is a placeholder for actual preprocessing steps
        print("Preprocessing data...")
        preprocessed_data = data
        print("Data preprocessing complete.")
        return preprocessed_data

    def save_preprocessed_data(self, data):
        print("Saving preprocessed data to:", self.output_file)
        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=4)
        print("Data saved successfully.")

    def run(self):
        print("Starting data fetcher...")
        data = self.fetch_data()
        preprocessed_data = self.preprocess_data(data)
        self.save_preprocessed_data(preprocessed_data)
        print("Data fetcher run complete.")

if __name__ == "__main__":
    source_url = "https://cve.circl.lu/api/cve"  # CIRCL CVE Search API endpoint
    output_file = "preprocessed_data.json"
    data_fetcher = DataFetcher(source_url, output_file)
    data_fetcher.run()
