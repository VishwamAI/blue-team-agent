import requests
import json
import time

class DataFetcher:
    def __init__(self, source_url, output_file):
        self.source_url = source_url
        self.output_file = output_file

    def fetch_data(self):
        headers = {
            'Accept': 'application/json'
        }
        print("Fetching data from:", self.source_url)
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.get(self.source_url, headers=headers, timeout=10)
                response.raise_for_status()
                if response.status_code == 200:
                    print("Data fetched successfully.")
                    return response.json()
                else:
                    print("Failed to fetch data. Status code:", response.status_code)
            except requests.exceptions.RequestException as e:
                print(f"An error occurred while fetching data (attempt {attempt + 1}/{retries}):", e)
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Unable to fetch data.")
                    return None

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
        if data:
            preprocessed_data = self.preprocess_data(data)
            self.save_preprocessed_data(preprocessed_data)
        print("Data fetcher run complete.")

if __name__ == "__main__":
    source_url = "https://services.nvd.nist.gov/rest/json/cves/2.0?resultsPerPage=10"  # NVD API endpoint for the latest 10 CVEs
    output_file = "preprocessed_data.json"
    data_fetcher = DataFetcher(source_url, output_file)
    data_fetcher.run()
