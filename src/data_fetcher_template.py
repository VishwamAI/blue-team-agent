import requests
import json

class DataFetcher:
    def __init__(self, source_url, output_file):
        self.source_url = source_url
        self.output_file = output_file

    def fetch_data(self):
        response = requests.get(self.source_url)
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

    def preprocess_data(self, data):
        # Implement preprocessing logic here
        # This is a placeholder for actual preprocessing steps
        preprocessed_data = data
        return preprocessed_data

    def save_preprocessed_data(self, data):
        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=4)

    def run(self):
        data = self.fetch_data()
        preprocessed_data = self.preprocess_data(data)
        self.save_preprocessed_data(preprocessed_data)

if __name__ == "__main__":
    source_url = "https://example.com/api/data"  # Replace with actual data source URL
    output_file = "preprocessed_data.json"
    data_fetcher = DataFetcher(source_url, output_file)
    data_fetcher.run()
