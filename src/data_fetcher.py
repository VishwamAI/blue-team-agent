import requests
import json

class DataFetcher:
    def __init__(self, source_url):
        self.source_url = source_url

    def fetch_data(self):
        try:
            response = requests.get(self.source_url)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None

    def preprocess_data(self, data):
        # Implement preprocessing logic here
        # This is a placeholder implementation
        preprocessed_data = []
        for item in data:
            # Example: Extract relevant fields and convert to required format
            preprocessed_item = {
                "field1": item.get("field1"),
                "field2": item.get("field2"),
                # Add more fields as needed
            }
            preprocessed_data.append(preprocessed_item)
        return preprocessed_data

    def save_preprocessed_data(self, preprocessed_data, output_file):
        try:
            with open(output_file, 'w') as f:
                json.dump(preprocessed_data, f)
            print(f"Preprocessed data saved to {output_file}")
        except IOError as e:
            print(f"Error saving preprocessed data: {e}")

if __name__ == "__main__":
    # Example usage
    source_url = "https://example.com/api/data"
    output_file = "preprocessed_data.json"

    data_fetcher = DataFetcher(source_url)
    data = data_fetcher.fetch_data()
    if data:
        preprocessed_data = data_fetcher.preprocess_data(data)
        data_fetcher.save_preprocessed_data(preprocessed_data, output_file)
