import requests
import json

class DataFetcher:
    def __init__(self, source_url):
        """
        Initialize the DataFetcher with the source URL.

        :param source_url: URL of the data source to fetch data from.
        """
        self.source_url = source_url

    def fetch_data(self):
        """
        Fetch data from the source URL.

        :return: JSON data fetched from the source URL, or None if an error occurs.
        """
        try:
            response = requests.get(self.source_url)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None

    def preprocess_data(self, data):
        """
        Preprocess the fetched data.

        This method should be customized to implement specific preprocessing logic
        based on the structure and requirements of the data.

        :param data: Raw data fetched from the source URL.
        :return: Preprocessed data ready for training the model.
        """
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
        """
        Save the preprocessed data to a JSON file.

        :param preprocessed_data: Data that has been preprocessed and is ready for training.
        :param output_file: Path to the file where the preprocessed data will be saved.
        """
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
