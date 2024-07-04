import requests
from OTXv2 import OTXv2
import pandas as pd
import os
import json

class OTXDataFetcher:
    def __init__(self, api_key, output_file):
        self.api_key = api_key
        self.output_file = output_file
        self.otx = OTXv2(api_key)

    def fetch_data(self):
        try:
            # Fetch the latest threat intelligence data
            pulses = self.otx.getall()
            print("Data fetched successfully.")
            return pulses
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def preprocess_data(self, data):
        # Implement preprocessing logic here
        preprocessed_data = []
        for pulse in data:
            preprocessed_data.append({
                'name': pulse['name'],
                'description': pulse['description'],
                'tags': pulse['tags'],
                'indicators': pulse['indicators']
            })
        return preprocessed_data

    def save_preprocessed_data(self, data):
        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=4)

    def run(self):
        data = self.fetch_data()
        if data:
            preprocessed_data = self.preprocess_data(data)
            self.save_preprocessed_data(preprocessed_data)

if __name__ == "__main__":
    # Retrieve the API key from the environment variable
    api_key = os.getenv("Hugging_Face_Hugging_Face")
    output_file = "otx_data.json"

    # Create an instance of OTXDataFetcher and run the data fetching process
    fetcher = OTXDataFetcher(api_key, output_file)
    fetcher.run()
