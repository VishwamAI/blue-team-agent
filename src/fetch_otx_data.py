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
            # Fetch the latest threat intelligence data with a timeout
            pulses = self.otx.getall(timeout=30)
            print("Data fetched successfully.")
            return pulses
        except requests.exceptions.Timeout:
            print("Error: Request timed out.")
            return None
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
        print("Starting data fetch process...")
        data = self.fetch_data()
        if data:
            print("Preprocessing data...")
            preprocessed_data = self.preprocess_data(data)
            print("Saving preprocessed data...")
            self.save_preprocessed_data(preprocessed_data)
            print("Data fetch and preprocessing completed successfully.")
        else:
            print("No data fetched. Exiting.")

if __name__ == "__main__":
    # Use the provided API key
    api_key = "20605e2bc55efa3fbcc3251c289a8002553a3e7766ac92fbfe60b3053b2025e1"
    output_file = "otx_data.json"

    # Create an instance of OTXDataFetcher and run the data fetching process
    fetcher = OTXDataFetcher(api_key, output_file)
    fetcher.run()
