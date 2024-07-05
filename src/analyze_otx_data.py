import json

def analyze_otx_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        print(f"Total number of pulses: {len(data)}")

        # Print a summary of the first pulse
        if len(data) > 0:
            first_pulse = data[0]
            print("\nSummary of the first pulse:")
            print(f"Name: {first_pulse.get('name', 'N/A')}")
            print(f"Description: {first_pulse.get('description', 'N/A')}")
            print(f"Tags: {first_pulse.get('tags', 'N/A')}")
            print(f"Number of indicators: {len(first_pulse.get('indicators', []))}")

        # Print the keys of the first pulse
        if len(data) > 0:
            print("\nKeys in the first pulse:")
            for key in data[0].keys():
                print(key)

    except Exception as e:
        print(f"Error analyzing data: {e}")

if __name__ == "__main__":
    file_path = "otx_data.json"
    analyze_otx_data(file_path)
