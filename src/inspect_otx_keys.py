import json

def inspect_otx_keys(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if len(data) > 0:
                first_pulse = data[0]
                print("Keys in the first pulse:")
                for key in first_pulse.keys():
                    print(key)
            else:
                print("No data found in the file.")
    except Exception as e:
        print(f"Error reading data: {e}")

if __name__ == "__main__":
    file_path = "otx_data.json"
    inspect_otx_keys(file_path)
