import json

def inspect_otx_data(file_path, num_lines=10):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            for i, item in enumerate(data):
                if i >= num_lines:
                    break
                print(json.dumps(item, indent=2))
    except Exception as e:
        print(f"Error reading data: {e}")

if __name__ == "__main__":
    file_path = "otx_data.json"
    inspect_otx_data(file_path)
