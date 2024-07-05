import json

def inspect_indicators_structure(file_path, num_lines=10):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            for i, item in enumerate(data):
                if i >= num_lines:
                    break
                indicators = item.get('indicators', [])
                print(f"Indicators in pulse {i + 1}:")
                for indicator in indicators:
                    print(json.dumps(indicator, indent=2))
    except Exception as e:
        print(f"Error reading data: {e}")

if __name__ == "__main__":
    file_path = "otx_data.json"
    inspect_indicators_structure(file_path)
