import json

# Load the JSON file
with open('Files/exercises_schema_v2_2025-09-22.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Function to recursively replace &nbsp; in the text fields
def remove_nbsp(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = remove_nbsp(value)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = remove_nbsp(obj[i])
    elif isinstance(obj, str):
        return obj.replace('&nbsp;', '')
    return obj

# Clean the data
cleaned_data = remove_nbsp(data)

# Write the cleaned data back to a new file
with open('Files/exercises_schema_v2_2025-09-22 copy.json', 'w', encoding='utf-8') as file:
    json.dump(cleaned_data, file, ensure_ascii=False, indent=4)

print("File cleaned successfully!")
