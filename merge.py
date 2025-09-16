import json

def merge_json_files(file_paths):
    merged_data = []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            merged_data.append(data)
    
    return merged_data

# List of file paths
file_paths = ["Files/7th_grade_lesson_9_מערכת_צירים.json", "Files/8th_grade_lesson_1_מערכת_צירים ומשוואות (חזרה).json",
               "Files/8th_grade_lesson_2_הצגה_של_ישר_.json", "Files/8th_grade_lesson_3_משוואת_ישר_ושיפוע.json"]

# Merge the data
merged_data = merge_json_files(file_paths)

# Print merged data or write to a new file
with open('merged_output.json', 'w', encoding='utf-8') as output_file:
    json.dump(merged_data, output_file, ensure_ascii=False, indent=4)

print("Merge completed successfully!")
