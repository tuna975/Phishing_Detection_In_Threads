import json

# Load original
with open('cleaned_threads_v2/top_150_for_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Check if it's a dict or list
if isinstance(data, dict):
    # Convert dict to list
    result = []
    for thread_id, thread in data.items():
        thread['label'] = 0
        thread['attack_type'] = 'legitimate'
        result.append(thread)
else:
    # Already a list
    result = data
    for thread in result:
        thread['label'] = 0
        thread['attack_type'] = 'legitimate'

# Save
with open('legitimate_threads_fixed.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"Fixed {len(result)} threads")