import json
import os

path = "outputs/saes/multipartite_003_bsae/metrics_summary.json"
if not os.path.exists(path):
    print(f"File not found: {path}")
else:
    with open(path, "r") as f:
        data = json.load(f)
    
    print("Top level keys:", list(data.keys()))
    if not data:
        exit()
        
    first_site = list(data.keys())[0]
    print(f"Keys for {first_site}:", list(data[first_site].keys()))
    
    if "sequence" in data[first_site]:
        print(f"Sequence keys for {first_site}:", list(data[first_site]["sequence"].keys()))
        # Check for 'banded' or similar
        for key in data[first_site]["sequence"].keys():
            print(f"  Structure under {key}:")
            sub_dict = data[first_site]["sequence"][key]
            if sub_dict:
                sample_key = next(iter(sub_dict.keys()))
                print(f"    Sample key example: {sample_key}")
                sample_entry = sub_dict[sample_key]
                print(f"    Sample entry keys: {list(sample_entry.keys())}")
