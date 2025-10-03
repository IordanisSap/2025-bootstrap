import json, glob, os, sys

# Usage: python merge_json.py [output_filename]
out_file = sys.argv[1] if len(sys.argv) > 1 else "combined.json"

# Collect all *.json in the current directory, excluding the output file
json_files = sorted(
    f for f in glob.glob("*.json")
    if os.path.basename(f) != os.path.basename(out_file)
)

objects = []
for path in json_files:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"{path} does not contain a JSON object at the top level.")
        objects.append(data)

with open(out_file, "w", encoding="utf-8") as out:
    json.dump(objects, out, ensure_ascii=False, indent=2)

print(f"Wrote {len(objects)} objects from {len(json_files)} files to {out_file}")