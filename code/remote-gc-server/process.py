import json
import sys
import os

if __name__ == "__main__":
    conf_base = sys.argv[1]
    conf_new = sys.argv[2]
    base_path = sys.argv[3]

    data = {}
    with open(conf_base) as f:
        data = json.load(f)

    data["core_assets"] = os.path.relpath(data["core_assets"], base_path)
    data["abs_file"] = os.path.relpath(data["abs_file"], base_path)

    with open(conf_new, "w") as f:
        json.dump(data, f)
