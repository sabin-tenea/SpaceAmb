import json
from collections import Counter
import os

def inspect(json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    
    atoms = data['atoms']
    print(f"Total Atoms: {len(atoms)}")
    
    # Family counts
    family_counts = Counter(a['family'] for a in atoms)
    print("\nCounts per Family:")
    for fam, count in sorted(family_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {fam}: {count}")

    # Subtype counts per family
    print("\nSubtypes per Family:")
    for fam in sorted(family_counts.keys()):
        subs = [a.get('subtype', 'None') for a in atoms if a['family'] == fam]
        sub_counts = Counter(subs)
        print(f"\n  [{fam.upper()}]")
        for sub, scount in sorted(sub_counts.items()):
            print(f"    - {sub}: {scount}")

if __name__ == "__main__":
    inspect("data/raw/atoms.json")
