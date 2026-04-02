import pandas as pd
import os

def analyze(csv_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # We'll check the top 10 atoms for some specific "combined" queries
    # Query IDs are usually q_amb_{ambiance}_prog_{program}
    queries = [
        "q_amb_relaxing_prog_living_room",
        "q_amb_spooky_prog_dance_hall",
        "q_amb_somber_prog_conference_room",
        "q_amb_sterile_prog_hospital_room",
        "q_amb_melancholic_prog_library"
    ]
    
    for qid in queries:
        subset = df[df['query_id'] == qid]
        if subset.empty:
            # Fallback if names are different
            print(f"\nNo data for {qid}")
            continue
            
        print(f"\n=== Top 10 Atoms for {qid} ===")
        top10 = subset.sort_values('weighted_score', ascending=False).head(10)
        print(top10[['text', 'family', 'weighted_score']].to_string(index=False))

if __name__ == "__main__":
    analyze("data/processed/atom_scores.csv")
