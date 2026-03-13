"""Spot-check: verify well-known real-world companies exist in the unified CSV."""
import pandas as pd

df = pd.read_csv("processing/data/startups_unified.csv")

KNOWN_YC = [
    "Stripe", "Airbnb", "Dropbox", "Coinbase", "Reddit",
    "Instacart", "DoorDash", "Twitch", "GitLab", "Notion",
]

KNOWN_CB = [
    "Google", "Facebook", "Amazon", "Twitter", "Uber",
    "Spotify", "Netflix", "Slack", "Shopify", "Tesla Motors",
]

def check(names, label):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    for name in names:
        match = df[df["name"].str.lower() == name.lower()]
        if len(match):
            r = match.iloc[0]
            print(f"  ✅ {name:<22} source={r['source']:<12} country={r['country']:<6} year={r['founded_year']}")
        else:
            # try partial match
            partial = df[df["name"].str.lower().str.contains(name.lower(), na=False)]
            if len(partial):
                r = partial.iloc[0]
                print(f"  ⚠️  {name:<22} partial match -> '{r['name']}' source={r['source']}")
            else:
                print(f"  ❌ {name:<22} NOT FOUND")

check(KNOWN_YC, "Well-Known YC Companies")
check(KNOWN_CB, "Well-Known Crunchbase/Tech Companies")

# Also show some suspicious rows
print(f"\n{'='*55}")
print("  Sample SUSPICIOUS rows (is_suspicious=True)")
print(f"{'='*55}")
sus = df[df["is_suspicious"] == True].head(10)
for _, r in sus.iterrows():
    print(f"  🔍 {r['name']:<25} desc_len={len(str(r['description'])):<5} year={r['founded_year']}  source={r['source']}")
