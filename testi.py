# Data on listassa yhdessä objektissa, otetaan se ulos
import json
import pandas as pd
with open("tiedosto.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Oletetaan että data on listassa 1. objektin alla
#data = raw_data[0]
data=[]
count=0
for i in raw_data:
    for j in i:
        j["bounding_box"]["left"]+=count
    count+=1

data = [item for sublist in raw_data for item in sublist]
# Poimitaan tag_name ja bounding_box.top, left
entries = []
print(data)
count=0
for item in data:

    tag = item.get("tag_name")
    box = item.get("bounding_box", {})
    top = box.get("top")
    left = box.get("left")
    height=box.get("height")
    if tag!="hintalappu":
        continue
    entries.append({"tag": tag+str(count), "top": top, "left": left,"height":height})
    count+=1
df = pd.DataFrame(entries)

# Ryhmitetään tuotteet riveihin top-arvon mukaan (klusteroidaan yksinkertaisesti)
df = df.sort_values(by=["top", "left"]).reset_index(drop=True)

# Luodaan likimääräinen rivi-indeksi: erotetaan top-arvon mukaan
row_threshold = 0.1  # Riittävä ero rivien välillä
rows = []
current_top = None
row_idx = -1
print(df)
for _, row in df.iterrows():
    if current_top is None or abs(row["top"] - current_top) > 2*row["height"]:
        row_idx += 1
        current_top = row["top"]
    rows.append(row_idx)

df["shelf_row"] = rows

# Järjestetään jokainen rivi vasemmalta oikealle (left-arvon mukaan)
df = df.sort_values(by=["shelf_row", "left"]).reset_index(drop=True)

# Lisätään paikka rivillä
df["position_on_row"] = df.groupby("shelf_row").cumcount() + 1

# Valitaan lopulliset sarakkeet
final_df = df[["shelf_row", "position_on_row", "tag"]]

# Tallennetaan sekä Excel- että CSV-muotoon
excel_path = "planogrammi.xlsx"
csv_path = "planogrammi.csv"

final_df.to_excel(excel_path, index=False)
final_df.to_csv(csv_path, index=False)

excel_path, csv_path