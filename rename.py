import pandas as pd
import json
df = pd.read_excel(r"C:\Users\VeikkoHerola\Downloads\SKUt mallissa ja k toimittajassa Lejos.xlsx",sheet_name="Myydyimmät")
print(df.columns)
df["PRODUCT_NAME_FI"] = df["PRODUCT_NAME_FI"].str.replace(" ", "_", regex=False)
d = df.set_index(1111)[["PRODUCT_NAME_FI", "GTIN"]].apply(lambda row: row.tolist(), axis=1).to_dict()
import os

path = r"D:/tmp/for large_faissbase_3625"

folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


df_uudet=pd.read_excel("planogrammi.xlsx",sheet_name="Sheet2")
for folder in folders:
    if folder not in d:
        try:
            ean=df_uudet[df_uudet["Löytyy mallista mutta ei Keskon listassa"]==folder]["EAN"].tolist()[0]
            d[folder] = [folder, ean]
        except:
            print(folder)
print(".................")
for folder in folders:
    if folder not in d:
        print(folder)
with open("nimi.json", "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2, ensure_ascii=False)