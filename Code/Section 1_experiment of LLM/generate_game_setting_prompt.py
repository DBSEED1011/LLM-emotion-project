import os
import pandas as pd

subjnum = []

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "Emo&TPP data.xlsx"  )
sheet_name = "Sheet1"    

df = pd.read_excel(file_path, sheet_name=sheet_name)
start_row = 60 * subjnum  
end_row = 60 * (subjnum + 1) 
df = df.iloc[start_row:end_row] 
df = df[["id", "trial", "amount_of_allocation", "cost_level", "amount_of_cost"]]
df["index"] = [i // 60 for i in range(len(df))]
df = df[["index"] + [col for col in df.columns if col != "index"]]
json_data = df.to_json(orient="records", force_ascii=False)

output_file = os.path.join(current_dir, f"{subjnum}_game_setting_prompt.json")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(json_data)
