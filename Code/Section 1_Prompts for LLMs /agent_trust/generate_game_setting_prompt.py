import pandas as pd

subjnum = 208

file_path = r"C:/Users/Thinkpadx13/Desktop/agent-trust-multiple/agent-trust-raw/agent_trust/prompt/Emo&TPP data.xlsx"  
sheet_name = "Sheet1"    
# n = 20     
# df = pd.read_excel(file_path, sheet_name=sheet_name).head(n*60)
df = pd.read_excel(file_path, sheet_name=sheet_name)
start_row = 60 * subjnum  
end_row = 60 * (subjnum + 1) 
df = df.iloc[start_row:end_row] 
df = df[["id", "trial", "amount_of_allocation", "cost_level", "amount_of_cost"]]
df["index"] = [i // 60 for i in range(len(df))]
df = df[["index"] + [col for col in df.columns if col != "index"]]
json_data = df.to_json(orient="records", force_ascii=False)

with open(fr"C:/Users/Thinkpadx13/Desktop/agent-trust-multiple/agent-trust-raw/agent_trust/prompt/{subjnum}_game_setting_prompt.json", "w", encoding="utf-8") as f:
    f.write(json_data)
