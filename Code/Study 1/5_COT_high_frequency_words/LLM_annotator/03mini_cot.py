import pandas as pd
from tqdm import tqdm
import time
import httpx
from openai import OpenAI

# ---------------------
# 1. 参数设置
# ---------------------
api_key = "sk-LRQN9RYuOhPDM7xPM770lRMdrQlC2w2p8CWFiGslCZWg5aXL"
TEMPERATURE = 0
BASE_URL = "https://api.midsummer.work"

# ---------------------
# 2. 模型调用函数（适配 midsummer API）
# ---------------------
def llm_res(prompt, model_name="o3-mini"):
    client = OpenAI(
        base_url=f"{BASE_URL}/v1",
        api_key=api_key,
        http_client=httpx.Client(
            base_url=BASE_URL,
            follow_redirects=True,
            timeout=httpx.Timeout(600.0),
        ),
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=30000,
    )
    return response.choices[0].message.content

# ---------------------
# 3. 构造 Prompt 并调用模型
# ---------------------
def classify_words_with_llm(words):
    emotion_seed = [
        "happy", "joy", "satisfied", "confident", "excited", "grateful", "hopeful",
        "comfort", "trust", "relief", "pleasure", "optimism", "compassion",
        "angry", "sad", "fear", "worried", "anxious", "disappointed", "frustrated",
        "stressed", "upset", "pain", "grief", "guilt", "discomfort", "disgust",
        "surprise", "anticipation", "curiosity", "uncertainty", "shock", "hesitation", "emotional", "react", "triggered"
    ]

    fairness_seed = [
        "fair", "fairness", "just", "justice", "equitable", "equality", "equal", "balance",
        "impartial", "unbiased", "bias", "biased", "discrimination", "prejudice", "stereotyping",
        "unfair", "unjust", "favoritism", "exclusion", "marginalize",
        "inclusive", "inclusiveness", "diversity", "accessibility", "representation",
        "underrepresented", "minority", "fairness-aware"
    ]

    cost_seed = [
        "cost", "costly", "expensive", "expense", "price", "overhead", "compute", "resource",
        "memory", "bandwidth", "energy", "efficiency", "time-consuming", "effort", "workload",
        "labor", "manual", "maintenance", "delay", "slow", "latency", "scalability", "risk",
        "trade-off", "compromise", "burden", "consequence", "penalty", "failure", "damage",
        "negative impact", "unintended effect"
    ]

    prompt = f"""
You are an expert annotator helping to classify words from a decision-making corpus.
You are provided with 3 expert-curated seed lists of words belonging to three semantic categories:

- Emotion words: {emotion_seed}
- Fairness words: {fairness_seed}
- Cost words: {cost_seed}

Your task is to classify each given word into one of the following categories: "emotion", "fairness", "cost", or "other".

- Use these seed words as a guide to infer semantic similarity.
- If a word is not exactly in the list but clearly represents a concept semantically aligned with that category, classify it accordingly.
- Provide a JSON list of responses, each object with: word, category, confidence (0 to 1), and rationale.

Now classify the following words:
{words}
"""

    try:
        result_text = llm_res(prompt)
        parsed = eval(result_text)
        return parsed
    except Exception as e:
        print("❌ Error:", e)
        return []

# ---------------------
# 4. 读取 CSV 并批量执行
# ---------------------
def classify_csv(input_csv, output_csv, batch_size=50):
    df = pd.read_csv(input_csv)
    all_words = df["word"].tolist()
    results = []

    for i in tqdm(range(0, len(all_words), batch_size)):
        batch = all_words[i:i + batch_size]
        print(f"→ Processing batch {i+1} to {i+len(batch)}...")
        batch_result = classify_words_with_llm(batch)
        results.extend(batch_result)
        time.sleep(1)

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"\n✅ Done! Result saved to {output_csv}")

# ---------------------
# 5. 启动入口
# ---------------------
if __name__ == "__main__":
    classify_csv("301_350.csv", "301_350_Annotated_Midsummer.csv")
