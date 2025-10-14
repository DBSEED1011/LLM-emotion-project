import pandas as pd
import json
import time
import httpx
from tqdm import tqdm
from openai import OpenAI

# --------------------------------------------
# 1. Global configuration
# --------------------------------------------
API_KEY = "YOUR_API_KEY_HERE"  # 🔒 Masked for security
BASE_URL = "https://api.midsummer.work"
MODEL_NAME = "claude-3-5-sonnet-20241022"
TEMPERATURE = 0
MAX_TOKENS = 2048
BATCH_SIZE = 30  # Max words per batch (Claude recommends ≤ 30)

# --------------------------------------------
# 2. LLM request wrapper (Claude)
# --------------------------------------------
def llm_res(prompt):
    client = OpenAI(
        base_url=f"{BASE_URL}/v1",
        api_key=API_KEY,
        http_client=httpx.Client(
            base_url=BASE_URL,
            follow_redirects=True,
            timeout=httpx.Timeout(600.0),
        ),
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content

# --------------------------------------------
# 3. Construct prompt + call Claude for classification
# --------------------------------------------
def classify_words_with_claude(words):
    # Seed lists
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

    # Convert word list to markdown-style list
    word_lines = "\n".join(f"- {w}" for w in words)

    prompt = f"""
You are an expert annotator helping to classify words from a decision-making corpus.

You are provided with 3 seed lists of words:

**Emotion**: {emotion_seed}
**Fairness**: {fairness_seed}
**Cost**: {cost_seed}

Your task is to classify each word below into one of four categories: "emotion", "fairness", "cost", or "other".

Return a JSON array. Each item should follow this format:
{{
  "word": "example",
  "category": "emotion",
  "confidence": 0.92,
  "rationale": "It reflects a feeling or emotional state"
}}

Now classify these words:
{word_lines}
"""

    try:
        result_text = llm_res(prompt)
        json_start = result_text.find("[")
        parsed = json.loads(result_text[json_start:])
        return parsed
    except Exception as e:
        print("❌ Error:", e)
        with open("error_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        return []

# --------------------------------------------
# 4. Main function: read CSV, classify, and save
# --------------------------------------------
def classify_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    all_words = df["word"].tolist()
    results = []

    for i in tqdm(range(0, len(all_words), BATCH_SIZE)):
        batch = all_words[i:i + BATCH_SIZE]
        print(f"→ Processing batch {i+1} to {i+len(batch)}...")
        batch_result = classify_words_with_claude(batch)
        results.extend(batch_result)
        time.sleep(1)

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"\n✅ Done! Result saved to {output_csv}")

# --------------------------------------------
# 5. Entry point
# --------------------------------------------
if __name__ == "__main__":
    classify_csv("Human_Top180_Words.csv", "Top180_Annotated_Claude.csv")
