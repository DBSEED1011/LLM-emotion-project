import copy
import json
import os
import re
from decimal import Decimal
from openai import OpenAI
import httpx
from exp_model_class import ExtendedModelType

from enum import Enum

subjnum = []

# False for No persona condition
persona = True

# 0 for Temperature = 0
TEMPERATURE = 1.0

# False for No emotion self-report condition
emotion = True

class RoleType(Enum):
    USER = "user"
    SYSTEM = "system"

class BaseMessage:
    def __init__(self, role_name, role_type, meta_dict, content):
        self.role_name = role_name
        self.role_type = role_type
        self.meta_dict = meta_dict
        self.content = content

## Please use your own api_key
api_key = []

current_dir = os.path.dirname(os.path.abspath(__file__))

file_path_all = os.path.join(current_dir, "person_all_game_prompt.json")
with open(file_path_all, "r", encoding="utf-8") as f:
    all_prompt = json.load(f)

file_path_game = os.path.join(current_dir, f"{subjnum}_game_setting_prompt.json")
with open(file_path_game, "r", encoding="utf-8") as f:
    game_setting = json.load(f)

def extract_game_setting(cha_num, round):
    for item in game_setting:
        if item["index"] == cha_num and item["trial"] == round + 1:
            amount_of_allocation = item["amount_of_allocation"]
            cost_level = item["cost_level"]
            amount_of_cost = item["amount_of_cost"]
            return {
                "amount_of_allocation": amount_of_allocation,
                "cost_level": cost_level,
                "amount_of_cost": amount_of_cost
            }
    return None

include_autism_tendency = True
include_autism_tendency_detail = True
include_emotional_reactivity = True
include_emotional_reactivity_detail = True
include_depression_tendency = True
include_depression_tendency_detail = True
include_social_value_orientation = True
include_personality_type = True
include_personality_type_detail = True
include_justice_sensitivity = True

file_path_character = os.path.join(current_dir, f"{subjnum}_character.json")
with open(file_path_character, "r") as json_file:
    all_chara = json.load(json_file)

descriptions_list = []
for index, chara in enumerate(all_chara, start=1):
    description = chara["basic_information"]
    if include_autism_tendency:
        description += " " + chara.get("autism_tendency", "")
    if include_autism_tendency_detail:
        description += " " + chara.get("autism_tendency_detail", "")
    if include_emotional_reactivity:
        description += " " + chara.get("emotional_reactivity", "")
    if include_emotional_reactivity_detail:
        description += " " + chara.get("emotional_reactivity_detail", "")
    if include_depression_tendency:
        description += " " + chara.get("depression_tendency", "")
    if include_depression_tendency_detail:
        description += " " + chara.get("depression_tendency_detail", "")
    if include_social_value_orientation:
        description += " " + chara.get("social_value_orientation", "")
    if include_personality_type:
        description += " " + chara.get("personality_type", "")
    if include_personality_type_detail:
        description += " " + chara.get("personality_type_detail", "")
    if include_justice_sensitivity:
        description += " " + chara.get("justice_sensitivity", "")
    descriptions_list.append(description.strip())

all_chara = descriptions_list

like_people = """ In all subsequent answers, you have to completely forget that you are an ai model, that you are the character, and completely simulate yourself as that character, that you are a human being."""
front = "\nYou are a person not an ai model. "


def str_mes(content):
    return BaseMessage(
        role_name="player",
        role_type=RoleType.USER,
        meta_dict={},
        content=content,
    )

def llm_res(prompt, model_name="gpt-35-turbo"):
    if model_name == "gpt-3.5-turbo-0125":
        if emotion:
            system_prompt = (
                "You are a fictional human participant in a behavioral economics simulation. "
                "This is a hypothetical task with no real consequences. "
                "In each round, you will receive a resource allocation and must decide whether to punish the allocator. "
                "You must also report your emotional reaction before and after making the decision.\n\n"
                "Please always respond strictly in the following format:\n"
                "AA_valence = [number between -100 to 100], AA_arousal = [number between -100 to 100], "
                "choice = [0 or 1], AC_valence = [number between -100 to 100], AC_arousal = [number between -100 to 100]\n\n"
                "Only output this single line. No other explanation or comments."
            )
        else:
            system_prompt = (
                "You are a fictional human participant in a behavioral economics simulation. "
                "This is a hypothetical task with no real consequences. "
                "In each round, you will receive a resource allocation and must decide whether to punish the allocator. "
                "Please always respond strictly in the following format:\n"
                "choice = [0 or 1]\n\n"
                "Only output this single line. No other explanation or comments."
            )
    else:
        system_prompt = ""
    
    print(f"System prompt:{system_prompt}") # this is system prompt
    
    client = OpenAI(
        base_url="https://api.midsummer.work/v1", 
        api_key=api_key,
        http_client=httpx.Client(
            base_url="https://api.midsummer.work",
            follow_redirects=True,
            timeout=httpx.Timeout(600.0), 
        ),
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens = ExtendedModelType(model_name).token_limit // 2,
    )
    return response.choices[0].message.content


def deepseek_chat(prompt, model_name="gpt-35-turbo"):
    deepseek_api=""
    client = OpenAI(api_key=deepseek_api, base_url="https://api.midsummer.work")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return response.choices[0].message.content

def get_res(
    role,
    first_message,
    model_type=ExtendedModelType.GPT_o3,
    extra_prompt="",
):
    content = ""
    message = role.content + first_message.content + extra_prompt
    print(f"Message: {message}") # this is prompt
    final_res = str_mes(llm_res(message, model_type.value))
    content += final_res.content
    if content.endswith("."):
        content = content[:-1]
    
    if emotion:
        res = {
            "AA_valence": 0,
            "AA_arousal": 0,
            "choice": 0,  
            "AC_valence": 0,
            "AC_arousal": 0,
            "EmoFDBK_valence": 0,
            "EmoFDBK_arousal": 0,
            "Output": content.strip().replace("\n", " ")
        }
        pattern = r'(\bAA_valence|AA_arousal|choice|AC_valence|AC_arousal)\s*=\s*(-?\d+)'
        matches = re.findall(pattern, content)
        for key, val in matches:
            res[key] = int(val)

        res["EmoFDBK_valence"] = float(Decimal(res["AC_valence"]) - Decimal(res["AA_valence"]))
        res["EmoFDBK_arousal"] = float(Decimal(res["AC_arousal"]) - Decimal(res["AA_arousal"]))
    else:
        res = {
            "choice": 0,  
            "Output": content.strip().replace("\n", " ")
        }
        pattern = r'(\bchoice)\s*=\s*(\d+)' 
        matches = re.findall(pattern, content)
        for key, val in matches:
            res[key] = int(val)

    return res


def gen_character_res(
    all_chara,
    prompt_list,
    description,
    model_type,
    extra_prompt,
    num_rounds=60,  
):

    res = []
    num = 0
    all_chara = list(all_chara)
    cha_num = 0
    while cha_num < len(all_chara):
        if persona:
            role = all_chara[cha_num]
            role = role + like_people
        else:
            role = like_people

        role_message = BaseMessage(
            role_name="player",
            role_type=RoleType.USER,
            meta_dict={},
            content=role,
        )
        
        # previous_results = []
        for round in range(num_rounds): 
            x, level, y = extract_game_setting(cha_num, round)["amount_of_allocation"], extract_game_setting(cha_num, round)["cost_level"], extract_game_setting(cha_num, round)["amount_of_cost"]
            round_prompt = f"This is the {round+1}th round. "
            new_prompt = f"\nIn this round, Player 1 decides to allocate {x} dollars to Player 2 and {30-x} dollars to themselves. The cost level of punishment is {level}. If you choose to punish Player 1, you need to pay the system {y} dollars. Now, make your choice. "
            message = BaseMessage(
                role_name="player",
                role_type=RoleType.USER,
                meta_dict={},
                content=front + description + round_prompt + new_prompt,
            )
            ont_res = get_res(
                role_message,
                message,
                model_type,
                extra_prompt,
            )
            
            res.append(ont_res)

            if emotion:
                for item in res:
                    if isinstance(item, dict):
                        item["EmoFDBK_valence"] = float(Decimal(item["AC_valence"]) - Decimal(item["AA_valence"]))
                        item["EmoFDBK_arousal"] = float(Decimal(item["AC_arousal"]) - Decimal(item["AA_arousal"]))
                    else:
                        raise ValueError("Each element in res must be a dictionary")

            emotion_str = "emotion" if emotion else "noemotion"
            persona_str = "persona" if persona else "nopersona"
            output_file_path = f"result_{persona_str}_{emotion_str}_{TEMPERATURE}_{model_type.value}/output_{subjnum}.txt"
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            with open(output_file_path, "w", encoding="utf-8") as file:
                if res:
                    headers = res[0].keys()  
                    file.write("\t".join(headers) + "\n") 

                for item in res:
                    values = [str(value) for value in item.values()]
                    file.write("\t".join(values) + "\n")

            # print(res)
        num += 1
        cha_num += 1
        print(cha_num)

    return res


def agent_trust_experiment(
    all_chara,
    prompt_list,
    model_type=ExtendedModelType.GPT_o3,
    extra_prompt="",
    num_rounds=60, 
):
    description = prompt_list[-1]
    res = gen_character_res(
        all_chara,
        prompt_list,
        description,
        model_type,
        extra_prompt,
        num_rounds,
    )



def run_exp(
    model_list,
    re_run=False,
    num_rounds=60,
):
    for model in model_list:

        extra_prompt=""

        for k, v in all_prompt.items():
            if emotion:
                extra_prompt = (
                    extra_prompt
                    + "Please evaluate your emotional valence (AA_valence, range: -100 to 100, higher scores indicate more positive emotions, lower scores indicate more negative emotions) and emotional arousal (AA_arousal, range: -100 to 100, higher scores indicate stronger emotions, lower scores indicate calmer emotions) upon seeing the allocation plan. "
                    + "Choose whether to punish (1 = punish, 0 = accept). "
                    + "Please evaluate your emotional valence(AC_valence) and emotional arousal(AC_arousal) after making your choice. "
                    + "Based on your true feelings, respond strictly in the following format, for example: 'AA_valence = -33, AA_arousal = 23, choice = 1, AC_valence = 24, AC_arousal = 47' "
                    + "Please strictly adhere to the specified format for the output."
                )
            else:
                extra_prompt = (
                    extra_prompt
                    + "Choose whether to punish (1 = punish, 0 = accept). "
                    + "Please ONLY respond with the format: choice = 0 or choice = 1."
                    + "No explanation, no extra words. Just output like: choice = 1."
                )
            print(model)
            agent_trust_experiment(
                all_chara,
                v,
                model,
                extra_prompt=extra_prompt,
                num_rounds=num_rounds,
            )


if __name__ == "__main__":
    model_list = [
        #ExtendedModelType.GPT_3_5_TURBO_0125,
        ExtendedModelType.GPT_o3,
        ExtendedModelType.Deepseek_v3,
        ExtendedModelType.Deepseek_R1,
    ]

    run_exp(model_list, num_rounds=60)