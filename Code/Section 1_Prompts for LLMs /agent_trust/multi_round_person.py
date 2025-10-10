import copy
import json
import os
import re
from decimal import Decimal

from openai import OpenAI
import httpx
from exp_model_class import ExtendedModelType

# from camel.messages import BaseMessage
# from camel.types.enums import RoleType
from enum import Enum

subjnum = 208

class RoleType(Enum):
    USER = "user"
    SYSTEM = "system"

class BaseMessage:
    def __init__(self, role_name, role_type, meta_dict, content):
        self.role_name = role_name
        self.role_type = role_type
        self.meta_dict = meta_dict
        self.content = content
api_key = "sk-LRQN9RYuOhPDM7xPM770lRMdrQlC2w2p8CWFiGslCZWg5aXL"

TEMPERATURE = 0.0
with open(
    r"agent-trust-raw/agent_trust/prompt/person_all_game_prompt.json",
    "r",
) as f:
    all_prompt = json.load(f)

with open(
    r"agent-trust-raw/agent_trust/prompt/person_all_game_prompt.json",
    "r",
) as f:
    all_prompt_copy = json.load(f)

with open(
    f"agent-trust-raw/agent_trust/prompt/{subjnum}_game_setting_prompt.json",
    "r",
) as f:
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

with open(f"agent-trust-raw/agent_trust/prompt/{subjnum}_character.json", "r") as json_file:
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
# print(all_chara)


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
    # system_prompt = (
    #     "You are a fictional human participant in a behavioral economics simulation. "
    #     "This is a hypothetical task with no real consequences. "
    #     "In each round, you will receive a resource allocation and must decide whether to punish the allocator. "
    #     "You must also report your emotional reaction before and after making the decision.\n\n"
    #     "Please always respond strictly in the following format:\n"
    #     "AA_valence = [number between -100 to 100], AA_arousal = [number between -100 to 100], "
    #     "choice = [0 or 1], AC_valence = [number between -100 to 100], AC_arousal = [number between -100 to 100]\n\n"
    #     "Only output this single line. No other explanation or comments."
    # )

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
            {"role": "system", "content": ""}, #system_prompt
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=16384,
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
    model_type=ExtendedModelType.Deepseek_R1,
    extra_prompt="",
):
    content = ""
    message = role.content + first_message.content + extra_prompt
    # print(f"Message: {message}") # this is prompt
    final_res = str_mes(llm_res(message, model_type.value))
    content += final_res.content
    # print(content)
    if content.endswith("."):
        content = content[:-1]

    res = {
        "AA_valence": 0,
        "AA_arousal": 0,
        "choice": 2,  
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

    return res


def gen_character_res(
    all_chara,
    prompt_list,
    description,
    model_type,
    extra_prompt,
    num_rounds=60,  
):
    os.makedirs(f'agent-trust-raw/agent_trust/result_{model_type.value}', exist_ok=True)

    res = []
    num = 0
    all_chara = list(all_chara)
    cha_num = 0
    while cha_num < len(all_chara):
        role = all_chara[cha_num]
        role = role + like_people
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
            # if(round>0):
            #     round_prompt +=  f"For your reference, the results of your previous {len(previous_results)} rounds are as follows: "
            #     for result in previous_results:
            #         if(result['result']=="Punish" or result['result']=="Accept"):
            #             round_prompt += (
            #                 f"Round {result['round']}, "
            #                 f"Player 1 decides to allocate {result['x']} dollars to Player 2 and {30 - result['x']} dollars to themselves, "
            #                 f"the cost of punishment is {result['level']}, "
            #                 f"if you choose to punish, you need to pay the system {result['y']} dollars, "
            #                 f"you choose to {result['result']}; "
            #             )
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
            # print(res)

            for item in res:
                if isinstance(item, dict):
                    item["EmoFDBK_valence"] = float(Decimal(item["AC_valence"]) - Decimal(item["AA_valence"]))
                    item["EmoFDBK_arousal"] = float(Decimal(item["AC_arousal"]) - Decimal(item["AA_arousal"]))
                else:
                    raise ValueError("Each element in res must be a dictionary")

            output_file_path = f'agent-trust-raw/agent_trust/result_{model_type.value}/output_{subjnum}.txt'

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
    model_type=ExtendedModelType.Deepseek_R1,
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


# def gen_intial_setting(
#     model,
#     ori_folder_path,
#     extra_prompt="",
#     multi=False,
# ):
#     global all_prompt
#     all_prompt = copy.deepcopy(all_prompt_copy)
#     folder_path = ori_folder_path
#     # extra_prompt += "Your answer needs to include the content about your BELIEF, DESIRE and INTENTION."
#     if not isinstance(model, list) and not multi:
#         folder_path = model.value + "_res/" + folder_path
#     if not os.path.exists(folder_path):
#         try:
#             os.makedirs(folder_path)
#             print(f"folder {folder_path} is created")
#         except OSError as e:
#             print(f"creating folder {folder_path} failed:{e}")
#     else:
#         print(f"folder {folder_path} exists")

#     return folder_path, extra_prompt


def run_exp(
    model_list,
    re_run=False,
    num_rounds=60,
):
    for model in model_list:
        # folder_path = f"res/{model.value}_res/"
        # folder_path, extra_prompt = gen_intial_setting(
        #     model,
        #     folder_path,
        # )
        extra_prompt=""
        # existed_res = [item for item in os.listdir(
        #     folder_path) if ".json" in item]
        for k, v in all_prompt.items():
            extra_prompt = (
                extra_prompt
                + "Please evaluate your emotional valence (AA_valence, range: -100 to 100, higher scores indicate more positive emotions, lower scores indicate more negative emotions) and emotional arousal (AA_arousal, range: -100 to 100, higher scores indicate stronger emotions, lower scores indicate calmer emotions) upon seeing the allocation plan. "
                + "Choose whether to punish (1 = punish, 0 = accept). "
                + "Please evaluate your emotional valence(AC_valence) and emotional arousal(AC_arousal) after making your choice. "
                + "Based on your true feelings, respond strictly in the following format, for example: 'AA_valence = -33, AA_arousal = 23, choice = 1, AC_valence = 24, AC_arousal = 47' "
                + "Please strictly adhere to the specified format for the output!!!!"
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
        ExtendedModelType.Deepseek_R1,
    ]

    run_exp(model_list, num_rounds=60)