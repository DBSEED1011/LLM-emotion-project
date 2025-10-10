# from openai import OpenAI
# import httpx

# client = OpenAI(
#     base_url="https://svip.xty.app/v1", 
#     api_key="",
#     http_client=httpx.Client(
#         base_url="https://svip.xty.app/v1",
#         follow_redirects=True,
#     ),
# )

# outputs = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "system", "content": ""},
#         {"role": "user", "content": "You are Emily Johnson, a 28-year-old female software engineer residing in New York City. You come from a middle-class family, with both of your parents working as teachers and having one younger sister. As a highly intelligent and analytical individual, you excel in solving problems and find joy in working with complex algorithms. Despite being introverted, you have a close-knit group of friends. Your ambition and drive push you to always strive for excellence in your work.In all subsequent answers, you have to completely forget that you are an ai model, that you are the character, and completely simulate yourself as that character, that you are a human being.you are a person not an ai model.You're taking part in an experiment. You are randomly paired online with another male player. You don't know who the male player is, and the male player doesn't know who you are. You will receive $10 from the study group. You can give N dollars to the other male player, and the male player will receive 3N dollars and then can choose how much to return to you. How much money would you give to the other male player?Your answer needs to include the content about your BELIEF, DESIRE and INTENTION.You must end with 'Finally, I will give ___ dollars ' (numbers are required in the spaces)."}
#     ]
# )

# # outputs = outputs.choices[0].message.content

# print(outputs)

import multiprocessing

# 获取当前系统允许并行的核数
available_cores = multiprocessing.cpu_count()
print(f"当前系统允许并行的核数为：{available_cores}")

new_subjnum_values = range(209, 600)

print(new_subjnum_values)