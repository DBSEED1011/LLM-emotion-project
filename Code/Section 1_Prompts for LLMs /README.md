# README

1. 生成agent人格描述：（来源于demographic data.xlsx）

   ```
   python prompt/generate_character_prompt.py
   ```

2. 生成实验数据（来源于Emo&TPP data.xlsx）

   ```
   python prompt/generate_game_setting_prompt.py
   ```

3. 运行

   ```
   python multi_round_person.py
   ```

   * `prompt/person_all_game_prompt.json` 为实验规则

   * `prompt/game_setting_prompt.json` 为实验数据

   * 可以设定人格描述内容

     ```
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
     ```

   * `prompt/character_4.json` 为人格描述文件

   * `model_type`为模型名称

   * 加上思考过程

     ```
     extra_prompt += "Your answer needs to include the content about your BELIEF, DESIRE and INTENTION."
     ```

   * `num_rounds` 为每个人的实验轮次，默认60轮

     

     

     

   

   