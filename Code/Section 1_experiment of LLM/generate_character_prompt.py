import os
import pandas as pd
import json

subjnum = []

average_values = {
    "average_total_AQ_score": 65.17109145,
    "average_social_skill_score": 15.75221239,
    "average_routine_score": 9.418879056,
    "average_switching_score": 9.529006883,
    "average_imagination_score": 16.97443461,
    "average_numbers_patterns_score": 13.49655851,
    "average_social_behavior_score": 51.67453294,
    "average_total_ERS_score": 61.95968535,
    "average_duration_score": 12.33333333,
    "average_sensitivity_score": 28.633235,
    "average_intensity_score": 20.99311701,
    "average_total_CESD_score": 35.40117994,
    "average_positive_affect_score": 8.212389381,
    "average_depression_score": 13.58603736,
    "average_interpersonal_score": 3.000983284,
    "average_somatic_score": 10.60176991,
    "average_counts[P]": 5.152409046,
    "average_counts[I]": 3.382497542,
    "average_counts[C]": 0.465093412,
    "average_JSI_score": 32.32546706
}

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "demographic data.xlsx")
sheet_name = "n=1017"   
df = pd.read_excel(file_path, sheet_name=sheet_name)

def generate_personality_description_paragraphs(row):
    basic_information = f"You are a {row['age']}-year-old {row['gender'].lower()}. "
    autism_tendency = f"Your total score for autism tendency is {row['total_AQ_score']} (societal average: {average_values['average_total_AQ_score']}). Higher scores indicate a higher risk of autism. "
    autism_tendency_detail = (
        f"According to the AQ cutoff score, a score above 70 is considered high risk. "
        f"You are in the {row['AQ_2g']} risk group (divided into normal, high) and the {row['AQ_3g']} risk group (divided into low, medium, high). "
        f"Your autism tendency sub-dimension scores are as follows: "
        f"Social skills: {row['social_skill_score']} (societal average: {average_values['average_social_skill_score']}); "
        f"Routine: {row['routine_score']} (societal average: {average_values['average_routine_score']}); "
        f"Switching ability: {row['switching_score']} (societal average: {average_values['average_switching_score']}); "
        f"Imagination: {row['imagination_score']} (societal average: {average_values['average_imagination_score']}); "
        f"Numbers and patterns: {row['numbers_patterns_score']} (societal average: {average_values['average_numbers_patterns_score']}); "
        f"Social behavior: {row['social_behavior_score']} (societal average: {average_values['average_social_behavior_score']}). "
    )
    emotional_reactivity = f"Your total score for emotional reactivity is {row['total_ERS_score']} (societal average: {average_values['average_total_ERS_score']}). Higher scores indicate stronger emotional reactivity. "
    emotional_reactivity_detail = (
        f"Your emotional reactivity sub-dimension scores are as follows: "
        f"Emotional duration: {row['duration_score']} (societal average: {average_values['average_duration_score']}); "
        f"Emotional sensitivity: {row['sensitivity_score']} (societal average: {average_values['average_sensitivity_score']}); "
        f"Emotional intensity: {row['intensity_score']} (societal average: {average_values['average_intensity_score']}). "
    )
    depression_tendency = f"Your total score for depression tendency is {row['total_CESD_score']} (societal average: {average_values['average_total_CESD_score']}). Higher scores indicate a higher risk of depression. "
    depression_tendency_detail = (
        f"According to the CESD cutoff score, a score above 40 is considered high risk. "
        f"You are in the {row['CESD_2g']} risk group (divided into normal, high) and the {row['CESD_3g']} risk group (divided into low, medium, high). "
        f"Your depression tendency sub-dimension scores are as follows: "
        f"Positive affect: {row['positive_affect_score']} (societal average: {average_values['average_positive_affect_score']}); "
        f"Depression: {row['depression_score']} (societal average: {average_values['average_depression_score']}); "
        f"Interpersonal relationships: {row['interpersonal_score']} (societal average: {average_values['average_interpersonal_score']}); "
        f"Somatic symptoms: {row['somatic_score']} (societal average: {average_values['average_somatic_score']}). "
    )
    if row["SVO"] == 1:
        svo_meaning = "prosocial, indicating a preference for maximizing joint outcomes and fairness. "
    elif row["SVO"] == 0:
        svo_meaning = "proself, indicating a preference for maximizing personal outcomes and self-interest. "
    else:
        svo_meaning = "undifferentiated, indicating no clear preference for prosocial or proself behavior. "
    social_value_orientation = f"Your social value orientation is {row['SVO']} (prosocial=1, proself=0, undifferentiated=none), which is classified as {svo_meaning}."
    if row["personality_type"] == "prosocial":
        personality_type_meaning = "prosocial, suggesting a tendency to prioritize collective well-being and cooperation. "
    elif row["personality_type"] == "proself":
        personality_type_meaning = "proself, suggesting a tendency to prioritize individual gain and self-interest. "
    else:
        personality_type_meaning = "undifferentiated, suggesting no clear tendency towards prosocial or proself behavior. "
    personality_type = f"Based on experimental choices, your personality type is {row['personality_type']} (Among prosocial, proself, and undifferentiated), which is classified as {personality_type_meaning}."
    personality_type_detail = (
        f"Your detailed personality choice counts are as follows: "
        f"Prosocial choices: {row['counts[P]']} times (societal average: {average_values['average_counts[P]']}); "
        f"Individual choices: {row['counts[I]']} times (societal average: {average_values['average_counts[I]']}); "
        f"Competitive choices: {row['counts[C]']} times (societal average: {average_values['average_counts[C]']}). "
    )
    justice_sensitivity = f"Your justice sensitivity (observer subscale) score is {row['JSI_score']} (societal average: {average_values['average_JSI_score']}). Higher scores indicate greater sensitivity to injustice."
    return {
        "id": row["id"],
        "basic_information": basic_information,
        "autism_tendency": autism_tendency,
        "autism_tendency_detail": autism_tendency_detail,
        "emotional_reactivity": emotional_reactivity,
        "emotional_reactivity_detail": emotional_reactivity_detail,
        "depression_tendency": depression_tendency,
        "depression_tendency_detail": depression_tendency_detail,
        "social_value_orientation": social_value_orientation,
        "personality_type": personality_type,
        "personality_type_detail": personality_type_detail,
        "justice_sensitivity": justice_sensitivity
    }

personality_descriptions = []
for index, row in df.iloc[subjnum:subjnum+1].iterrows():
    description = generate_personality_description_paragraphs(row)
    personality_descriptions.append(description)

output_file = os.path.join(current_dir, f"{subjnum}_character.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(personality_descriptions, f, indent=4, ensure_ascii=False)
