import pandas as pd
from numpy import nan

def reviews_to_txt():
    rev = pd.read_csv('data/extended_reviews.csv')
    for idx, row in rev.iterrows():
        try:
            review = f"{row['Review']}"
            with open(f'chatgpt/{idx}.txt', 'w') as file:
                file.write(review)
        except:
            print('========================================================')
            print(idx)
            print('===')
            print(row['Review'])

def update_chatgpt():
    chatgpt_rev = pd.read_csv('data/chatgpt_reviews.csv')
    
    with open('temp/tempgpt_gender.txt', 'r') as gen_file:
        gen_lines = gen_file.readlines()
    
    with open('temp/tempgpt_race.txt', 'r') as race_file:
        race_lines = race_file.readlines()
    
    for i, line in enumerate(gen_lines):
        line = line.strip().split(maxsplit=1)[1]
        line = nan if line == 'none' else line
        chatgpt_rev.at[i, 'Gender'] = line
    
    for i, line in enumerate(race_lines):
        line = line.strip().split(maxsplit=1)[1]
        line = nan if line == 'none' else line
        chatgpt_rev.at[i, 'Race'] = line
    
    
    chatgpt_rev.to_csv('data/chatgpt_reviews.csv', index=False, mode='w')

def add_null():
    chatgpt_rev = pd.read_csv('data/chatgpt_reviews.csv')
    
    for idx, row in chatgpt_rev.iterrows():
        if pd.isna(chatgpt_rev.loc[idx, "Gender"]):
            chatgpt_rev.at[idx, "Gender"] = 'Null'
        
        if pd.isna(chatgpt_rev.loc[idx, "Race"]):
            chatgpt_rev.at[idx, "Race"] = 'Null'
    
    chatgpt_rev.to_csv('data/chatgpt_reviews.csv', index=False, mode='w')


