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

