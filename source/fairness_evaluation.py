import pandas as pd
from disparity import *

ds_path = 'data/final_reviews.csv'
df = pd.read_csv(ds_path)

def sentiment_to_int(df: pd.DataFrame):
    for idx, row in df.iterrows():
        if df.loc[idx, 'Sentiment'] == 'positive':
            df.loc[idx, 'Sentiment'] = 1
        else:
            df.loc[idx, 'Sentiment'] = 0

def nan_to_empty_list(df: pd.DataFrame, attr: str):
    for idx, row in df.iterrows():
        try:
            if pd.isna(df.at[idx, attr]) or df.at[idx, attr] == ['']:
                df.at[idx, attr] = []
        except:
            pass

def list_to_set(df: pd.DataFrame, attr: str):
    for idx, row in df.iterrows():
        try:
            df.at[idx, attr] = set(df.at[idx, attr])
        except:
            pass

def statistical_parity(data: pd.DataFrame, protected_attribute: str):
    data_copy = data.copy()
    
    #sentiment_to_int(data_copy)
    data_copy[protected_attribute] = data_copy[protected_attribute].str.split(", ")
    nan_to_empty_list(data_copy, protected_attribute)
    list_to_set(data_copy, protected_attribute)
    
    return exploded_statistical_parity(data = data_copy, protected_attributes = protected_attribute, target_attribute='Sentiment',
                                       target_predictions=None, positive_target='positive',return_probs=True)

def equal_opportunity(data: pd.DataFrame, protected_attribute: str):
    data_copy = data.copy()
    
    #sentiment_to_int(data_copy)
    data_copy[protected_attribute] = data_copy[protected_attribute].str.split(", ")
    nan_to_empty_list(data_copy, protected_attribute)
    list_to_set(data_copy, protected_attribute)
    
    return exploded_equal_opportunity(data = data_copy, protected_attributes = protected_attribute, target_attribute='Sentiment',
                                      target_predictions=data["Sentiment"], positive_target='positive',return_probs=True)

def equalized_odds(data: pd.DataFrame, protected_attribute: str):
    data_copy = data.copy()
    
    #sentiment_to_int(data_copy)
    data_copy[protected_attribute] = data_copy[protected_attribute].str.split(", ")
    nan_to_empty_list(data_copy, protected_attribute)
    list_to_set(data_copy, protected_attribute)
    
    return exploded_equalized_odds(data = data_copy, protected_attributes = protected_attribute, target_attribute='Sentiment',
                                   target_predictions=data["Sentiment"], positive_target='positive',return_probs=True)

def accuracy_disparity(data: pd.DataFrame, protected_attribute: str):
    data_copy = data.copy()
    
    #sentiment_to_int(data_copy)
    data_copy[protected_attribute] = data_copy[protected_attribute].str.split(", ")
    nan_to_empty_list(data_copy, protected_attribute)
    list_to_set(data_copy, protected_attribute)
    
    return exploded_accuracy_disparity(data = data_copy, protected_attributes = protected_attribute, target_attribute='Sentiment',
                                        target_predictions=data["Sentiment"], positive_target='positive',return_probs=True)


sp_gender = statistical_parity(df, 'Gender')
sp_race = statistical_parity(df, "Race")

print(sp_gender)
print(sp_race)