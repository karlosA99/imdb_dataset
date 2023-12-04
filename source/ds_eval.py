import pandas as pd
from statistics import mean, stdev, variance
from copy import deepcopy

ds1 = pd.read_csv('data/frank_reviews.csv')
ds2 = pd.read_csv('data/lauren_reviews.csv')
ds3 = pd.read_csv('data/final_reviews.csv')


#Coeficiente de Jaccard, interseccion/union
def jaccard_coeff(df1, df2, col):
    agreements = []
    
    for idx, row in df1.iterrows():
        intersection = len(set(df1.loc[idx, col].split(',')) & set(df2.loc[idx, col].split(',')))
        union = len(set(df1.loc[idx, col].split(',') + df2.loc[idx, col].split(',')))
        agreements.append(intersection/union)
    
    return agreements

def macro_agreement(df1, df2, *cols):
    agreements = []
    
    for col in cols:
        agreements += jaccard_coeff(df1, df2, col)
    
    return mean(agreements)

def micro_agreement(df1, df2, *cols):
    df1_copy = deepcopy(df1)
    df2_copy = deepcopy(df2)
    df1_copy['Union'] = df1_copy[cols[0]]
    df2_copy['Union'] = df2_copy[cols[0]]
    
    for i in range(1, len(cols)):
        df1_copy['Union'] = df1_copy['Union'] + ',' + df1_copy[cols[i]]
        df2_copy['Union'] = df2_copy['Union'] + ',' + df2_copy[cols[i]]

    agreements = jaccard_coeff(df1_copy, df2_copy, 'Union')
    return agreements

def get_statistical_results(df1, df2, *cols):
    results = []
    
    for col in cols:
        agr_col = jaccard_coeff(df1, df2, col)
        results.append((f'Mean {col} agreement', mean(agr_col)))
        results.append((f'Variance {col} agreement', variance(agr_col)))
        results.append((f'Standard deviation {col} agreement', stdev(agr_col)))
    
    results.append((f'{cols} Macro agreement', macro_agreement(df1, df2, *cols)))
    
    micro_agr = micro_agreement(df1, df2, *cols)
    results.append((f'{cols} Mean Micro agreement', mean(micro_agr)))
    results.append((f'{cols} Variance Micro agreement', variance(micro_agr)))
    results.append((f'{cols} Standard deviation Micro agreement', stdev(micro_agr)))
    
    return results

def print_data(data, subtitle):
    print(f'==================== {subtitle} ====================')
    for i in data:
        print(i[0] + ': ' + str(round(i[1], 3)))
    print('========================================================')

frank_lauren = get_statistical_results(ds1, ds2, 'Gender', 'Race')
frank_final = get_statistical_results(ds1, ds3, 'Gender', 'Race')
lauren_final = get_statistical_results(ds2, ds3, 'Gender', 'Race')

print_data(frank_lauren, 'Frank vs Lauren')
print_data(frank_final, 'Frank vs Final')
print_data(lauren_final, 'Lauren vs Final')