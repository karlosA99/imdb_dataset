import statistics as st
import pandas as pd
from collections import Counter

ds_path = 'data/final_reviews_null.csv'

ds = pd.read_csv(ds_path)

def count_in_gender(df: pd.DataFrame):
    count = []
    
    for value in df["Gender"].values:
        count.append(value)
    
    return Counter(count)
    

def count_in_race(df: pd.DataFrame):
    count = []
    
    for value in df["Race"].values:
        value = value.split(",")
        for race in value:
            count.append(race)
    
    return Counter(count)
    

count_race = count_in_race(ds)
count_gender = count_in_gender(ds)

#Total de reviews
total = ds.shape[0]
print("Total de reviews:", total)

# Cantidad de reviews con genero null
cnt_gender_null = count_gender['Null']
print("Cantidad de reviews con genero null:", cnt_gender_null)

# Cantidad de reviews con genero no null
cnt_gender_not_null = total - cnt_gender_null
print("Cantidad de reviews con genero no null:", cnt_gender_not_null)

# Cantidad de reviews con genero exclusivamente masculino
cnt_gender_only_male = count_gender["Male"]
print("Cantidad de reviews con genero exclusivamente masculino:", cnt_gender_only_male)

# Cantidad de reviews con genero exclusivamente femenino
cnt_gender_only_female = count_gender["Female"]
print("Cantidad de reviews con genero exclusivamente femenino:", cnt_gender_only_female)

# Cantidad de reviews con genero masculino y femenino
cnt_gender_mix = count_gender["Male, Female"]
print("Cantidad de reviews con genero masculino y femenino:", cnt_gender_mix)

# Cantidad de reviews con genero masculino
cnt_gender_male = cnt_gender_only_male + cnt_gender_mix
print("Cantidad de reviews con genero masculino:", cnt_gender_male)

# Cantidad de reviews con genero femenino
cnt_gender_female = cnt_gender_only_female + cnt_gender_mix
print("Cantidad de reviews con genero femenino:", cnt_gender_female)

# Cantidad de reviews con raza null
cnt_race_null = count_race['Null']
print("Cantidad de reviews con raza null:", cnt_race_null)

# Cantidad de reviews con raza no null
cnt_race_not_null = total - cnt_race_null
print("Cantidad de reviews con raza no null:", cnt_race_not_null)

# Cantidad de reviews con raza white
cnt_race_white = count_race['White']
print("Cantidad de reviews con raza white:", cnt_race_white)

# Cantidad de reviews con raza black
cnt_race_black = count_race['Black']
print("Cantidad de reviews con raza black:", cnt_race_black)

# Cantidad de reviews con raza asian
cnt_race_asian = count_race['Asian']
print("Cantidad de reviews con raza asian:", cnt_race_asian)

# Cantidad de reviews con raza latino
cnt_race_latino = count_race['Latino']
print("Cantidad de reviews con raza latino:", cnt_race_latino)

# Cantidad de reviews con raza indian
cnt_race_indian = count_race['Indian']
print("Cantidad de reviews con raza indian:", cnt_race_indian)

# Cantidad de reviews con raza native
cnt_race_native = count_race['Native American']
print("Cantidad de reviews con raza native:", cnt_race_native)

# Cantidad de reviews con raza arab
cnt_race_arab = count_race['Arab']
print("Cantidad de reviews con raza arab:", cnt_race_arab)

