from os import walk

from numpy import NaN

from utils.website_list import df
from utils.utils import DATABASE_PATH

total = 0
types = [0, 0, 0] # TRUE, MIXED, FALSE

for (_, _, files) in walk(DATABASE_PATH):
    for file in files:
        label = file.split('-')[0]
        label = label.replace('_', ' ')

        type = df.loc[df["Nom"] == label]["Overview"].values[0]
        if type == True:
            types[0] += 1
        elif type == False:
            types[2] += 1

        total += 1
types[1] = total - types[0] - types[2] 

print(f"True / Mixed / False : {types}")
print(f"Total : {total}")
