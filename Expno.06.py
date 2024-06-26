import pandas as pd

# Step 1: Create the dataset
data = {
    'Origin': ['Japan', 'Japan', 'Japan', 'USA', 'Japan'],
    'Manufacturer': ['Honda', 'Toyota', 'Toyota', 'Chrysler', 'Honda'],
    'Color': ['Blue', 'Green', 'Blue', 'Red', 'White'],
    'Decade': [1980, 1970, 1990, 1980, 1980],
    'Type': ['Economy', 'Sports', 'Economy', 'Economy', 'Economy'],
    'Example Type': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive']
}

df = pd.DataFrame(data)

# Step 2: Initialize the most specific hypothesis
hypothesis = ['Ø', 'Ø', 'Ø', 'Ø', 'Ø']

# Step 3: Find-S Algorithm
for index, row in df.iterrows():
    if row['Example Type'] == 'Positive':
        for i in range(len(hypothesis)):
            if hypothesis[i] == 'Ø':
                hypothesis[i] = row[i]
    elif hypothesis[i] != row[i]:
                hypothesis[i] = '?'

# Step 4: Output the most specific hypothesis
print("The most specific hypothesis is:", hypothesis)

