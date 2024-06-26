import pandas as pd

data = {
    'Example': [1, 2, 3, 4],
    'Shape': ['Circular', 'Circular', 'Oval', 'Oval'],
    'Size': ['Large', 'Large', 'Large', 'Large'],
    'Color': ['Light', 'Light', 'Dark', 'Light'],
    'Surface': ['Smooth', 'Irregular', 'Smooth', 'Irregular'],
    'Thickness': ['Thick', 'Thick', 'Thin', 'Thick'],
    'Target Concept': ['+', '+', '-', '+']
}

df = pd.DataFrame(data)

hypothesis = ['0', '0', '0', '0', '0']

for index, row in df.iterrows():
    if row['Target Concept'] == '+':
        for i in range(len(hypothesis)):
            if hypothesis[i] == '0':
                hypothesis[i] = row.iloc[i + 1]  

print("The most specific hypothesis is:", hypothesis)
