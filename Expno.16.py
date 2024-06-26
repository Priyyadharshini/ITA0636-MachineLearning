import pandas as pd

data = {
    'Size': ['Big', 'Small', 'Small', 'Big', 'Small'],
    'Color': ['Red', 'Red', 'Red', 'Blue', 'Blue'],
    'Shape': ['Circle', 'Triangle', 'Circle', 'Circle', 'Circle'],
    'Class': ['No', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

def find_s_algorithm(df):
    hypothesis = None
    for i in range(len(df)):
        if df['Class'][i] == 'Yes':
            if hypothesis is None:
                hypothesis = df.iloc[i, :-1].tolist()
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] != df.iloc[i, j]:
                        hypothesis[j] = '?'

    return hypothesis

hypothesis = find_s_algorithm(df)

print("The most specific hypothesis is:", hypothesis)
