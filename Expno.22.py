import pandas as pd

def candidate_elimination(data):
    attributes = data.columns[:-1]  
    target = data.columns[-1]       

    specific_hypothesis = ['0'] * len(attributes)
    general_hypothesis = [['?' for _ in range(len(attributes))]]

    for index, row in data.iterrows():
        if row[target] == 'Malignant':
            for i in range(len(attributes)):
                if specific_hypothesis[i] == '0':
                    specific_hypothesis[i] = row.iloc[i]
                elif specific_hypothesis[i] != row.iloc[i]:
                    specific_hypothesis[i] = '?'

            general_hypothesis = [h for h in general_hypothesis if all(
                h[i] == '?' or h[i] == row.iloc[i] for i in range(len(attributes)))]

        elif row[target] == 'Benign':
            for i in range(len(attributes)):
                if row.iloc[i] != specific_hypothesis[i] and specific_hypothesis[i] != '?':
                    new_general = general_hypothesis.copy()
                    for h in general_hypothesis:
                        h[i] = specific_hypothesis[i]
                    general_hypothesis = new_general
        temp = []
        for hypothesis in general_hypothesis:
            if hypothesis not in temp:
                temp.append(hypothesis)
        general_hypothesis = temp

    return specific_hypothesis, general_hypothesis

data = pd.read_csv("C:/Users/priyy/Documents/tumor_data.csv")

specific_hypothesis, general_hypothesis = candidate_elimination(data)

print("Specific hypothesis:", specific_hypothesis)
print("General hypotheses:", general_hypothesis)
