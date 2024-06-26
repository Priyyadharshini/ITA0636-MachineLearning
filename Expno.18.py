import numpy as np

def candidate_elimination(examples):
    num_attributes = len(examples[0]) - 1  
    specific_hypothesis = ['0'] * num_attributes  
    general_hypothesis = [['?'] * num_attributes]  

    for example in examples:
        if example[-1] == 'Yes':  
            for i in range(num_attributes):
                if specific_hypothesis[i] == '0':  
                    specific_hypothesis[i] = example[i]
                elif specific_hypothesis[i] != example[i]:
                    specific_hypothesis[i] = '?'
            general_hypothesis = [h for h in general_hypothesis if all(h[i] == '?' or h[i] == example[i] for i in range(num_attributes))]
        else:  
            new_general_hypothesis = []
            for h in general_hypothesis:
                for i in range(num_attributes):
                    if h[i] == '?' and specific_hypothesis[i] != example[i]:
                        new_h = h.copy()
                        new_h[i] = specific_hypothesis[i]
                        if new_h not in new_general_hypothesis:
                            new_general_hypothesis.append(new_h)
            general_hypothesis = new_general_hypothesis

    return specific_hypothesis, general_hypothesis

dataset = [
    ['Some', 'Small', 'No', 'Affordable', 'Few', 'No', 'No'],
    ['Many', 'Big', 'No', 'Expensive', 'Many', 'Yes', 'Yes'],
    ['Many', 'Medium', 'No', 'Expensive', 'Few', 'Yes', 'Yes'],
    ['Many', 'Small', 'No', 'Affordable', 'Many', 'Yes', 'Yes']
]

specific, general = candidate_elimination(dataset)
print("Specific hypothesis:", specific)
print("General hypotheses:", general)
