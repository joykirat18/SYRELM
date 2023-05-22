import operator
import math


def calculator( arg1_value, arg2_value, operation):
    if(operation == 'add'):
        return operator.add(arg1_value, arg2_value)
    elif(operation == 'sub' or operation == 'subtract' or operation == 'subtact'):
        return operator.sub(arg1_value, arg2_value)
    elif(operation == 'mul' or operation == 'multiply' or operation == 'mult'):
        return operator.mul(arg1_value, arg2_value)
    elif(operation == 'div' or operation == 'divide'):
        if(arg2_value == 0):
            return 0
        return operator.truediv(arg1_value, arg2_value)
    elif(operation == 'mod' or operation == 'modulo'):
        return operator.mod(arg1_value, arg2_value)
    elif(operation == 'pow' or operation == 'power' or operation == 'raise_to'):
        return operator.pow(arg1_value, arg2_value)
    elif(operation == 'floor'):
        return operator.floordiv(arg1_value, arg2_value)
    elif(operation == 'gcd' or operation == 'GCD' or operation == 'find_gcd'):
        return math.gcd(int(arg1_value), int(arg2_value))
    elif(operation == 'lcm' or operation == 'LCM' or operation == 'find_lcm'):
        return abs(int(arg1_value)*int(arg2_value)) / math.gcd(int(arg1_value), int(arg2_value))


import re
def generateVector(sentence, answer):
    pattern = r'var(\d+)\s*=\s*\[find\]\((.*?)\)\s*#\s*(\d+(\.\d+)?)'
    variables = {}
    # print(re.findall(pattern, sentence))
    find_values = []
    for match in re.findall(pattern, sentence):
        variables['var'+match[0]] = float(match[2])
        find_values.append(float(match[2]))
    pattern = r'(\w+)\s*=\s*\[(\w+)\]\((.*?)\)'
    sequences = re.findall(pattern, sentence)
    from generate_iterative import is_float, calculator
    import operator
    # print(sequences)
    values = []
    for seq in sequences:
        temp = []
        temp.append(seq[0])
        temp.append(seq[1])
        if(seq[1] == 'find'):
            if(seq[0] in variables):
                temp.append(variables[seq[0]])
            else:
                temp.append(0)
        elif(seq[1] in ['add', 'sub', 'subtract', 'subtact', 'mul', 'multiply', 'mult', 'div', 'divide', 'mod', 'modulo', 'pow', 'power', 'raise_to', 'floor', 'gcd', 'GCD', 'find_gcd', 'lcm', 'LCM', 'find_lcm']):
            if(',' in seq[2]):
                operand = seq[2].split(',')
                temp.append(operand[0].strip())
                temp.append(operand[1].strip())
            else:
                temp.append(seq[2].strip())
                temp.append(seq[2].strip())
        values.extend(temp)
    returnStat = re.findall(r'\[return\]\((.*?)\)', sentence)
    if(len(returnStat) > 0):
        values.append('return')
        values.append(returnStat[0])
    values.append(answer)
    return values, find_values

def padArray(array1,array2):
    max_length = max(len(array1), len(array2))
    padding = [''] * (max_length - len(array1))
    array1_padded = array1 + padding

    padding = [''] * (max_length - len(array2))
    array2_padded = array2 + padding
    array1_padded = ' '.join([str(elem) for elem in array1_padded])
    array2_padded = ' '.join([str(elem) for elem in array2_padded])
    return array1_padded, array2_padded


from sklearn.preprocessing import LabelEncoder
def convertToIds(value1, value2):
    values = value1 + value2
    le = LabelEncoder()
    index_values = le.fit_transform(values)

    # get the mapping between labels and index values
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    for i in range(len(value1)):
        value1[i] = mapping[str(value1[i])]
    for i in range(len(value2)):
        value2[i] = mapping[str(value2[i])]
    return value1, value2
    
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def parse_Interim_pseudocode(sentence):   
    # try:
    import re
    pattern = r'var(\d+)\s*=\s*\[find\]\((.*?)\)\s*#\s*(\d+(\.\d+)?)'
    variables = {}
    for match in re.findall(pattern, sentence):
        # print(match)
        variables['var'+match[0]] = float(match[2])
    # print(variables)
    pattern = r'(\w+)\s*=\s*\[(\w+)\]\((.*?)\)'
    sequences = re.findall(pattern, sentence)
    # print(sequences)
    for sequence in sequences:
        if(sequence[1] != 'find'):
            if(',' in sequence[2]):
                args = sequence[2].strip().split(',')
                if(len(args) == 2):
                    arg1 = sequence[2].strip().split(',')[0].strip()
                    arg2 = sequence[2].strip().split(',')[1].strip()
                    operation = sequence[1].strip()
                    if(operation in ['add', 'sub', 'subtract', 'subtact', 'mul', 'multiply', 'mult', 'div', 'divide', 'mod', 'modulo', 'pow', 'power', 'raise_to', 'floor', 'gcd', 'GCD', 'find_gcd', 'lcm', 'LCM', 'find_lcm']):
                        if(arg1 in variables and arg2 in variables):
                            variables[sequence[0]] = calculator(variables[arg1], variables[arg2], operation)
                        elif(arg1 in variables and arg2 not in variables and is_float(arg2)):
                            variables[sequence[0]] = calculator(variables[arg1], float(arg2), operation)
                        elif(arg1 not in variables and arg2 in variables and is_float(arg1)):
                            variables[sequence[0]] = calculator(float(arg1), variables[arg2], operation)
                        elif(arg1 not in variables and arg2 not in variables and is_float(arg1) and is_float(arg2)):
                            variables[sequence[0]] = calculator(float(arg1), float(arg2), operation)
    
    pattern = r'\[(\w+)\]'
    operations = re.findall(pattern, sentence)
    if(len(operations) > 0):
        if(operations[-1] != 'return'):
            if(len(sequences) > 0):
                if(sequences[-1][0] not in variables):
                    return ("NONE", False)
                return (variables[sequences[-1][0]], False)
            else:
                return ("NONE", False)
        elif(operations[-1] == 'return'):
            returnStat = re.findall(r'\[return\]\((.*?)\)', sentence)

            if(len(returnStat) > 0):
                if(returnStat[0] not in variables):
                    return ("NONE", True)
                return (variables[returnStat[0]], True)
            else:
                return ("NONE", True)
    return ("NONE", False) 
        
import datasets
import numpy as np
from datasets.config import importlib_metadata, version
from nltk.translate import meteor_score
from nltk import word_tokenize
def compute(predictions, references, alpha=0.9, beta=3, gamma=0.5):
    multiple_refs = isinstance(references[0], list)
        # the version of METEOR in NLTK version 3.6.5 and earlier expect tokenized inputs
    if multiple_refs:
        scores = [
            meteor_score.meteor_score(
                [word_tokenize(ref) for ref in refs],
                word_tokenize(pred),
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            for refs, pred in zip(references, predictions)
        ]
    else:
        scores = [
            meteor_score.single_meteor_score(
                word_tokenize(ref), word_tokenize(pred), alpha=alpha, beta=beta, gamma=gamma
            )
            for ref, pred in zip(references, predictions)
        ]


    return {"meteor": np.mean(scores)}