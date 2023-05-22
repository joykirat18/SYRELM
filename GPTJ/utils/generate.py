import torch
import operator
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate(model, tokenizer, prompt,target,k=0,p=0.99,output_length=120,temperature=1,num_return_sequences=1,repetition_penalty=1.0):
    model.to('cuda')
    model.eval()
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=True,return_tensors='pt')

    with torch.no_grad():
        output_sequence = model.generate(
            # input_ids=encoded_prompt,
            input_ids=encoded_prompt.to('cuda'),
            max_new_tokens=output_length,
            temperature=temperature,
            top_k=k,
            top_p=p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
        
        )

    output_sequence = output_sequence[0].tolist()
    text = tokenizer.decode(output_sequence, clean_up_tokenization_spaces=True)
    
    text = text[:text.find('<endoftext>')].strip()
    if('<sep>' in text):
        text = text[text.find('<sep>') + 5:].strip()
    else:
        text = text[len(prompt + ' <sep>'):].strip()

    return text


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
    
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def parse_pseudocode(sentence):   
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
                        
        # print(sequence)
        # print(variables)
    returnStat = re.findall(r'\[return\]\((.*?)\)', sentence)

    if(len(returnStat) > 0):
        if(returnStat[0] not in variables):
            return "NONE"
        # print(variables[returnStat[0]])
        return variables[returnStat[0]]
    else:
        return "NONE"

from transformers import pipeline, set_seed, StoppingCriteriaList, StoppingCriteria
import torch
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, tokenizer=None,stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.tokenizer=tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # print(input_ids)
        full_sentence = self.tokenizer.decode(input_ids[0].tolist(), clean_up_tokenization_spaces=True).strip()
        last_token = self.tokenizer.decode(input_ids[0][-1].tolist(), clean_up_tokenization_spaces=True).strip()
        import re
        pattern = r'\[(\w+)\]'
        sequence = re.findall(pattern, full_sentence)
        # print(sequence)
        if(len(sequence) > 0 and sequence[-1] != 'find' and last_token == '#'):
            return True
        if(last_token == '<endoftext>'):
            return True
        return False




def generate_pseudoCode(model, tokenizer, prompt,target,k=4,p=0.7,output_length=120,temperature=1,num_return_sequences=1,repetition_penalty=1.0):
    # model.to('cuda')
    # model.eval()
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=True,return_tensors='pt')[0]
    stop_words = ["<endoftext>"]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer=tokenizer,stops=stop_words)])
    # input_ids = encoded_prompt['input_ids']
    # attention_mask = encoded_prompt['attention_mask']
    query_tensor = encoded_prompt
    i = 0
    while(i<=5):
    # print(encoded_prompt)
        with torch.no_grad():
            output_sequence = model.generate(
                encoded_prompt.to('cuda'),
                max_new_tokens=output_length,
                temperature=temperature,
                top_k=k,
                top_p=p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                stopping_criteria=stopping_criteria
                # num_return_sequences=num_return_sequences,
            
                )

        if len(output_sequence.shape) > 2:
            output_sequence.squeeze_()
        if(output_sequence.shape[-1] > len(query_tensor) + 120):
            break
        # print(output_sequence)
        i += 1
        output_sequence = output_sequence[0].tolist()
        text = tokenizer.decode(output_sequence, clean_up_tokenization_spaces=True)
        from utils.generate_iterative import parse_Interim_pseudocode
        (output, endOfSentence) = parse_Interim_pseudocode(text)
        # output = 'NONE'
        if(output != 'NONE'):
            if(endOfSentence):
                text += (' ' + str(output))
            else:
                text += (' ' + str(output) + ',')
        # print(text)
        encoded_prompt = tokenizer.encode(text, add_special_tokens=True,return_tensors='pt')[0]


        
        if('<endoftext>' in text):
            break
    # print(text)
    text = text[:text.find('<endoftext>')].strip()
    if('<sep>' in text):
        text = text[text.find('<sep>') + 5:].strip()
    else:
        text = text[len(prompt + ' <sep>'):].strip()

    response_tensor = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")[0]
    
    # query_tensor = encoded_prompt
    response_text = text
    # query_tensor = query_tensor.long()
    # response_tensor = response_tensor.long()
    if(len(response_text) <= 4):
        response_text = "NO RESPONSE GENERATED"
        response_tensor = tokenizer.encode(response_text, add_special_tokens=True, return_tensors="pt")[0]

    return query_tensor, response_tensor, response_text


from tqdm import tqdm
import json
def getAccuracy(model, tokenizer, datasetPath):
    correct = 0
    total = 0

    with open(datasetPath, 'r') as f:
        data = json.load(f)
    import random
    # random.shuffle(data)
    # data = data[:50]
    for d in tqdm(data):
        # try:
        question = d['Body'].strip() + ' ' + d['Question'].strip() + ' Translate the following into pseudoCode'
        answer = d['Answer']
        response = generate(model,tokenizer,question,answer)
        # response = generate_pseudoCode(model,tokenizer,question,answer)
        # print(question)
        print(response)
        print(answer)
        parsed = parse_pseudocode(response)
        print(parsed)
        print(parsed == answer)
        if(parsed == answer):
            correct += 1
        total += 1
        # except:
        #     print("error")
        #     pass
    print(str(datasetPath) + " accuracy: ", correct, total, correct/total)
    # model.train()
    return correct / total

if __name__ == "__main__":
    from loadModel import loadLoraCheckPoint
    model,tokenizer = loadLoraCheckPoint()
    getAccuracy(model, tokenizer, 'data/eval_dataset/svamp.json')


