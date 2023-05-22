max_reward = 1
import torch
from utils.pseudoCodeUtils import parse_pseudocode, parseLetterIntergers, findNumericalValues, get_match_reward, calculate_answer_diff
def getReward(response_text, question_text,gold_response_text,gold_answer):
    print(response_text)
    print(question_text)

    if(parse_pseudocode(response_text) == "NONE"):
        R1 = 0
    else:
        print("error R1")
        R1 = max_reward
    
    try:
        question_text_parsed = parseLetterIntergers(question_text)
        N = set(findNumericalValues(question_text_parsed))
        S = []

        response_text_ans = response_text.split(',')
        for sentence in response_text_ans:
            if('[find]' in sentence):
                S.extend(findNumericalValues(sentence))
        S = set(S)

        R2 = max_reward - len(N-S)/len(N) 
    except:
        print("error R2")
        R2 = 0
        pass

    try:
        R3 = max_reward * get_match_reward(response_text, gold_response_text)
    except:
        print("error R3")
        R3 = 0
        pass

    try:
        answer = parse_pseudocode(response_text)
        gold_answer = float(gold_answer)
        R4 = calculate_answer_diff(answer, gold_answer)
    except:
        print("error R4")
        R4 = 0
        pass

    print(R1, R2, R3, R4)
    R = (R1 + R2 + R3 + R4) / 4
    
    return torch.tensor(float(R))
