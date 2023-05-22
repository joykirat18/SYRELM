import re
import math
import operator

def calculator_parse( arg1_value, arg2_value, operation):
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
                            variables[sequence[0]] = calculator_parse(variables[arg1], variables[arg2], operation)
                        elif(arg1 in variables and arg2 not in variables and is_float(arg2)):
                            variables[sequence[0]] = calculator_parse(variables[arg1], float(arg2), operation)
                        elif(arg1 not in variables and arg2 in variables and is_float(arg1)):
                            variables[sequence[0]] = calculator_parse(float(arg1), variables[arg2], operation)
                        elif(arg1 not in variables and arg2 not in variables and is_float(arg1) and is_float(arg2)):
                            variables[sequence[0]] = calculator_parse(float(arg1), float(arg2), operation)

    returnStat = re.findall(r'\[return\]\((.*?)\)', sentence)

    if(len(returnStat) > 0):
        if(returnStat[0] not in variables):
            return "NONE"
        return float(variables[returnStat[0]])
    else:
        return "NONE"

def parseLetterIntergers(text):
    text = text.lower()

    pattern = r'\b((?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)?[-\s]?(?:one|two|three|four|five|six|seven|eight|nine)?|(?:eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen))\b'

    # Dictionary to map words to their numerical values
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000, 'million': 1000000
    }

    # Function to convert words to their numerical values
    def words_to_numbers(match):
        words = match.group(1)
        words = words.strip()
        if '-' in words:
            # Handle hyphenated words
            word_list = words.split('-')
            # Check that each word in the hyphenated phrase is a valid number
            for word in word_list:
                if word.strip() not in word_to_num:
                    return ' ' + words
            return ' ' + str(sum(word_to_num[w] for w in word_list))
        elif ' ' in words:
            # Handle multiple words
            word_list = words.split(' ')
            # Check that each word in the phrase is a valid number
            for word in word_list:
                if word not in word_to_num:
                    return ' ' + words
            return ' ' + str(sum(word_to_num[w] for w in word_list))
        else:
            # Handle single words
            strip_words = words.strip()
            if strip_words not in word_to_num:
                return ' ' + words
            return ' ' + str(word_to_num[strip_words])

    # Replace words with their numerical values using regular expressions
    result = re.sub(pattern, words_to_numbers, text, flags=re.IGNORECASE)

    # Print the original text and the result
    return result.replace('  ', ' ')


def findNumericalValues(sentence):
    pattern = r'(?<![\w])\d+\.?\d*(?![\w\$])'
    numerical_values = re.findall(pattern, sentence)
    # print(numerical_values)
    return numerical_values

def get_match_reward(response_text, gold_response_text):
    import re
    pattern = r'\[(\w+)\]'
    generated_sequence = re.findall(pattern, response_text)
    gold_sequence = re.findall(pattern, gold_response_text)

    num_matching_ops = sum([1 for m_op, gt_op in zip(generated_sequence, gold_sequence) if m_op == gt_op])
    return (num_matching_ops / len(gold_sequence))

def calculate_answer_diff(generated_answer, gold_answer):
    absolute_difference = abs(gold_answer - generated_answer)
    scaled_difference = absolute_difference / max(gold_answer, generated_answer)
    reward = 1 - scaled_difference
    # reward = min(max(reward, -1), 1)
    return reward