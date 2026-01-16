import re
try:
    from . import prime_math
except:
    import prime_math



def first_option_postprocess(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        f'[Tt]he correct answer is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct answer option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct answer is:?.*?\\(([{options}])\\)', # match: "The correct answer is (A) Materials used in product: Direct Materials "
        f'boxed\{{([{options}])\}}',  # boxed{([A-D])}
        f'boxed\{{[^a-zA-Z0-9]*([{options}])[^a-zA-Z0-9]*\}}',  # boxed with any non-alphanum chars around
        f'boxed\{{([{options}])[^a-zA-Z0-9].*?\}}',  # boxed{A. ...} or boxed{A: ...} etc.
        f'boxed\{{\(([{options}])\)[^a-zA-Z0-9].*?\}}',  # boxed{(A) ...}
        f'boxed\{{\[([{options}])\][^a-zA-Z0-9].*?\}}',  # boxed{[A] ...}
        f'boxed\{{([{options}])\}}',  # boxed{A}
        f'<answer>\s*\(?([{options}])[\s\.\)]',
        f'<answer>\s*([{options}])\s*</answer>', # '<answer>C</answer>'
        f'(?i)ANSWER\s*:\s*([{options}])',
        f'(?i)ANSWER\s*:\s*\(([{options}])\)',
        f'答案是?\s*([{options}])',
        f'答案是?\s*：\s*([{options}])',
        f'答案是?\s*:\s*([{options}])',
        f'答案是选项?\s*:\s*([{options}])',
        f'答案选项应?该?为\s*([{options}])',
        f'答案选项应?该?是\s*([{options}])',
        f'答案应该?是\s*([{options}])',
        f'答案应该?选\s*([{options}])',
        f'答案选项为?\s*：\s*([{options}])',
        f'答案选项为?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        f'选项为?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        f'答案选项是?\s*:\s*([{options}])',
        f'答案为\s*([{options}])',
        f'答案选\s*([{options}])',
        f'选择?\s*([{options}])',
        f'故选?\s*([{options}])',
        f'只有选?项?\s?([{options}])\s?是?对',
        f'只有选?项?\s?([{options}])\s?是?错',
        f'只有选?项?\s?([{options}])\s?不?正确',
        f'只有选?项?\s?([{options}])\s?错误',
        f'说法不?对选?项?的?是\s?([{options}])',
        f'说法不?正确选?项?的?是\s?([{options}])',
        f'说法错误选?项?的?是\s?([{options}])',
        f'([{options}])\s?是正确的',
        f'([{options}])\s?是正确答案',
        f'选项\s?([{options}])\s?正确',
        f'所以答\s?([{options}])',
        f'所以\s?([{options}][.。$]?$)',
        f'所有\s?([{options}][.。$]?$)',
        f'[\s，：:,]([{options}])[。，,\.]?$',
        f'[\s，,：:][故即]([{options}])[。\.]?$',
        f'[\s，,：:]因此([{options}])[。\.]?$',
        f'[是为。]\s?([{options}])[。\.]?$',
        f'因此\s?([{options}])[。\.]?$',
        f'显然\s?([{options}])[。\.]?$',
        f'答案是\s?(\S+)(?:。|$)',
        f'答案应该是\s?(\S+)(?:。|$)',
        f'答案为\s?(\S+)(?:。|$)',
        f'[Tt]he answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        f'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        f'[Tt]he final answer is:?\s+\(?([{options}])\)?',
        f'^选项\s?([{options}])',
        f'^([{options}])\s?选?项',
        # f'\s[{options}][\s。，,：:\.$]',
        f'^[{options}][\s。，,：:\.$]',
        f'答案:\s*([{options}])',
        f'故此为\s*([{options}])',
    ]
    cushion_patterns = [
    ]
    
    # flake8: noqa
    # yapf: enable
    text = text.replace("\n\nAssistant:", "")
    text = text.strip()
    text = text.replace("**", "")
    
    if cushion:
        patterns.extend(cushion_patterns)
    if not text:
        return ''
    for pattern in patterns:
        text = text.strip()
        # match = re.search(pattern, text, re.DOTALL)
        match = re.findall(pattern, text, re.DOTALL)
        
        if match:
            if match[-1] is not None and match[-1] != '':
                outputs = match[-1]
            else:
                outputs = ''
            for i in options: # 'ABCD' BC => B CD => C
                if i in outputs: # 5.0 \\text{ J/K}
                    return i
    return 'not-macthed'



def match_answer_pattern(response_text: str, answer_pattern: str):
    match = re.findall(answer_pattern, response_text, re.DOTALL)
    extracted_answer = match[-1] if match else ''
    return extracted_answer


def extract_last_answer(text):
    answers = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    
    if answers:
        # Return the last one found
        if answers[-1].strip() == 'answer here':
            return text
        else:
            return answers[-1]
    else:
        return ""

def remove_think_tags(input_text):
    # This pattern matches <think> followed by any content (non-greedy) until </think>
    pattern = r'<think>.*?</think>'
    # Substitute all occurrences of the pattern with an empty string
    cleaned_text = re.sub(pattern, '', input_text, flags=re.DOTALL)
    return cleaned_text

import re

def extract_option(prompt_str, ground_truth):
    # Define regex pattern to match the option line
    for keyword in ['Options:\n', 'Choices:\n']:
        if keyword in prompt_str:
            prompt_str = prompt_str.split(keyword)[1]
    # Handles formats like:
    # A. text
    # A) text
    # (A) text
    # A text
    pattern = re.compile(
        rf'^{ground_truth}[.)\s]*\s*(\(?{ground_truth}\)?[.)\s]*)?\s*(.*)$',
        re.IGNORECASE | re.MULTILINE
    )
    
    # Search for the option in the prompt
    match = pattern.search(prompt_str)
    if match:
        res = match.group(2).replace('<|im_end|>', '').strip()
        return res
    
    # If not found, try alternative patterns
    alt_patterns = [
        rf'^\s*{ground_truth}\s*[.)]\s*(.*)$',  # A. text
        rf'^\s*{ground_truth}\s*\)\s*(.*)$',    # A) text
        rf'^\s*\({ground_truth}\)\s*(.*)$',      # (A) text
        rf'^\s*{ground_truth}\s+(.*)$',          # A text
    ]
    
    for alt_pattern in alt_patterns:
        match = re.search(alt_pattern, prompt_str, re.IGNORECASE | re.MULTILINE)
        if match:
            res = match.group(1).replace('<|im_end|>', '').strip()
            return res
    
    return 'None'  # Return None if option not found

def direct_match_mcq(output, gt_option, gt_full='None'):
    assert gt_option in 'ABCDEFGHIJKLMNOP'
    
    end = r')[\s\.\}]'
    patterns = [
        r'boxed\{\s*(' + re.escape(gt_option) + end,
        r'boxed\{\s*(' + re.escape(gt_full) + end, 

        r'boxed\{\s*' + re.escape('\\text{')+ '(' + re.escape(gt_option) +  r')[\s\.\}]', # '\\boxed{\\text{A}}', '\\boxed{\\text{A.}}', '\\boxed{\\text{A   }}'

        r'<answer>\s*(' + re.escape(gt_option) + r')\.?\s*</answer>',
        r'<answer>\s*(' + re.escape(gt_full) + r')\s*</answer>',
        r'<answer>\s*(' + re.escape(gt_full) + r')[\s\.]', # to capture the case when '''/think>\n<answer>A. 9.1445 x 10^-27 J T-1</answer>''' if the GT is "A".

        r'ANSWER:\s*(' +  re.escape(gt_option) + r')',
        r'Answer:\s*(' +  re.escape(gt_option) + r')',
        r'he answer to the question is:?\s*(' + re.escape(gt_option) + r')',
        r'he correct option is:?\s*(' + re.escape(gt_option) + r')',
        r'he correct answer is:?\s*(' + re.escape(gt_option) + r')',
        r'he answer is:?\s*(' + re.escape(gt_option) + r')[\s\.]',
        r'he correct answer is:?\s*\((' + re.escape(gt_option) + r')\)',
        r'he correct answer would be option:?\s*(' + re.escape(gt_option) + r')',
        r'Option (' + re.escape(gt_option) + r') therefore correctly asserts',
        r'Answer choice closest to our calculated value:?\s*(' + re.escape(gt_option) + r')',
        r"Here's an answer to your multiple-choice question:?\s*(" + re.escape(gt_option) + r')',
    ]
    for pattern in patterns:
        try:
            match = re.search(pattern, output, re.DOTALL)
            if match and match.group(1) is not None and match.group(1) != '':
                return 1,  match.group(1)
        except re.error:
            print(f'\n\n\n{pattern}')
            print(output)            
            
    match = re.findall(r'boxed\{\s*(.*?)\}', output, re.DOTALL)
    if match:
        if match[-1] is not None and match[-1] != '':
            outputs = match[-1]            
            res = prime_math.compute_score(outputs, gt_full)
            return res
            # if prime match: return 1, outputs
    return None

def compute_score(model_output: str, ground_truth: str, options: str, prompt_str: str = None) -> bool:
    model_output = str(model_output)
    ground_truth = str(ground_truth).strip()

    if 'user\n' in model_output:
        model_output = model_output.split('user\n')[0] # avoid the issue that base model appends additional conversation

    direct_match_result = direct_match_mcq(model_output, gt_option=ground_truth, gt_full=extract_option(prompt_str, ground_truth))
    if direct_match_result and direct_match_result[0]:
        return direct_match_result

    extracted_answer = first_option_postprocess(model_output, options=options)
    extracted_answer = extracted_answer.strip()
    ground_truth = ground_truth.strip()

    if extracted_answer == ground_truth:
        return 1, extracted_answer
    else:
        return 0, extracted_answer
    
    