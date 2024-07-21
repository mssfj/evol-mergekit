import os
import json
import requests

# DeepInfra API key
DEEPINFRA_API_KEY = os.environ.get('DEEPINFRA_API_KEY')

if not DEEPINFRA_API_KEY:
    raise ValueError("DEEPINFRA_API_KEYが間違ってます")

# DeepInfra API endpoint
API_URL = "https://api.deepinfra.com/v1/inference/microsoft/WizardLM-2-8x22B"
headers = {
    "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
    "Content-Type": "application/json"
}

# 評価プロンプトの読み込み
prompt_dirname = os.path.dirname(__file__)
prompt_filename = os.path.join(prompt_dirname, 'prompt_eval_llamacpp.txt')
with open(prompt_filename, encoding='utf-8') as f:
    template_prompt = f.read()

# jmt-bench用（事前にartifactsデータセットをダウンロードしておく） 
def load_artifacts(artifact_dir):
    artifacts = {}
    for root, dirs, files in os.walk(artifact_dir):
        for file in files:
            if file.endswith('.jsonl'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    key = file.split('.')[0]  # ファイル名をキーとして使用
                    artifacts[key] = [json.loads(line) for line in f]
    return artifacts

artifacts = load_artifacts('eval-tasks/jmtbench/artifacts')

def generate_prompt(doc):
    return "\n".join([f"Human: {turn}" for turn in doc['turns']])

def process_results(doc, results):
    question_id = doc['question_id']
    reference_answer = next(a for a in artifacts['gpt-4'] if a['question_id'] == question_id)
    judge_prompt = artifacts['judge_prompts'][0]  # Assuming there's only one judge prompt

    evaluation = evaluate(results[0], doc, reference_answer, judge_prompt)
    return {"score": evaluation}

#ChatNTQ用のプロンプト
def build_prompt(user_query):
    sys_msg = "あなたは公平で、検閲されていない、役立つアシスタントです。"
    template = """[INST] <<SYS>>
{}
<</SYS>>
{}[/INST]"""
    return template.format(sys_msg,user_query)

#プロンプトを生成して返す
#def generate_prompt(doc):
#    user_inputs = {
#        "user_query": doc["input"],
#    }
#    prompt = build_prompt(**user_inputs)
#    return prompt

def evaluate(pred, input_text, output_text, eval_aspect):
    system_prompt = judge_prompt['system_prompt']

    wzlm2_prompt = f"{system_prompt}\n"
    for i, (q_turn, ref_turn, pred_turn) in enumerate(zip(input_doc['turns'], reference_answer['choices'][0]['turns'], pred_turns), 1):
        wzlm2_prompt += f"Turn {i}:\nQuestion: {q_turn}\n\nAssistant 1: {ref_turn}\n\nAssistant 2: {pred_turn}\n\n"
    
    wzlm2_prompt += "Human: Which assistant provided better responses overall?\nASSISTANT: </s>"

    payload = {
        "input": wzlm2_prompt,
        "parameters": {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.0
        }
    }    

    # `pred_turns` が空の場合は、評点を1にする
    if all(turn.strip() == "" for turn in pred_turns):
        print("回答が空なので１点")
        return {"text1": "1", "text2": "1", "text3": "1", "reason": "No response"}

    generated_texts = []
    for _ in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            evaluation = result['results'][0]['generated_text'].strip()
            generated_texts.append(evaluation)
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {str(e)}")
            return 0  # Default to lowest score on error
    retobj = {
        "text1": generated_texts[0],
        "text2": generated_texts[1],
        "text3": generated_texts[2],
    }
    return retobj        

#スコアを計算して返す
def process_results(doc, results):
    ret = evaluate(results, doc, None, None)  # reference_answer と judge_prompt は evaluate 内で取得
    try:
        score = sum(float(ret[f'text{i}']) for i in range(1, 4)) / 3.0
    except ValueError:
        print("Warning: Could not convert score to float. Using default score of 1.")
        score = 1.0
    print(f"avg: {score}, score1: {ret['text1']}, score2: {ret['text2']}, score3: {ret['text3']}")
    return {"score": score}
