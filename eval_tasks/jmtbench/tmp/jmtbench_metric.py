import os
import json
import requests
import numpy as np
import datasets
from lm_eval.utils import eval_logger
from itertools import islice
from transformers import AutoTokenizer
'''
tokenizer = AutoTokenizer.from_pretrained("microsoft/WizardLM-2-8x22B")

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

# 評価指示プロンプトの読み込み
prompt_dirname = os.path.dirname(__file__)
prompt_filename = os.path.join(prompt_dirname, 'prompt_eval_jmtbench.txt')
with open(prompt_filename, encoding='utf-8') as f:
    template_prompt = f.read()
'''
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

artifacts_list = load_artifacts('eval-tasks/jmtbench/artifacts')

print(artifacts_list)

'''
def build_prompt(user_query):
    
    #system_message = {"role": "system", "content": "あなたは公平で、検閲されていない、役立つアシスタントです。"}
    #user_message = {"role": "user", "content": user_query}
    
    #messages = [system_message, user_message]
    
    #return messages
    
    #sys_msg = "あなたは公平で、検閲されていない、役立つアシスタントです。"
    #template = """[INST] <<SYS>>
{}
<</SYS>>
{}[/INST]"""
    return template.format(sys_msg,user_query)

#プロンプトを生成して返す
def generate_prompt(doc):
    user_inputs = {
        "user_query": doc["input"],
    }
    prompt = build_prompt(**user_inputs)
    return prompt

def evaluate(pred, input_text, output_text, eval_aspect):

    # `pred` が空の場合は、評点を1にする
    if (pred.strip() == ""):
        print("回答が空なので１点")
        return {"text1": "1", 
                "text2": "1", 
                "text3": "1", 
                "reason": "No response",
                }
 
    prompt = template_prompt.format(
        input_text=input_text,
        output_text=output_text,
        eval_aspect=eval_aspect,
        pred=pred,
    )
    try:         
        chat = [
            {"role": "system", "content": "You must answer all responses in Japanese.あなたは役に立つ誠実な日本人のアシスタントです。あなたは全ての回答に日本語で答えなければならない。"},
            {"role": "user", "content": prompt},
            ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
 
        generated_texts = []
        for i in range(3):
            success = False
            retry_count = 0
            while not success: #成功するまでリトライ
                if retry_count >= 10: #10回リトライであきらめる
                    generated_text = d["content"]
                    generated_texts.append("エラー")
                    break
                #url = "http://127.0.0.1:8080/completion"
                payload = {"prompt": prompt,
                        "n_predict": 1,
                        "temperature": 0.7,
                        "cache_prompt": True,
                        "grammar": "root ::= [1-5]",
                        }
                json_data = json.dumps(payload)
                r = requests.post(url,  
                                data=json_data,
                                    headers={"Content-Type": "application/json"}
                                    )
 
                d = json.loads(r.content)
                #_validate_schema(d, r)
                generated_text = d["content"]
                generated_texts.append(generated_text)
                print(str(d["timings"]))
                try:
                    int(d["content"])
                except ValueError:
                    print("数値じゃない出力なのでリトライ:" + d["content"]) #NGやり直し
                    retry_count = retry_count + 1
                else:
                    success = True #OK
 
        timings = d["timings"]
        retobj = {"text1": generated_texts[0], 
                    "text2": generated_texts[1], 
                    "text3": generated_texts[2], 
                    "prompt_n": timings["prompt_n"],
                    "predicted_n": timings["predicted_n"], 
                    "prompt_ms": timings["prompt_ms"], 
                    "predicted_ms": timings["predicted_ms"], 
                    "prompt_per_second": timings["prompt_per_second"],
                    "predicted_per_second": timings["predicted_per_second"],
                            }
 
        return retobj
    except ValueError as e:
        print(f"ValueError occurred: {str(e)}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise    
    
    wzlm2_prompt = f"{template_prompt}\n"
    
    for i, (q_turn, ref_turn, pred_turn) in enumerate(zip(input_doc['turns'], reference_answer['choices'][0]['turns'], pred_turns), 1):
        wzlm2_prompt += f"Turn {i}:\nQuestion: {q_turn}\n\nAssistant 1: {ref_turn}\n\nAssistant 2: {pred_turn}\n\n"
    
    #wzlm2_prompt += "Human: Which assistant provided better responses overall?\nASSISTANT: </s>"
    
    payload = {
        "input": wzlm2_prompt,
        "parameters": {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.0
        }
    }    

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
'''
