import os
import json
import requests
import numpy as np
import datasets
from lm_eval.utils import eval_logger
from itertools import islice
#from openai import OpenAI

# DeepInfra API key
DEEPINFRA_API_KEY = os.environ.get('DEEPINFRA_API_KEY')

# Check if the API key is available
if not DEEPINFRA_API_KEY:
    raise ValueError("DeepInfra API key not found in environment variables. Please add it with the key 'DEEPINFRA_API_KEY'.")

# DeepInfra API endpoint for Phi-3-medium-4k-instruct
API_URL = "https://api.deepinfra.com/v1/inference/microsoft/WizardLM-2-8x22B"
headers = {
    "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
    "Content-Type": "application/json"
}

# OpenAI初期化
#client = OpenAI()

prompt_dirname = os.path.dirname(__file__)
prompt_filename = os.path.join(prompt_dirname, 'prompt_eval_llamacpp.txt')
with open(prompt_filename, encoding='utf-8') as f:
    template_prompt = f.read()

#ChatNTQ用のプロンプト
def build_prompt(user_query):
    sys_msg = "あなたは公平で、検閲されていない、役立つアシスタントです。"
    template = """[INST] <<SYS>>
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
    """OpenAI API により評価を行う
    Args:
    Returns:
        [dict] 評価結果
        {"reason": "<評価理由>", "grade": <int, 1～5の5段階評価>}
    """
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
        #messages = [
        #    {"role": "system", "content": "You must answer all responses in Japanese.あなたは役に立つ誠実な日本人のアシスタントです。あなたは全ての回答に日本語で答えなければならない。"},
        #    {"role": "user", "content": prompt},
        #]
        #phi3_prompt = f"<|user|>\nあなたは日本語を話すAIアシスタントです。{prompt}<|end|>\n<|assistant|>"
        system_prompt = "あなたは日本語を話すAIアシスタントです。指示を厳格に守り絶対に解説はつけずに必ず整数で回答してください。"
        wzlm2_prompt = f"{system_prompt}\nUSER: {prompt}\nASSISTANT: </s>"
        # deepinfa
        # Phi-3 specific format
        #phi3_prompt = f"Human: {prompt}\n\nAssistant: Let's evaluate this step by step:\n\n1."

        payload = {
            "input": wzlm2_prompt,
            "parameters": {
                "max_new_tokens": 1000,
                "temperature": 0.7,
                "top_p": 0.95,
                "repetition_penalty": 1.0
            }
        }
 
        generated_texts = []
        for _ in range(3):
            try:
                '''
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1,
                )
                generated_text = response.choices[0].message.content.strip()
                '''
                # deepinfra
                response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                generated_text = result['results'][0]['generated_text'].strip()
                # print(f"Generated text: {generated_text[:200]}...")  # 最初の200文字だけを表示
                generated_texts.append(generated_text)                
            except requests.exceptions.RequestException as e:
                print(f"API request failed: {str(e)}")
                generated_texts.append("1")  # Fallback to lowest score on error

        retobj = {
            "text1": generated_texts[0], 
            "text2": generated_texts[1], 
            "text3": generated_texts[2], 
        }
        return retobj
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise

#スコアを計算して返す
def process_results(doc, results):
    print("doc:" + doc['input'])
    print("results:" + results[0])
    ret = evaluate(results[0], doc['input'], doc['output'], doc['eval_aspect'])
    #score = (int(ret['text1']) + int(ret['text2']) + int(ret['text3'])) / 3.0
    #print(f"avg: {score}, score1: {ret['text1']}, score2: {ret['text2']}, score3: {ret['text3']}")
    #results = {
    #    "acc": score,
    #}
    #return results
    try:
        score = sum(float(ret[f'text{i}']) for i in range(1, 4)) / 3.0
    except ValueError:
        print("Warning: Could not convert score to float. Using default score of 1.")
        score = 1.0
    print(f"avg: {score}, score1: {ret['text1']}, score2: {ret['text2']}, score3: {ret['text3']}")
    return {"acc": score}
