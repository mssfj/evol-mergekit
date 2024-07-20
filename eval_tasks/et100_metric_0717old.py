import os
import json
import requests
import numpy as np
import datasets
from lm_eval.utils import eval_logger
from itertools import islice
from openai import OpenAI

prompt_dirname = os.path.dirname(__file__)
prompt_filename = os.path.join(prompt_dirname, 'prompt_eval_llamacpp.txt')
with open(prompt_filename, encoding='utf-8') as f:
    template_prompt = f.read()

# OpenAI初期化
client = OpenAI()

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
        chat = [
            {"role": "system", "content": "You must answer all responses in Japanese.あなたは役に立つ誠実な日本人のアシスタントです。あなたは全ての回答に日本語で答えなければならない。"},
            {"role": "user", "content": prompt},
            ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        generated_texts = []
        for _ in range(3):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=1,
            )
            generated_text = response.choices[0].message.content.strip()
            generated_texts.append(generated_text)                
        
        retobj = {
            "text1": generated_texts[0], 
            "text2": generated_texts[1], 
            "text3": generated_texts[2], 
        }

        return retobj
    except ValueError as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise

#スコアを計算して返す
def process_results(doc, results):
    
    print("doc:" + doc['input'])
    print("results:" + results[0])

    ret =evaluate(results[0], doc['input'], doc['output'], doc['eval_aspect'] )

    score = (int(ret['text1']) + int(ret['text2']) + int(ret['text3'])) /3.0

    print(f"avg: {score}, score1: {ret['text1']}, score2: {ret['text2']}, score3: {ret['text3']}")

    results = {
        "acc": score,
    }
    return results
