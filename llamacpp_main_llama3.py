import json
import os
import time
import jsonlines
import llamacpp_judge_llama3
import cohere
import subprocess
import requests

# Function to read data from a JSONL file
def read_jsonl(file_path):
    with jsonlines.open(file_path) as reader:
        return [obj for obj in reader]

def call_evaluate_with_error_handling(pred, input_text, output_text, eval_aspect):
    try:
        result = llamacpp_judge_llama3.evaluate(pred, input_text, output_text, eval_aspect)
        return result
    except cohere.errors.too_many_requests_error.TooManyRequestsError as e:
        print(f"TooManyRequestsError occurred after all retries: {str(e)}")
        return {"reason": "API rate limit exceeded\n" + str(e), "grade": 1}
    except ValueError as e:
        print(f"ValueError occurred after all retries: {str(e)}")
        return {"reason": "Invalid API response\n" + str(e), "grade": 1}
    except Exception as e:
        print(f"An unexpected error occurred after all retries: {str(e)}")
        return {"reason": "Unexpected error\n" + str(e), "grade": 1}

def evaluate(dir_name, slot_save_path, save_kvcache, restore_kvcache, force_save_kvcache): #1モデルの評価 Dir_name:評価対象のディレクトリ名
    # Load dataset
    base_directory = "L:\GitHubProjects\gpt4-autoeval\content\eval"
    #model_name = "command_r_iq2_xs" #これをモデル毎に変える
    dataset = read_jsonl("L:\GitHubProjects\gpt4-autoeval\content\dataset.jsonl")
    preds = read_jsonl(os.path.join( base_directory , dir_name, "preds.jsonl"))

    kvcache_basename = "elyzatasks"

    with jsonlines.open(os.path.join( base_directory , dir_name, 'result_llama2.jsonl'), mode='w') as writer:
        # Evaluate each sample of the dataset, and write the result to the file
        for i, (eval_data, pred_data) in enumerate(zip(dataset, preds)):

            pred = pred_data["pred"]
            input_text = eval_data["input_text"]
            output_text = eval_data["output_text"]
            eval_aspect = eval_data["eval_aspect"]

            
            #kvキャッシュ名
            kvcache_name = kvcache_basename + format(i, "03") + ".bin"
            kvcache_path = os.path.join( slot_save_path , kvcache_name)
            #kvキャッシュがあれば読み込む
            if restore_kvcache and os.path.isfile(kvcache_path):
                url = "http://127.0.0.1:8080/slots/0?action=restore"
                payload = {"filename": kvcache_name}
                json_data = json.dumps(payload)
                r = requests.post(url,  
                                data=json_data,
                                    headers={"Content-Type": "application/json"}
                                    )
                print("kvcache restored:")
                print(json.loads(r.content))

            print("--------------------------------------------------------------------------------\n" + "設問" + str(i) + "--------------------------------------------------------------------------------")
            result = call_evaluate_with_error_handling(pred, input_text, output_text, eval_aspect)
            writer.write(result)

            #kvキャッシュが無ければ保存
            if save_kvcache and (not os.path.isfile(kvcache_path) or force_save_kvcache ):
                url = "http://127.0.0.1:8080/slots/0?action=save"
                payload = {"filename": kvcache_name}
                json_data = json.dumps(payload)
                r = requests.post(url,  
                                data=json_data,
                                    headers={"Content-Type": "application/json"}
                                    )
                print("kvcache saved:")
                print(json.loads(r.content))

            print(f"==============================")
            print(f"Q. {input_text}")
            print(f"A. {pred}")
            print(f"Llama3. {result}")
            print(f"")

save_kvcache = True
force_save_kvcache = False
restore_kvcache = True
launch_server = False
slot_save_path = "I:\llama\kvcache\elyzatasks_Llama-3-70B-Instruct_Q5_K_M"

command = ["I:\llama\llamacpp\server.exe" , 
           "-m", "I:\llama\llamacpp\models\Meta-Llama-3-70B-Instruct.Q5_K_M.gguf", 
           "-c", "2048", 
           "-ngl", "31",
           "--slot-save-path", slot_save_path,
           ]

def get_folder_names(directory):
    folder_names = []
    
    # ディレクトリ内のアイテムを反復処理
    for item in os.listdir(directory):
        # アイテムがディレクトリ（フォルダ）であるかを確認
        if os.path.isdir(os.path.join(directory, item)):
            folder_names.append(item)
    
    return folder_names

eval_base_dir = "L:\GitHubProjects\gpt4-autoeval\content\eval";
eval_dirs = get_folder_names(eval_base_dir)
# フォルダ名を表示
print("フォルダ名のリスト:")
for folder in eval_dirs:
    print(folder)

start = time.time()
if launch_server:
    # サーバーを立ち上げる
    server_process = subprocess.Popen(command)
    # サーバーが立ち上がるまで待機する（適宜調整）
    time.sleep(5)
try:
    for eval_dir in eval_dirs:
        evaluate(eval_dir, slot_save_path, save_kvcache, restore_kvcache, force_save_kvcache)
finally:
    if launch_server:
        # 作業が終わったらサーバーを終了させる
        server_process.terminate()
    end = time.time()
    print("total time:" + str(end -start))
