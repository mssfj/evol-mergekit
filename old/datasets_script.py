from datasets import load_dataset, Dataset
import pandas
import os
from huggingface_hub import login

# 3. Hugging Face認証（必要な場合）
token = os.getenv("HUGGINGFACE_TOKEN")
login(token=token)

# 4. メイン処理
if __name__ == "__main__":

    # (1) spartqa-mchoiceデータセットの準備
    ds = load_dataset("metaeval/spartqa-mchoice")["train"]
    ds_p = ds.shuffle(seed=9163).select(range(1000))
    ds_p.push_to_hub(f"{os.environ['HUGGINGFACE_USERNAME']}/spartqa-train-1k", private=True)

    # (3) vicgalle/alpaca-gpt4データセットの準備。
    ds = load_dataset("vicgalle/alpaca-gpt4")["train"]
    df = ds.to_pandas()

    no_input = df[df.input.map(len) < 1]
    examples = no_input.sample(n=500, replace=False, random_state=749)
    ds_p = Dataset.from_pandas(examples)
    ds_p.push_to_hub(f"{os.environ['HUGGINGFACE_USERNAME']}/alpaca-gpt4-500", private=True)
