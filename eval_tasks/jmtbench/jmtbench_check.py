import wandb
import os

# wandbにログイン（初回のみ必要）
wandb.login()

# Wandbプロジェクトを初期化
wandb.init(project="mtbench-data-check", job_type="data_inspection")

# アーティファクトのパス
paths = [
    'wandb-japan/llm-leaderboard/mtbench_ja_question:v0',
    'wandb-japan/llm-leaderboard/mtbench_ja_referenceanswer:v0',
    'wandb-japan/llm-leaderboard/mtbench_ja_prompt:v1'
]

try:
    for path in paths:
        print(f"Checking artifact: {path}")
        
        # アーティファクトを取得
        artifact = wandb.use_artifact(path)
        
        # アーティファクトをダウンロード
        artifact_dir = artifact.download()
        
        # ダウンロードしたファイルの一覧を表示
        print("Files in the artifact:")
        for root, dirs, files in os.walk(artifact_dir):
            for file in files:
                file_path = os.path.join(root, file)
                print(f" - {file_path}")
        
                # ファイルの内容を表示（例：最初の500文字）
                print(f"\nContent preview of {file}:")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        print(f.read(500))  # 最初の500文字を表示
                except UnicodeDecodeError:
                    print("Unable to read file contents (possibly binary file)")
        
        print("\n" + "="*50 + "\n")

finally:
    # Wandbのランを終了
    wandb.finish()
