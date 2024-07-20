# mergekitによるevolve-mergeのお試しスクリプト 

# 参考URL
https://note.com/npaka/n/nb9abed7b1ecf
https://soysoftware.sakura.ne.jp/archives/3872

## 設定
### ~/.bashrcに各種アクセストークンを

export LANG='ja_JP.UTF-8'
export WANDB_API_KEY="**************************************"
export OPENAI_API_KEY="******************************************" #評価にGPT-4oを使う場合 
export HUGGINGFACE_TOKEN="**************************************"
export WANDB_ENTITY="********"
export PATH="$HOME/.local/bin:$PATH"

### bashに反映
source ~/.bashrc

### conda仮想環境をアクティベート
#### condaが未インストールの場合はセットアップスクリプトのコメントアウト
conda activate merge

### セットアップスクリプトの実行
bash mergekit-setup.sh
