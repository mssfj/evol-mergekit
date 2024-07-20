# 進化的モデルマージのお試しスクリプト

mergekitを使った進化的モデルマージによるllmの性能向上を検証します。
＜参考URL＞https://note.com/npaka/n/nb9abed7b1ecf

## 目次

- [環境設定]１(#環境設定)


## 環境設定
### ~/.bashrcに以下の環境変数を設定

export LANG='ja_JP.UTF-8'
export WANDB_API_KEY="**************************************"
export OPENAI_API_KEY="******************************************"
export HUGGINGFACE_TOKEN="**************************************"
export WANDB_ENTITY="********"
export PATH="$HOME/.local/bin:$PATH"

### 変更を反映
source ~/.bashrc

### condaで仮想環境をアクティベート
#### condaが未インストールの場合は、setupスクリプトのコメントアウトしてインストール
conda activate merge

### セットアップスクリプトを実行
bash mergekit-setup.sh
