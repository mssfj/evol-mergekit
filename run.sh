#!/bin/bash

# wandbのAPIキーを環境変数から取得
WANDB_API_KEY="${WANDB_API_KEY:-}"

# APIキーが設定されているか確認
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY is not set. Please set it in your environment."
    exit 1
fi

# run
mergekit-evolve ./eval_tasks/evol_merge_config.yaml \
--storage-path ./evol_merge_storage \
--task-search-path ./eval_tasks \
--wandb \
--merge-cuda \
--vllm \
#--in-memory \
