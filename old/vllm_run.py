import sys
from vllm import LLM, SamplingParams

def main():
    # モデルの読み込み
    print("モデルを読み込んでいます...")
    model = LLM(
        model="Sreenington/Phi-3-mini-4k-instruct-AWQ",
        quantization="awq",
        dtype="float16",
        gpu_memory_utilization=0.9
    )
    print("モデルの準備ができました。プロンプトを入力してください（終了するには 'quit' と入力）:")

    # サンプリングパラメータの設定
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)

    while True:
        # 標準入力からプロンプトを読み込む
        prompt = input("> ")
        if prompt.lower() == 'quit':
            break

        # 推論の実行
        try:
            outputs = model.generate([prompt], sampling_params)
            print("\n生成された応答:")
            print(outputs[0].outputs[0].text)
            print("\n")
        except Exception as e:
            print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
