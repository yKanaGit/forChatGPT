# main.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import snapshot_download
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# モデルのダウンロード（キャッシュに保存）
model_path = snapshot_download(
    repo_id="bigscience/bloom-560m",
    use_auth_token=True
)

# トークナイザーとモデルのロード
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Transformersパイプライン作成
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
)

# LangChain向けラッパー
llm = HuggingFacePipeline(pipeline=pipe)

# プロンプトテンプレート
prompt = PromptTemplate(
    template="質問に答えてください: {question}",
    input_variables=["question"],
)

# RunnableSequence構成（プロンプト→LLM）
chain: RunnableSequence = prompt | llm

if __name__ == "__main__":
    # 質問を入力
    q = input("質問を入力してください: ")
    # 実行
    response = chain.invoke({"question": q})
    # 結果表示
    print("AIの回答:", response)