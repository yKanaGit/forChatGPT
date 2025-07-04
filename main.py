# main.py
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import snapshot_download
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from mcp.server.fastmcp import FastMCP, Context

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

# MCP server setup
server = FastMCP()

@server.tool()
async def ask_local(prompt: str, ctx: Context) -> str:
    """Answer a question using the local model."""
    result = chain.invoke({"question": prompt})
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a simple Hugging Face model or start an MCP server"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run as an MCP server communicating over stdio",
    )
    args = parser.parse_args()

    if args.serve:
    else:
        q = input("質問を入力してください: ")
        response = chain.invoke({"question": q})
        print("AIの回答:", response)
