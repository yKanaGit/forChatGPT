# main.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import snapshot_download
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import anyio

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


async def run_mcp(question: str) -> str:
    """Call the MCP example server to answer the question."""
    params = StdioServerParameters(command=sys.executable, args=["examples/mcp_server.py"])
    async with stdio_client(params) as (read_stream, write_stream):
        session = ClientSession(read_stream, write_stream)
        await session.initialize()
        result = await session.call_tool("ask_openai", {"prompt": question})
        if result.isError:
            return f"Error: {result.error}"
        return result.content[0].text if result.content else ""

if __name__ == "__main__":
    # 質問を入力
    q = input("質問を入力してください: ")
    # LangChain パイプラインの実行
    response = chain.invoke({"question": q})
    print("AIの回答:", response)

    # MCP 経由でも同じ質問を投げる例
    print("\n--- MCP 経由の回答例 ---")
    mcp_response = anyio.run(run_mcp, q)
    print(mcp_response)
