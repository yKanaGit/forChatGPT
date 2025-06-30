from mcp.server.fastmcp import FastMCP, Context
from langchain_openai import ChatOpenAI

server = FastMCP()

@server.tool()
async def ask_openai(prompt: str, ctx: Context) -> str:
    """Use an OpenAI chat model to answer a question."""
    llm = ChatOpenAI(temperature=0)
    result = await llm.ainvoke(prompt)
    return result.content

if __name__ == "__main__":
    # run the server using stdio transport so the client can spawn it
    server.run("stdio")
