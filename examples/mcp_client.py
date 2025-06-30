import anyio
import sys
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

async def main(question: str) -> None:
    params = StdioServerParameters(command=sys.executable, args=["examples/mcp_server.py"])
    async with stdio_client(params) as (read_stream, write_stream):
        session = ClientSession(read_stream, write_stream)
        await session.initialize()
        result = await session.call_tool("ask_openai", {"prompt": question})
        if result.isError:
            print("Error:", result.error)
        else:
            # content is a list of TextContent blocks
            if result.content:
                print(result.content[0].text)

anyio.run(main, "LangChainとは何ですか？")
