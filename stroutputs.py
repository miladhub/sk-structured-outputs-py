import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent

from semantic_kernel.kernel_pydantic import KernelBaseModel

class Step(KernelBaseModel):
    explanation: str
    output: str

class Reasoning(KernelBaseModel):
    steps: list[Step]
    final_answer: str

kernel = Kernel()

service_id = "structured-output"
import os
chat_service = OpenAIChatCompletion(service_id=service_id, api_key=os.getenv("OPENAI_API_KEY"), ai_model_id='gpt-4o-mini')

kernel.add_service(chat_service)

system_message = """
You are a helpful math tutor. Guide the user through the solution step by step.
"""

req_settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
req_settings.max_tokens = 2000
req_settings.temperature = 0.7
req_settings.top_p = 0.8
req_settings.function_choice_behavior = FunctionChoiceBehavior.Auto(filters={"excluded_plugins": ["chat"]})
req_settings.response_format = Reasoning

history = ChatHistory()
history.add_user_message("how can I solve 8x + 7y = -23, and 4x=12?")

chat_function = kernel.add_function(
    prompt=system_message + """{{$chat_history}}""",
    function_name="chat",
    plugin_name="chat",
    prompt_execution_settings=req_settings,
)

async def main():
    stream = True
    if stream:
        answer = kernel.invoke_stream(
            chat_function,
            chat_history=history,
        )
        print("AI Tutor:> ", end="")
        result_content: list[StreamingChatMessageContent] = []
        async for message in answer:
            result_content.append(message[0])
            print(str(message[0]), end="", flush=True)
        if result_content:
            result = "".join([str(content) for content in result_content])
    else:
        result = await kernel.invoke(
            chat_function,
            chat_history=history,
        )
        print(f"AI Tutor:> {result}")
    history.add_assistant_message(str(result))

if __name__ == "__main__":
    asyncio.run(main())

