import os
from openai import OpenAI

# 从环境变量读取 API Key，安全
client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key="UR1QANJ6WRPSMXAYYJCDP3OSEGI6LMMK0RCQ4E19",
    default_headers={"X-Failover-Enabled": "true"},
)

print("Gitee AI(输入 quit 或 exit 退出）\n")

while True:
    user_input = input("user: ")
    if user_input.lower() in ["quit", "exit"]:
        break

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful and harmless assistant. You should think step-by-step."},
            {"role": "user", "content": user_input}
        ],
        model="GLM-4_5",
        stream=True,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.7,
        extra_body={"top_k": 50},
        frequency_penalty=1,
    )

    print("AI: ", end="", flush=True)
    for chunk in response:
        if len(chunk.choices) == 0:
            continue
        delta = chunk.choices[0].delta
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            # 灰色打印思考过程（可选）
            print(f"\033[90m{delta.reasoning_content}\033[0m", end="", flush=True)
        elif delta.content:
            print(delta.content, end="", flush=True)
    print("\n")