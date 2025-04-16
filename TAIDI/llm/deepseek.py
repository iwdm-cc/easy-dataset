# openai_module.py
import openai


def get_openai_response(prompt, content):
    # 确保已安装 OpenAI SDK: `pip3 install openai`
    print("正在调用 OpenAI API...")
    # 初始化 OpenAI API 客户端
    client = openai.OpenAI(api_key="sk-hzcmamnutosmpwgtfqsxwdgcxigqulmfchyruffvwmcasqio", base_url="https://api.siliconflow.cn/v1")

    # 创建聊天完成请求
    response = client.chat.completions.create(
        model="THUDM/GLM-Z1-9B-0414",  # 指定使用的模型
        messages=[  # 构建消息列表
            {"role": "system", "content": prompt},  # 系统角色消息，提供背景或提示词
            {"role": "user", "content": content},  # 用户角色消息，发送用户问题
        ],
        stream=False  # 是否使用流式响应（默认非流式）
    )


    # 返回生成的回复内容
    return response.choices[0].message.content

#
# def get_openai_response(prompt, content):
#     # 确保已安装 OpenAI SDK: `pip3 install openai`
#     print("正在调用 OpenAI API...")
#     # 初始化 OpenAI API 客户端
#     client = openai.OpenAI(api_key="sk-xAJ1IWaFvtiarnkR5e8b2fF2162746379254C328D411C8D7",
#                            base_url="https://chat.zju.edu.cn/api/ai/v1")
#
#     # 创建聊天完成请求
#     response = client.chat.completions.create(
#         model="deepseek-r1-671b",  # 指定使用的模型
#         messages=[  # 构建消息列表
#             {"role": "system", "content": prompt},  # 系统角色消息，提供背景或提示词
#             {"role": "user", "content": content},  # 用户角色消息，发送用户问题
#         ],
#         stream=False  # 是否使用流式响应（默认非流式）
#     )
#
#     # 返回生成的回复内容
#     return response.choices[0].message.content