# 这是一个示例 Python 脚本。
#支持的MODEL列表
# Available   : 4
#   - deepseek-v3-250324
#   - deepseek-v3-2-251201
#   - glm-4-7-251222
#   - doubao-seed-2-0-pro-260215
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
from abc import ABC, abstractmethod
from enum import Enum
import os

import httpx
from openai import OpenAI, AsyncOpenAI

VOLCES_API_KEY = "REPLACE_WITH_ENV_VALUE"
VOLCES_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
VOLCES_DPSK_V3 = "deepseek-v3-2-251201"
# VOLCES_DPSK_V3 = "deepseek-v3-250324" #常规
# VOLCES_DPSK_V3 = "doubao-seed-2-0-mini-260428" #token额度不足自动扣费


VOLCES_DPSK_R1 = "ep-20250218150536-p4zwm"
MISTRAL_API_KEY = "REPLACE_WITH_ENV_VALUE"

src_file = "../GB∕T 33000-2025 大中型企业安全生产标准化管理体系目录.docx"
import requests


import json


# 2. 提取最内层文件名到一维数组
file_names = []


class AIModel(ABC):
    def __init__(self,URL,KEY,MODEL,INPUT_MAXSIZE):
        self.VOLCES_BASE_URL = URL
        self.VOLCES_API_KEY = KEY
        self.MODEL_NAME = MODEL
        self.INPUT_MAXSIZE = INPUT_MAXSIZE

    @abstractmethod
    async def callAI(self,prompt,printPrompt=False):
        pass
class DEEPSEEK_V3(AIModel):
     async def callAI(self,prompt,printPrompt=False):
        return await call_deepseek_v32(prompt,printPrompt)


class DEEPSEEK_V4(AIModel):
     async def callAI(self,prompt,printPrompt=False):
        return await call_deepseek_v42(prompt,printPrompt)



class AIModelEnum(Enum):
    DeepSeek_V3 = DEEPSEEK_V3(URL=VOLCES_BASE_URL,KEY=VOLCES_API_KEY,MODEL="deepseek-v3-2-251201",INPUT_MAXSIZE=100000)
    # DeepSeek_V4 = DEEPSEEK_V4(URL="https://api.deepseek.com",KEY="REPLACE_WITH_ENV_VALUE",MODEL="deepseek-v4-flash",INPUT_MAXSIZE=700000)
    DeepSeek_V4 = DEEPSEEK_V4(URL="https://ark.cn-beijing.volces.com/api/v3",KEY="XXXXXX",MODEL="deepseek-v4-1-flash-260910",INPUT_MAXSIZE=100000)
    # v4 flash 正式版
    DeepSeek_V4_FLASH_OFFICAL = DEEPSEEK_V4(URL="https://ark.cn-beijing.volces.com/api/v3",KEY="XXXXX",MODEL="deepseek-v4-flash-ga-260731",INPUT_MAXSIZE=100000)


# VOLCES_DPSK_R1 = "ep-20250218150536-p4zwm"  # 备用：DeepSeek-R1部署ID

# 1. 最简调用（DeepSeek-V3，完整返回）
def call_deepseek_v3(prompt,printPrompt=False,printResponse=True):
    # 请求地址（火山方舟的chat/completions接口）
    print("\n")
    if printPrompt:
        print("我的提问:"+prompt)
    else:
        print(f"==V3 发起询问==输入长度：{len(prompt)}")
    url = f"{VOLCES_BASE_URL}/chat/completions"

    # 请求参数（适配DeepSeek-V3）
    payload = {
        "model": VOLCES_DPSK_V3,  # 核心：替换为你的DeepSeek-V3部署ID
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,  # 随机性
        "max_tokens": 16384,  # 最大返回长度（DeepSeek-V3支持更长上下文）
        "stream": False  # 关闭流式，完整返回
    }

    # 请求头（火山方舟授权格式）
    headers = {
        "Authorization": f"Bearer {VOLCES_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # 抛出HTTP错误
        result = response.json()

        # 提取回复内容
        if printResponse:
            print(result)
        answer = result["choices"][0]["message"]["content"]
        if printResponse:
            print("DeepSeek-V3回复：\n", answer)
        return answer

    except requests.exceptions.RequestException as e:
        print(f"请求失败：{e}")
        if response:
            print(f"错误详情：{response.json()}")
        return None
# 异步调用 DeepSeek-V3
async def call_deepseek_v32(prompt,printPrompt=False):
    if printPrompt:
        print("我的提问:"+prompt)
    else:
        print(f"==V3 发起询问==输入长度：{len(prompt)}")

    url = f"{VOLCES_BASE_URL}/chat/completions"
    payload = {
        "model": VOLCES_DPSK_V3,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 16384,
        "stream": False,
         "extra_body":{"thinking": {"type": "disabled"}},
    }
    headers = {
        "Authorization": f"Bearer {VOLCES_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:  # 异步客户端
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            print("DeepSeek-V3回复：\n", answer)
            return answer
        except httpx.HTTPStatusError as e:
            print(f"请求失败：{e}")
            if response:
                print(f"错误详情：{response.json()}")
            return None
def call_deepseek_v4(prompt,printPrompt=False,printResponse=True):
    if printPrompt:
        print("我的提问:"+prompt)
    else:
        print(f"=={AIModelEnum.DeepSeek_V4.value.MODEL_NAME} 发起询问==输入长度：{len(prompt)}")
    client = OpenAI(
        api_key=AIModelEnum.DeepSeek_V4.value.VOLCES_API_KEY,
        base_url=AIModelEnum.DeepSeek_V4.value.VOLCES_BASE_URL)
    try:
        response = client.chat.completions.create(
        model=AIModelEnum.DeepSeek_V4.value.MODEL_NAME,
        messages=[
            {"role": "user", "content":prompt},
        ],
        stream=False,
        max_tokens=16384,  # 取消注释：避免长脚本 JSON 被默认上限截断导致解析失败
        # reasoning_effort="high", #思考深度
        extra_body={"thinking": {"type": "disabled"}}
        )
        answer = response.choices[0].message.content
        if printResponse:
            print(f"DeepSeek-V4回复 输出长度:{len(answer or '')}：\n", answer)
        return answer
    except Exception as e:
        print(f"请求失败：{e}")
        return None

 # 异步调用 DeepSeek-V4
async def call_deepseek_v42(prompt,printPrompt=False):
    if printPrompt:
        print("我的提问:"+prompt)
    else:
        print(f"==V4 发起询问==输入长度：{len(prompt)}")

    client = AsyncOpenAI(  # 异步 OpenAI 客户端
        api_key=AIModelEnum.DeepSeek_V4.value.VOLCES_API_KEY,
        base_url=AIModelEnum.DeepSeek_V4.value.VOLCES_BASE_URL
    )
    try:
        response = await client.chat.completions.create(  # await 异步调用
            model=AIModelEnum.DeepSeek_V4.value.MODEL_NAME,
            messages=[{"role": "user", "content":prompt}],
            stream=False
        )
        answer = response.choices[0].message.content
        print(f"DeepSeek-V4回复 输出长度:{len(answer)}：\n", answer)
        return answer
    except Exception as e:
        print(f"请求失败：{e}")
        return None

async def test():
    answer  = await call_deepseek_v42("""
    今天厦门天气如何
""")
    # 2 从AI 提取文件内容

    print(answer)

if __name__ == "__main__":

    # 1 提取文件名
    import asyncio
    asyncio.run(test())




    # 3 markdown 转docx
    # if jsonResult != None:
    #     for key,value in jsonResult.items():
    #         md2docx.md_str_to_docx(jsonResult[key],f"files/{key}.docx".replace(".md",""))
    # else:print("JSON错误")
