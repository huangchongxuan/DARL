#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM ChatCompletion API 高级客户端
提供会话管理和交互模式
"""

import argparse
import json
import requests
import time
import openai
import traceback
import os
from typing import Dict, Any, List, Optional


class ChatSession:
    """聊天会话类，管理对话历史"""

    def __init__(self, system_message: Optional[str] = None):
        """初始化聊天会话"""
        self.messages = []
        if system_message:
            self.add_message("system", system_message)

    def add_message(self, role: str, content: str) -> None:
        """添加消息"""
        assert role in ["system", "user", "assistant", "tool"], f"Role must in [system, user, assistant, tool]"
        self.messages.append({"role": role, "content": content})

    def clear(self) -> None:
        """清空会话历史"""
        self.messages = []


class OpenAIClient:
    """
    A simple wrapper around the OpenAI Python SDK for chat completions with retry support.
    """
    def __init__(self, api_key: str = None, api_base: str = None, organization: str = None, model: str = "gpt-4o", max_tokens: int = 4096, system_prompt: str = "You are a helpful assistant."):
        """
        Initialize the client and conversation context.

        :param api_key: Your OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        :param organization: (Optional) Your OpenAI organization ID.
        :param model: The default model to use for chat completions.
        :param max_tokens: The maximum number of tokens for each completion.
        :param system_prompt: System-level instruction for the assistant.
        """
        assert api_key, 'Must provide api key'
        openai.api_key = api_key
        if organization:
            openai.organization = organization
        if api_base:
            openai.base_url = api_base

        self.model = model
        self.max_tokens = max_tokens
    def chat_sync(self,
                  system_prompt='You are a helpful assistant.',
                  user_prompt: str = '',
                  model: str = None,
                  max_tokens: int = None,
                  temperature: float = 0.1,
                  return_raw: bool = False):
        """
        Send a message and get a response.

        :param user_prompt: The user's message text.
        :param model: (Optional) Model name to override the default.
        :param max_tokens: (Optional) Max tokens override.
        :param temperature: Sampling temperature.
        :param return_raw: If True, return the full API response.
        :return: A tuple of (reply, full_response) if return_raw, otherwise reply.
        """
        # Append user message
        messages = [
            {
                'content': system_prompt,
                'role': 'system',
            },
            {
                'content': user_prompt,
                'role': 'user',
            },
        ]

        # Select model and max_tokens
        m = model or self.model
        mt = max_tokens or self.max_tokens

        # Call OpenAI API
        from openai import OpenAI

        openai.api_key = os.environ.get('OPENAI_API_KEY', None)
        openai.base_url = os.environ.get('OPENAI_API_BASE', None)

        response = openai.chat.completions.create(
            model=m,
            messages=messages,
            max_tokens=mt,
            temperature=temperature
        )

        # Extract assistant reply
        reply = response.choices[0].message.content

        if return_raw:
            return reply, response
        return reply

    def chat_sync_retry(self,
                        system_prompt = 'You are a helpful assistant.',
                        user_prompt: str = '',
                        model: str = None,
                        max_retry: int = 5,
                        **kwargs):
        """
        Retry chat on failure.

        :param user_prompt: The user's message.
        :param model: (Optional) Model name to override.
        :param max_retry: Number of retry attempts.
        :param kwargs: Additional args for chat_sync.
        :return: A tuple of (reply, full_response) if return_raw, otherwise reply.
        """
        for attempt in range(max_retry):
            try:
                return self.chat_sync(system_prompt, user_prompt, model=model, **kwargs)
            except Exception as e:
                traceback.print_exc()
                print(f"Attempt {attempt+1} failed: {e}")
                time.sleep(2)
        return None

class ChatClient:
    """聊天客户端类"""

    def __init__(self, server_url: str = "http://localhost:8000", model: str = "Qwen_25_7B_Instruct"):
        """初始化客户端"""
        self.server_url = server_url.rstrip('/')
        self.model = model
        self.session = ChatSession()

    def check_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            response = requests.get(f"{self.server_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                return health_data.get("status") == "ok" and health_data.get("model_loaded")
            return False
        except Exception:
            return False

    def wait_for_server(self, max_retries: int = 10, retry_interval: int = 2) -> bool:
        """等待服务器就绪"""
        for i in range(max_retries):
            if self.check_health():
                return True
            print(f"Server is not ready, retry after {retry_interval} seconds ({i + 1}/{max_retries})...")
            time.sleep(retry_interval)
        return False

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 2048,
             temperature: float = 0, top_p: float = 0.9, top_k: int = 50) -> Optional[Dict[str, Any]]:
        """发送聊天请求"""
        url = f"{self.server_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Request failed: HTTP {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"Request error: {e}")
            return None

    def chat_with_session(self, max_tokens: int = 1024, temperature: float = 0.7,
                          top_p: float = 0.9, top_k: int = 50) -> Optional[Dict[str, Any]]:
        """使用当前会话发送聊天请求"""
        result = self.chat(self.session.messages, max_tokens, temperature, top_p, top_k)
        if result and result.get("choices"):
            # 将助手回复添加到会话中
            assistant_message = result["choices"][0]["message"]["content"]
            self.session.add_message("assistant", assistant_message)
        return result

    def add_user_message(self, content: str) -> None:
        """添加用户消息到会话"""
        self.session.add_message("user", content)

    def reset_session(self, system_message: Optional[str] = None) -> None:
        """重置当前会话"""
        self.session = ChatSession(system_message)


def interactive_mode(client: ChatClient, system_message: Optional[str] = None):
    """交互模式"""
    client.reset_session(system_message)
    print("\nInteractive mode, type 'exit' to exit, type 'reset' to reset this chat\n")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        elif user_input.lower() == "reset":
            new_system = input("Input new system prompt (return to use default): ")
            client.reset_session(new_system or system_message)
            print("Chat is reseted")
            continue

        client.add_user_message(user_input)
        print("\n助手: ", end="", flush=True)
        result = client.chat_with_session()

        if result:
            assistant_message = result["choices"][0]["message"]["content"]
            print(assistant_message)
            print(f"\nToken usage: {result['usage']}")
        else:
            print("Request Failed")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="vLLM ChatCompletion API Client")
    parser.add_argument("--server", type=str, default=None, help="服务器地址, server ip")
    parser.add_argument("--model", type=str, default="/user/yaoshu/models/Qwen_25_7B_Instruct", help="模型名称, model name")
    parser.add_argument("--system", type=str, default=None, help="系统提示, system prompt")
    parser.add_argument("--user", type=str, help="用户输入, user prompt")
    parser.add_argument("--max-tokens", type=int, default=1024, help="生成的最大token数量, max gen tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度, temperature (sampling param )")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p采样参数, topp (sampling param)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k采样参数, topk (sampling param)")
    parser.add_argument("--interactive", action="store_true", help="交互模式, interactive mode")
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建客户端
    client = ChatClient(server_url=args.server, model=args.model)

    # 检查服务器健康状态
    print(f"Cheking ready status of {client.server_url}...")
    if not client.wait_for_server():
        print("Server is not ready")
        return

    print("Server is ready")

    if args.interactive:
        # 交互模式
        interactive_mode(client, args.system)
    else:
        # 单次请求模式
        if not args.user:
            print("Non-interactive mode must inlcude param: --user")
            return

        print("Dialogue begin...")
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.user})

        result = client.chat(
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )

        if result:
            assistant_message = result["choices"][0]["message"]["content"]
            print("\nChat result:")
            print("-" * 50)
            print(assistant_message)
            print("-" * 50)
            print(f"Token usage: {result['usage']}")


if __name__ == "__main__":
    main()