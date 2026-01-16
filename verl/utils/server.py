#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async vLLM server (>=0.8.0) with OpenAI-compatible ChatCompletion endpoint
"""

import os
import uuid
import time
import argparse
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import SamplingParams
from transformers import AutoTokenizer


# ========= 数据模型 =========
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant|tool)$")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int | None = 1024
    temperature: float | None = 0.7
    top_p: float | None = 0.9
    top_k: int | None = 50


# OpenAI-style response
class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


# ========= 全局引擎 =========
llm_engine: AsyncLLMEngine | None = None
tokenizer: AutoTokenizer | None = None


# ========= CLI =========
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--gpu-ids", default="0,1,2,3")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--max-model-len", type=int, default=None)
    return p.parse_args()


def build_prompt(messages: List[ChatMessage]) -> str:
    global tokenizer
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt


# ========= FastAPI 启动 =========
async def startup_event():
    global llm_engine
    global tokenizer
    args = parse_args()

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print("CUDA_VISIBLE_DEVICES =", args.gpu_ids)

    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        # max_model_len=args.max_model_len,
        # max_num_batched_tokens=4096,
        max_num_seqs=256,
        gpu_memory_utilization=0.8,
    )
    print("Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("Model loaded.")


# ========= 路由 =========
router = APIRouter()


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    global llm_engine
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="model not ready")

    prompt = build_prompt(req.messages)

    # 采样参数
    sp = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
    )

    request_id = str(uuid.uuid4())
    final_output = None
    async for out in llm_engine.generate(prompt, sp, request_id):
        final_output = out

    if final_output is None or not final_output.outputs:
        raise HTTPException(status_code=500, detail="generation failed")

    answer = final_output.outputs[0].text.strip()

    # usage 精确 token 统计
    prompt_tokens = len(llm_engine.tokenizer.encode(prompt))
    completion_tokens = len(llm_engine.tokenizer.encode(answer))
    now = int(time.time())

    return ChatCompletionResponse(
        id=f"chatcmpl-{request_id}",
        created=now,
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=answer),
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@router.get("/health")
async def health():
    return {"status": "ok", "model_loaded": llm_engine is not None}


# ========= 主模块 =========
def main():
    import uvicorn

    args = parse_args()
    app = FastAPI(title="vLLM-OpenAI 兼容服务")
    app.add_event_handler("startup", startup_event)
    app.include_router(router)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()