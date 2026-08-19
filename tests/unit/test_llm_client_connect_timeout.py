"""
Regression test: the shared httpx.AsyncClient used for every LLM provider
request must use a short TCP-connect timeout, distinct from the long
read/write timeout.

Before this fix, `httpx.Timeout(REQUEST_TIMEOUT)` set connect=read=write=pool
to the same 900s value. A provider that is unreachable at the TCP level
(dead local Ollama/llama.cpp process, DNS/network outage) would block a
chunk for up to 15 minutes before the retry loop even got a chance to run --
indistinguishable from the whole job freezing.
"""
import pytest

from src.config import LLM_CONNECT_TIMEOUT, REQUEST_TIMEOUT
from src.core.llm.providers.deepseek import DeepSeekProvider


@pytest.mark.asyncio
async def test_llm_client_uses_short_connect_timeout_and_long_read_timeout():
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.test/chat/completions",
    )
    try:
        client = await provider._get_client()
        timeout = client.timeout
        assert timeout.connect == LLM_CONNECT_TIMEOUT
        assert timeout.connect < REQUEST_TIMEOUT
        # Read/write/pool must stay long: a real LLM response can
        # legitimately take minutes for a large chunk.
        assert timeout.read == REQUEST_TIMEOUT
        assert timeout.write == REQUEST_TIMEOUT
        assert timeout.pool == REQUEST_TIMEOUT
    finally:
        await provider.close()
