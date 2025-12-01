
import redis
import json

class OpenAIResponseCache:
    def __init__(self, redis_url: str = "redis://localhost:6379", cache_ttl: int = 3600 * 24 * 14):
        self.redis_client = redis.asyncio.from_url(redis_url, decode_responses=True)
        self.cache_ttl = cache_ttl

    
    async def get(self, cache_key: str):
        data = await self.redis_client.get(cache_key)
        if data is None:
            return None
        return json.loads(data)

    async def set(self, cache_key: str, value: dict):
        await self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(value, ensure_ascii=False))


# CACHE_TTL = 3600 * 24 * 7  # Cache time-to-live in seconds
# redis_client = redis.asyncio.from_url("redis://localhost:6379", decode_responses=True)

# async def get_from_cache(cache_key: str):
#     data = await redis_client.get(cache_key)
#     if data is None:
#         return None
#     return json.loads(data)

# async def save_to_cache(cache_key: str, value: dict):
#     await redis_client.setex(cache_key, CACHE_TTL, json.dumps(value, ensure_ascii=False))
