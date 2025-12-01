# proxy.py
import asyncio
from calendar import c
import json
import logging
import tempfile
import uuid
from pathlib import Path
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import httpx
# cache.py
import redis
OPENAI_TIMEOUT = httpx.Timeout(None)
redis_client = redis.asyncio.from_url("redis://localhost:6379", decode_responses=True)


# ---- Logging setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

client = OpenAI()


# ---- Config ----
IDLE_TIMEOUT = 30.0                       # flush after 30s of no new requests
CHAT_ENDPOINT = "/v1/chat/completions"
EMBEDDING_ENDPOINT = "/v1/embeddings"
COMPLETION_WINDOW = "24h"

# ---- Shared state ----
chat_queue = []                                 # list[{"custom_id", "body", "future"}]
chat_queue_lock = asyncio.Lock()
embedding_queue = []                            # list[{"custom_id", "body", "future"}]
embedding_queue_lock = asyncio.Lock()



# cache_path = "/home/nccn12/graphiti/batching_service_cache/input_cache.json"
# cache = {}
# cache_lock = asyncio.Lock()


from cache import OpenAIResponseCache
import json

cache = OpenAIResponseCache()
idle_task: asyncio.Task | None = None           # debounced idle timer
embedding_idle_task: asyncio.Task | None = None

last_mtime = 0
import hashlib
def make_cache_key(body: dict) -> str:
    """Stable hash of request body for cache lookup."""
    body_str = json.dumps(body, sort_keys=True, ensure_ascii=False)
    return "request:" + hashlib.sha256(body_str.encode()).hexdigest()



# async def load_cache_if_changed():
#     if not os.path.exists(cache_path):
#         return
#     async with cache_lock:
#         with open(cache_path) as f:
#             global cache
#             cache = json.load(f)

# async def _read_cache_file() -> dict:
#     """Read JSON cache on a background thread (non-blocking)."""
#     def _read():
#         with open(cache_path, "r", encoding="utf-8") as f:
#             return json.load(f)
#     return await asyncio.to_thread(_read)

# async def _ensure_cache_file_exists():
#     """Create an empty cache file if none exists (atomic-ish)."""
#     if not os.path.exists(cache_path):
#         def _write_empty():
#             with open(cache_path + ".tmp", "w", encoding="utf-8") as f:
#                 json.dump({}, f, ensure_ascii=False)
#             os.replace(cache_path + ".tmp", cache_path)
#         await asyncio.to_thread(_write_empty)

# async def load_cache_from_disk():
#     """Call this at startup to populate the in-memory cache."""
#     global cache
#     await _ensure_cache_file_exists()
#     data = await _read_cache_file()
#     async with cache_lock:
#         cache.clear()
#         cache.update(data)
#     log.info(f"Cache loaded: {len(cache)} entries from {cache_path}")

# from fastapi.concurrency import asynccontextmanager
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     await load_cache_from_disk()
#     yield
#     log.info("Shutting down...")

app = FastAPI()

def compile_request(item: dict, endpoint:str) -> dict:
    """Compile a request for the OpenAI API."""
    return {
        "custom_id": item["custom_id"],
        "method": "POST",
        "url": endpoint,
        "body": item["body"],
    }

def _reset_idle_timer():
    """Debounce: cancel prior timer and start a fresh one."""
    global idle_task
    if idle_task and not idle_task.done():
        idle_task.cancel()
        log.debug("Chat idle timer reset (cancelled old timer)")
    idle_task = asyncio.create_task(_idle_then_flush())
    log.debug("Chat idle timer started")


def _reset_embedding_idle_timer():
    """Debounce: cancel prior timer and start a fresh one."""
    global embedding_idle_task
    if embedding_idle_task and not embedding_idle_task.done():
        embedding_idle_task.cancel()
        log.debug("Embedding idle timer reset (cancelled old timer)")
    embedding_idle_task = asyncio.create_task(_embedding_idle_then_flush())
    log.debug("Embedding idle timer started")


async def _idle_then_flush():
    """Sleep for IDLE_TIMEOUT; if no new requests arrive, flush."""
    try:
        await asyncio.sleep(IDLE_TIMEOUT)
        log.info("Idle timeout reached for chat → flushing")
        asyncio.create_task(flush(chat_queue, chat_queue_lock, endpoint=CHAT_ENDPOINT))
    except asyncio.CancelledError:
        log.info("Chat idle timer cancelled (new request arrived)")


async def _embedding_idle_then_flush():
    """Sleep for IDLE_TIMEOUT; if no new requests arrive, flush."""
    try:
        await asyncio.sleep(IDLE_TIMEOUT)
        log.info("Idle timeout reached for embeddings → flushing")
        asyncio.create_task(flush(embedding_queue, embedding_queue_lock, endpoint=EMBEDDING_ENDPOINT))
    except asyncio.CancelledError:
        log.info("Embedding idle timer cancelled (new request arrived)")


async def wait_for_batch(batch_id: str):
    """Poll the Batch API until terminal state."""
    log.info(f"Polling batch {batch_id}")
    while True:
        status = client.batches.retrieve(batch_id, timeout= OPENAI_TIMEOUT)
        log.info(f"Batch {batch_id} status={status.status}")
        if status and status.status in ("completed", "failed", "cancelled", "expired"):
            log.info(f"Batch {batch_id} finished with status={status.status}")
            return status
        await asyncio.sleep(15)



async def flush(queue, queue_lock, endpoint):
    """Flush queued requests as one batch job and resolve futures with results."""
    async with queue_lock:
        if not queue:
            return
        batch = list(queue)
        queue.clear()

    log.info(f"Flushing {len(batch)} requests → endpoint={endpoint}")

    requests = [
        compile_request(item, endpoint)
        for item in batch
    ]
    futures_by_id = {item["custom_id"]: item["future"] for item in batch}
    input_by_id = {item["custom_id"]: item["body"] for item in batch}
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with open(path, "w") as f:
            for r in requests:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        log.debug(f"Wrote batch input file: {path}")
        batch_input_file = client.files.create(
            file=open(path, "rb"),
            purpose="batch",
            timeout = OPENAI_TIMEOUT
        )
        log.info(f"Uploaded input file {batch_input_file.id}")

        created = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint=endpoint,
            completion_window=COMPLETION_WINDOW,
            timeout = OPENAI_TIMEOUT
        )
        log.info(f"Created batch {created.id} with {len(requests)} requests")

        status = await wait_for_batch(created.id)

        if status.status != "completed" or not status.output_file_id:
            log.error(f"Batch {created.id} failed with status={status.status}")
            err = {"error": {"message": f"batch {status.status}", "type": "batch_error"}}
            for fut in futures_by_id.values():
                if not fut.done():
                    fut.set_result(err)
            return
        lines = client.files.content(status.output_file_id).text.splitlines()
        log.info(f"Downloaded {len(lines)} results for batch {created.id}")
        results_by_id = {}
        
        for line in lines:
            try:
                rec = json.loads(line)
                cid = rec.get("custom_id")
                resp = rec.get("response", {})
                body = resp.get("body")
                if cid:
                    results_by_id[cid] = body
                    # Cache the result
                    await cache.set(make_cache_key(input_by_id.get(cid)), {
                            "input": input_by_id.get(cid),
                            "output": body,
                            "batch_id": created.id,
                            "custom_id": rec["custom_id"],
                            "completed_at": created.completed_at,
                        })
            except Exception as e:
                log.warning(f"Failed to parse result line: {e}")
                continue

        for cid, fut in futures_by_id.items():
            if fut.done():
                continue
            fut.set_result(results_by_id.get(cid, {"error": "missing result"}))
        log.info(f"Resolved {len(futures_by_id)} futures for batch {created.id}")

    except Exception as e:
        log.exception(f"Error during flush: {e}")
        err = {"error": {"message": str(e), "type": "internal_error"}}
        for fut in futures_by_id.values():
            if not fut.done():
                fut.set_result(err)
    finally:
        Path(path).unlink(missing_ok=True)

@app.post("/test_caching")
async def test_caching(request: Request):
    body = await request.json()
    cache_key = make_cache_key(body)
    cached = await cache.get(cache_key)
    if cached:
        return JSONResponse(cached["output"])
    else:
        return JSONResponse({"error": "not found"}, status_code=404)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    custom_id = f"req-{uuid.uuid4().hex}"

    cache_key = make_cache_key(body)
    cached = await cache.get(cache_key)
    if cached:
        logging.info(f"Cache hit for chat request {custom_id}")
        return JSONResponse(cached["output"])
    else:
        log.info(f"Received request: {body}")
        fut = asyncio.get_event_loop().create_future()
        async with chat_queue_lock:
            chat_queue.append({"custom_id": custom_id, "body": body, "future": fut})
            log.info(f"Enqueued chat request {custom_id} (queue size={len(chat_queue)})")
            _reset_idle_timer()

        result = await fut
        log.info(f"Returning result for chat request")
        return JSONResponse(result)


@app.post("/v1/embeddings")
async def create_embedding(request: Request):
    body = await request.json()

    custom_id = f"req-{uuid.uuid4().hex}"
    
    cache_key = make_cache_key(body)
    cached = await cache.get(cache_key)
    if cached:
        log.info(f"Cache hit for embedding request")
        return JSONResponse(cached["output"])
    else:
        fut = asyncio.get_event_loop().create_future()
        async with embedding_queue_lock:
            log.info(f"Received request:{body}")
            embedding_queue.append({"custom_id": custom_id, "body": body, "future": fut})
            
            log.info(f"Enqueued embedding request {custom_id} (queue size={len(embedding_queue)})")
            _reset_embedding_idle_timer()

        result = await fut
        log.info(f"Returning result for embedding request {custom_id}")
        return JSONResponse(result)
