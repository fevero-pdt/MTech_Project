from fastapi import FastAPI
import time, random

app = FastAPI()

@app.post("/invoke")
def invoke():
    work_ms = random.randint(80, 300)
    time.sleep(work_ms / 1000.0)
    return {"status": "ok", "work_ms": work_ms}
