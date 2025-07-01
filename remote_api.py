# Remote DB for the StopSec System
# PURPOSE: Stores Interference Report sent from the primary user.
# July 2025

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import time
import uvicorn

app = FastAPI()
remote_entries = []

class PseudonymEntry(BaseModel):
    string_value: str
    timestamp: float

@app.post("/write/")
def write_entry(entry: PseudonymEntry):
    now = time.time()

    # Prune entries older than 60 seconds
    global remote_entries
    remote_entries = [e for e in remote_entries if now - e["timestamp"] <= 60]

    # Store new entry
    entry_dict = entry.dict()
    remote_entries.append(entry_dict)

    print(f"[REMOTE DB] Stored pseudonym: {entry.string_value}")
    return {"status": "ok"}

@app.get("/read_all/")
def read_all():
    # Return most recent entries first
    sorted_entries = sorted(remote_entries, key=lambda x: x["timestamp"], reverse=True)
    return {"entries": sorted_entries}

@app.get("/read_by_string/{value}")
def read_by_string(value: str):
    matches = [entry for entry in remote_entries if entry['string_value'] == value]
    if matches:
        return matches[0]
    raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    
    print("[STARTING] Remote API server with Uvicorn...")
    uvicorn.run("remote_api:app", host="0.0.0.0", port=8080, reload=False)
    