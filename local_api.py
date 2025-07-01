# Local DB using by Secondary Users
# PURPOSE: Stores used pseudonyms for comparison with remote DB.
# July 2025

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import time

app = FastAPI()
local_entries = []

class PseudonymEntry(BaseModel):
    string_value: str
    timestamp: float

@app.post("/write/")
def write_entry(entry: PseudonymEntry):
    now = time.time()

    # Prune entries older than 60 seconds
    global local_entries
    local_entries = [e for e in local_entries if now - e["timestamp"] <= 60]

    # Store new entry
    entry_dict = entry.dict()
    local_entries.append(entry_dict)

    print(f"[LOCAL DB] Stored pseudonym: {entry.string_value}")
    return {"status": "ok"}

@app.get("/read_all/")
def read_all():
    # Return most recent entries first
    sorted_entries = sorted(local_entries, key=lambda x: x["timestamp"], reverse=True)
    return {"entries": sorted_entries}

@app.get("/read_by_string/{value}")
def read_by_string(value: str):
    matches = [entry for entry in local_entries if entry['string_value'] == value]
    if matches:
        return matches[0]
    raise HTTPException(status_code=404, detail="Not found")

@app.delete("/delete_by_string/{value}")
def delete_by_string(value: str):
    global local_entries
    initial_len = len(local_entries)
    local_entries = [entry for entry in local_entries if entry["string_value"] != value]
    if len(local_entries) < initial_len:
        print(f"[DELETE] Deleted entry with string_value: {value}")
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Entry not found")
