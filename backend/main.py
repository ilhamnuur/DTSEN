import os
import sys
import cv2
import numpy as np
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Adjust path to import local module
from core.captcha_solver import load_model, solve_captcha
from core.bps_client import fetch_captcha, submit_desil, HEADERS

app = FastAPI(title="DTSEN Fullstack API")

# Setup CORS to allow React Frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI Model on startup
print("Loading Local PyTorch Model...")
LOCAL_MODEL = load_model()
if not LOCAL_MODEL:
    print("[WARNING] Local model not found or failed to load. Will not be able to auto-solve.")

class SubmitAutoRequest(BaseModel):
    nik: str
    dob: str

class SubmitManualRequest(BaseModel):
    nik: str
    dob: str
    captcha: str
    captcha_key: str

def solve_local(img_bytes: bytes) -> str:
    if not LOCAL_MODEL:
        return None
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    pred, conf = solve_captcha(img_bgr, LOCAL_MODEL)
    print(f"Prediksi AI: {pred} (Conf: {conf:.2f})")
    return pred

@app.get("/api/get-captcha")
async def api_get_captcha():
    async with httpx.AsyncClient(headers=HEADERS) as client:
        c_key, c_raw, _ = await fetch_captcha(client)
        if not c_key:
            return {"success": False, "message": "Gagal mengambil captcha dari BPS"}
        
        img_src = "data:image/jpeg;base64," + c_raw if not c_raw.startswith("data:image") else c_raw
        return {
            "success": True,
            "captcha_key": c_key,
            "captcha_img": img_src
        }

@app.post("/api/submit")
async def api_submit_auto(req: SubmitAutoRequest):
    # Format DOB string if DD/MM/YYYY is passed
    parts = req.dob.split('/')
    dob_api = f"{parts[2]}-{parts[1]}-{parts[0]}" if len(parts) == 3 else req.dob

    if not LOCAL_MODEL:
        return {"success": False, "message": "Model AI belum diload. Jalankan setup retrain terlebih dahulu."}

    max_attempts = 5
    async with httpx.AsyncClient(headers=HEADERS) as client:
        for attempt in range(max_attempts):
            print(f"Auto-Bypass Attempt {attempt + 1}/{max_attempts}...")
            c_key, c_raw, img_bytes = await fetch_captcha(client)
            if not c_key:
                await asyncio.sleep(1)
                continue
            
            solved_text = solve_local(img_bytes)
            if not solved_text:
                continue
            
            result, code = await submit_desil(client, req.nik, dob_api, c_key, solved_text)
            
            if code == 200:
                if result.get("status") == "error":
                    msg = str(result.get("message", ""))
                    if "captcha" in msg.lower():
                        continue # Retry Captcha
                    return {"success": False, "message": f"Ditolak Sistem: {msg}"}
                else:
                    desil = result.get("data", {}).get("data", {}).get("desil_nasional")
                    if desil:
                        return {"success": True, "message": f"Anda berada pada Desil {desil}", "desil": desil}
                    return {"success": True, "message": "Data tidak ditemukan / NIK salah."}
            else:
                await asyncio.sleep(2)
        
        # Max attempts reached, fallback to manual
        c_key, c_raw, _ = await fetch_captcha(client)
        if c_key:
            img_src = "data:image/jpeg;base64," + c_raw if not c_raw.startswith("data:image") else c_raw
            return {
                "success": False,
                "fallback_manual": True,
                "message": "AI kesulitan memecahkan Captcha. Silakan isi secara manual.",
                "captcha_key": c_key,
                "captcha_img": img_src
            }
        return {"success": False, "message": "Max attempts reached & gagal mengambil manual captcha."}

@app.post("/api/submit_manual")
async def api_submit_manual(req: SubmitManualRequest):
    parts = req.dob.split('/')
    dob_api = f"{parts[2]}-{parts[1]}-{parts[0]}" if len(parts) == 3 else req.dob

    async with httpx.AsyncClient(headers=HEADERS) as client:
        result, code = await submit_desil(client, req.nik, dob_api, req.captcha_key, req.captcha)
        
        if code == 200:
            if result.get("status") == "error":
                return {"success": False, "message": str(result.get("message", "Error tidak diketahui"))}
            
            desil = result.get("data", {}).get("data", {}).get("desil_nasional")
            if desil:
                return {"success": True, "message": f"Anda berada pada Desil {desil}", "desil": desil}
            return {"success": True, "message": "Data tidak ditemukan / NIK salah."}
        
        return {"success": False, "message": f"Server Error: {code}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
