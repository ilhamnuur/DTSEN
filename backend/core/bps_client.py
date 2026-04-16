import httpx
import base64
from typing import Tuple, Optional

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://dtsen-form.web.bps.go.id",
    "Referer": "https://dtsen-form.web.bps.go.id/"
}

async def fetch_captcha(client: httpx.AsyncClient) -> Tuple[Optional[str], Optional[str], Optional[bytes]]:
    url = "https://dtsen-form-api.web.bps.go.id/api/reload-captcha"
    try:
        resp = await client.get(url, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        captcha_key = data["captcha"]["key"]
        img_b64 = data["captcha"]["img"]
        
        raw_b64 = img_b64.split(",", 1)[1] if "," in img_b64 else img_b64
        img_bytes = base64.b64decode(raw_b64)
        return captcha_key, raw_b64, img_bytes
    except Exception as e:
        return None, None, None

async def submit_desil(client: httpx.AsyncClient, nik: str, dob: str, captcha_key: str, captcha_text: str) -> Tuple[dict, int]:
    url = "https://dtsen-form-api.web.bps.go.id/api/cek-desil"
    payload = {"nik": nik, "key": captcha_key, "captcha": captcha_text, "tanggal_lahir": dob}
    try:
        resp = await client.post(url, json=payload, timeout=15.0)
        return resp.json(), resp.status_code
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500
