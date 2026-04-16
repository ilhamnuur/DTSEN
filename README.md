# DTSEN Desil Checker

<p align="center">
  <img src="https://img.shields.io/badge/versi-1.0.0-blue?style=for-the-badge" alt="Versi 1.0.0" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React_Vite-Frontend-61DAFB?style=for-the-badge&logo=react" alt="React Vite" />
  <img src="https://img.shields.io/badge/Lisensi-Internal-red?style=for-the-badge" alt="Lisensi" />
</p>

<p align="center">
  <b>Aplikasi Fullstack modern bersenjatakan AI Captcha Solver (PyTorch) lokal untuk mengecek dan memverifikasi status Desil DTSEN secara otomatis tanpa intervensi manual.</b>
</p>

<p align="center">
  <a href="#mulai-cepat">Mulai Cepat</a> · <a href="#fitur">Fitur</a> · <a href="#arsitektur">Arsitektur</a> · <a href="#instalasi">Instalasi</a> · <a href="#konfigurasi">Konfigurasi</a> · <a href="#changelog">Changelog</a>
</p>

---

## Gambaran Umum

**DTSEN Desil Checker** adalah platform berbasis antarmuka web modern yang mengotomatiskan interaksi pengecekan status Desil Masyarakat di situs BPS. Dibangun mengusung arsitektur *Decoupled Fullstack* (memisahkan React UI dan FastAPI Backend), aplikasi ini memecahkan kerumitan tebakan Captcha seketika lewat kekuatan *Computer Vision* murni secara lokal tanpa rely pada layanan API berbayar.

---

## Mulai Cepat

**Prasyarat:** Node.js (v16+), Python (v3.9+), PIP.

```bash
# 1. Jalankan Backend (AI Server)
cd dtsen/backend
pip install -r requirements.txt
python main.py

# 2. Jalankan Frontend (UI Server)
cd dtsen/frontend
npm install
npm run dev
```

---

## Fitur

### 🤖 Local AI Auto-Bypass
Sistem sepenuhnya otomatis meretas tantangan Captcha dari situs BPS melalui *backend* FastAPI. Ditenagai *Convolutional Neural Network* (PyTorch) dan Sklearn Ensemble dengan tingkat keyakinan prediktif super tinggi (~90%++).

### ⚡ Deteksi & Ekstraksi Data Pintar
Otomatis mendapatkan formula Tanggal Lahir (DD/MM/YYYY) secara lekas dan terisolasi dari sisipan 16-Digit NIK untuk efisiensi Input Form.

### 🛡️ Smart Fallback Manual
Jaring pengaman pintar untuk mendeteksi *“Max attempt reached”*. Jika AI gagal selama 5 putaran, sistem mengalihkan UI halus tanpa *reload* agar pengguna bisa memecahkan Captcha secara manual demi mencegah pemblokiran alamat IP.

### ✨ Premium Reactive UI
Antarmuka pengguna premium yang dibalut skema *Dark Mode* mendalam, *Glassmorphism*, dan animasi ringan. Memprioritaskan *User Experience* superior ala SPA (Single Page Application).

### 📡 Asynchronous Engine
Menggunakan pustaka `httpx` and coroutine Python untuk mencegah efek membeku *(hang/blocking)* pada pemrosesan permintaan API yang berlapis.

---

## Arsitektur

```text
┌─────────────────────────────────────────────────────┐
│                 React Vite Frontend                 │
│                 (App.jsx & UI State)                │
└──────────────────────────┬──────────────────────────┘
                           │ HTTP/AJAX (CORS)
┌──────────────────────────▼──────────────────────────┐
│                   FastAPI Backend                   │
│                                                     │
│  ┌─────────────────┐       ┌─────────────────────┐  │
│  │ HTTPX BPS Client│◄─────►│    Local Solver     │  │
│  │ (Async Request) │       │ (OpenCV + PyTorch)  │  │
│  └─────────────────┘       └──────────┬──────────┘  │
└──────────────────┬────────────────────│─────────────┘
                   │                    ▼ 197 MB
┌──────────────────▼────┐    ┌────────────────────────┐
│    BPS DTSEN Server   │    │  captcha_model.pkl     │
│  (API & Image Fetch)  │    │  (Weighted Neural Net) │
└───────────────────────┘    └────────────────────────┘
```

**Prinsip Desain:**
- **Decoupled Architecture** — Logika komputasi berat dan rendering *client* dipisahkan ketat.
- **Stateless Verification** — Aplikasi tidak menyimpan Cookie abadi; setiap sesi `POST` bersifat segar dan sekali pakai.

---

## Instalasi

### 1. Setup Backend (Inti AI)

1. Navigasi menuju `dtsen/backend`
2. Pasang referensi kepustakaan standar:
   ```bash
   pip install -r requirements.txt
   ```
3. *(Lanjutan Opsional)*: Bila perlu menambah akurasi, tempatkan berkas dataset dan jalankan perintah mandiri `retrain.py` (telah disediakan terpisah di alat latihan).
4. Nyalakan server uvicorn internal:
   ```bash
   python main.py
   ```

### 2. Setup Frontend (Layar UI)

1. Menuju terminal baru lalu buka `dtsen/frontend`
2. Pasang ketergantungan JavaScript:
   ```bash
   npm install
   ```
3. Mulai kompilasi *Hot-Reload*:
   ```bash
   npm run dev
   ```
4. Buka akses peramban secara *default* `http://localhost:5173`

---

## Konfigurasi

Sistem terdesain bekerja mulus tanpa konfigurasi mendalam. Beberapa variabel vital diatur statis pada tiap-tiap lingkungan:

| Berkas | Konstanta Utama | Deskripsi / Nilai Default |
|------------|---------|-----------|
| `backend/core/bps_client.py` | `URL` | Endpoint peluru HTTPS (`dtsen-form-api.web.bps.go.id`) |
| `backend/main.py` | `max_attempts` | Modifikasi kepekaan limit AI (Default `5`) |
| `frontend/src/App.jsx` | `API_BASE` | Port lintas pengumpul data FastAPI (`http://127.0.0.1:8000/api`) |

---

## Keamanan

- **100% Proteksi Privasi** — Seluruh deteksi gambar *(Computer Vision)* diproses langsung menggunakan mesin Anda. Tidak ada sekeping byte gambar maupun NIK yang menjalar menuju server API ke-3 seperti OpenAI/Gemini dalam mode *production*.
- **Kendali CORS yang Baik** — Membatasi manipulasi XHR hanya pada origin otentik Anda via perantara `localhost`.
- **Mitigasi Spam API** — Kecepatan Auto-Bypass telah dipasang *delay proxy* buatan untuk menyamar selayaknya perilaku *human-input*.

---

## Struktur Proyek

```text
dtsen/
├── backend/
│   ├── main.py                  # Kendali pusat endpoint FastAPI
│   ├── captcha_solver_model.pkl # Model pra-latih PyTorch
│   ├── requirements.txt         # Referensi modul Python
│   └── core/
│       ├── bps_client.py        # Logic HTTP BPS Fetching
│       └── captcha_solver.py    # Logic Preprocessing OCR OpenCV
└── frontend/
    ├── package.json             # NPM package manager Vite
    ├── vite.config.js
    └── src/
        ├── App.jsx              # Komponen utama React UI
        ├── index.css            # Vanilla stylings & animasi Glass
        └── main.jsx             # React Virtual DOM Injector
```

---

## Changelog

### v1.0.0 (Awal Implementasi Penuh)

**✅ Fitur Terselesaikan**
- Rilis perdana ekosistem pemisahan (Decoupled Fullstack).
- Integrasi sukses antara React UI dan Backend FastAPI.
- Menghapus kerangka lama monolitik Flask.
- Penerapan antarmuka dengan estetik Vanilla CSS penuh dan *Glassmorphism*.
- Validasi model PyTorch SVM+CNN langsung dalam skrip async Python.

---

<p align="center">
  <b>BPS Automation Tools</b><br />
  <sub>Penggunaan internal — Percepatan akses dan olah data fasilitatif</sub>
</p>
