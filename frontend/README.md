# DTSEN React Frontend (Vite)

Folder ini dikhususkan untuk membangun *UI/UX Layer* program BPS Auto-Bypass. Proyek ini dibangung menggunakan inisialisasi minimal dari Vite dikombinasikan dengan React untuk kecepatan ekstrem dan performa HMR (Hot Module Replacement) tinggi.

## Cara Kerja

1. **State Management**: Form pendaftaran sepenuhnya dikontrol lewat *React Hooks* (`useState`, `useEffect`).
2. **Koneksi Eksternal**: Berkomunikasi langsung ke `http://127.0.0.1:8000/api` (Pastikan direktori `../backend` sedang dalam status *Running*).
3. **Penyajian CSS**: Karena berpegang pada *best practice* kemandirian ringan tanpa ketergantungan utility class berlebih (bebas dari Tailwind/Bootstrap default), perombakan estetika desain (*Glassmorphism* & *Dark Mode Animations*) diurus tunggal melaui berkas `src/index.css`.

## Skrip NPM

Pada terminal, di dalam direktori ini:

- `npm run dev`: Membuka server *development* (Port 5173).
- `npm run build`: Melakukan *build & bundling* aset minimal ke folder `dist` apabila aplikasi sudah matang untuk diproduksi (Nginx/Apache).
- `npm run lint`: Mendeteksi baris kode Javascript yang inkonsisten sesuai standar *ESLint*.

## Optimasi Lebih Lanjut (Future Best Practices)
- Untuk ukuran proyek yang makin membesarkan, direkomendasikan membuat folder `src/components`, `src/hooks`, atau `src/services/api` guna memisahkan abstraksi pengambilan HTTP (seperti instruksi `fetch()`) keluar dari `App.jsx`.
- Pertimbangkan pindah ke TypeScript (`.tsx`) jika model objek hasil Desil API sering berganti struktur demi mencegah *breaking errors* mendadak.
