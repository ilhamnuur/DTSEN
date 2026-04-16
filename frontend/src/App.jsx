import { useState, useRef, useEffect } from 'react';
import './index.css';

const API_BASE = 'http://127.0.0.1:8000/api';

function App() {
  const [nik, setNik] = useState('');
  const [dob, setDob] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState({ type: '', message: '' });
  
  const [isManualMode, setIsManualMode] = useState(false);
  const [manualCaptcha, setManualCaptcha] = useState({ key: '', img: '', input: '' });

  // Deteksi NIK ke Tgl Lahir
  useEffect(() => {
    const rawNik = nik.replace(/\D/g, '');
    if (rawNik.length >= 12 && !isManualMode && rawNik.length <= 16) {
      let dd = parseInt(rawNik.substring(6, 8));
      const mm = rawNik.substring(8, 10);
      const yy = parseInt(rawNik.substring(10, 12));
      
      if (!isNaN(dd) && !isNaN(yy)) {
        if (dd > 40) dd -= 40; // Rumus perempuan
        const ddf = dd.toString().padStart(2, '0');
        const yyyy = (yy > 30) ? "19" + yy.toString().padStart(2, '0') : "20" + yy.toString().padStart(2, '0');
        setDob(`${ddf}/${mm}/${yyyy}`);
      }
    }
  }, [nik, isManualMode]);

  // Formatter DOB auto-slash
  const handleDobChange = (e) => {
    let val = e.target.value.replace(/\D/g, '');
    let newVal = '';
    if (val.length >= 1) newVal += val.substring(0, 2);
    if (val.length >= 3) newVal += '/' + val.substring(2, 4);
    if (val.length >= 5) newVal += '/' + val.substring(4, 8);
    setDob(newVal);
  };

  const loadManualCaptcha = async () => {
    try {
      const res = await fetch(`${API_BASE}/get-captcha`);
      const data = await res.json();
      if (data.success) {
        setManualCaptcha({ key: data.captcha_key, img: data.captcha_img, input: '' });
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setStatus({ type: '', message: '' });

    const payload = { nik, dob };

    let url = `${API_BASE}/submit`;
    if (isManualMode) {
      if (!manualCaptcha.input) {
        setLoading(false);
        return;
      }
      payload.captcha = manualCaptcha.input;
      payload.captcha_key = manualCaptcha.key;
      url = `${API_BASE}/submit_manual`;
    }

    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      if (data.success) {
        setStatus({ type: 'success', message: `🎉 ${data.message}` });
        if (isManualMode) setIsManualMode(false);
      } else {
        if (data.fallback_manual) {
          setIsManualMode(true);
          setManualCaptcha({ key: data.captcha_key, img: data.captcha_img, input: '' });
          setStatus({ type: 'error', message: `⚠️ ${data.message}` });
        } else {
          setStatus({ type: 'error', message: `⚠️ ${data.message}` });
          if (isManualMode) loadManualCaptcha();
        }
      }
    } catch (err) {
      setStatus({ type: 'error', message: "⛔ Koneksi ke API Server (FastAPI) Gagal." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="orb-1"></div>
      <div className="orb-2"></div>
      
      <main className="glass-panel p-8" style={{ width: '100%', maxWidth: '500px', borderRadius: '1.5rem', position: 'relative', zIndex: 1 }}>
        <div className="text-center mb-6">
          <div style={{ display: 'inline-block', padding: '0.25rem 0.75rem', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.2)', borderRadius: '9999px', marginBottom: '1rem' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#60a5fa', letterSpacing: '0.1em', textTransform: 'uppercase' }}>Fullstack System</span>
          </div>
          <h1 className="font-heading" style={{ fontSize: '2.5rem', margin: '0 0 0.5rem 0', fontWeight: 800 }}>DTSEN<span style={{color: '#3b82f6'}}>.</span></h1>
          <p style={{ color: '#94a3b8', fontSize: '0.875rem', margin: 0 }}>Verifikasi data tanpa perlu input Captcha.</p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <label htmlFor="nik" className="input-label">Nomor Induk Kependudukan</label>
            <input 
              type="text" 
              id="nik" 
              className="glass-input px-4 py-3 font-mono" 
              placeholder="16 DIGIT NIK" 
              maxLength="16" 
              required 
              autoComplete="off"
              value={nik}
              onChange={e => setNik(e.target.value)}
            />
          </div>

          <div className="input-group">
            <div className="flex justify-between items-center mb-2">
              <label htmlFor="dob" className="input-label" style={{marginBottom: 0}}>Tanggal Lahir</label>
              {!isManualMode && <span style={{ fontSize: '10px', color: '#34d399', background: 'rgba(52, 211, 153, 0.1)', padding: '2px 6px', borderRadius: '4px', fontWeight: 600 }}>Auto-Detected</span>}
            </div>
            <input 
              type="text" 
              id="dob" 
              className="glass-input px-4 py-3 font-mono" 
              placeholder="DD/MM/YYYY" 
              maxLength="10" 
              required 
              autoComplete="off"
              value={dob}
              onChange={handleDobChange}
            />
          </div>

          {!isManualMode ? (
            <div style={{ background: 'rgba(30, 58, 138, 0.2)', border: '1px solid rgba(59, 130, 246, 0.2)', borderRadius: '0.75rem', padding: '1rem', marginTop: '2rem', display: 'flex', gap: '0.75rem' }}>
              <div style={{ color: '#60a5fa', marginTop: '2px' }}>✨</div>
              <div style={{ fontSize: '0.75rem', color: '#cbd5e1', lineHeight: 1.5 }}>
                 <span style={{ fontWeight: 600, color: '#fff' }}>AI Auto-Bypass Aktif.</span> Kode keamanan akan diproses secara diam-diam oleh server FastAPI.
              </div>
            </div>
          ) : (
            <div style={{ background: 'rgba(120, 53, 15, 0.4)', border: '1px solid rgba(245, 158, 11, 0.3)', borderRadius: '0.75rem', padding: '1rem', marginTop: '2rem' }}>
              <label className="input-label" style={{ color: '#fbbf24', marginBottom: '0.75rem' }}>Verifikasi Manual Diperlukan</label>
              <div className="flex gap-3">
                <div className="flex gap-2" style={{ flexShrink: 0 }}>
                  <img src={manualCaptcha.img} alt="Captcha" style={{ height: '3rem', width: '8rem', background: '#0f172a', borderRadius: '0.5rem', objectFit: 'contain', border: '1px solid #334155' }} />
                  <button type="button" onClick={loadManualCaptcha} style={{ height: '3rem', width: '3rem', background: '#334155', border: 'none', borderRadius: '0.5rem', color: 'white', cursor: 'pointer', fontWeight: 'bold' }}>↻</button>
                </div>
                <input 
                  type="text" 
                  className="glass-input px-4 font-mono text-center" 
                  placeholder="Ketik..." 
                  maxLength="6" 
                  autoComplete="off"
                  value={manualCaptcha.input}
                  onChange={e => setManualCaptcha({...manualCaptcha, input: e.target.value})}
                  required
                />
              </div>
            </div>
          )}

          <div className="mt-6">
            {!loading ? (
              <button type="submit" className="btn-modern py-3 rounded-xl font-heading" style={{ fontSize: '1.125rem', fontWeight: 700, color: 'white' }}>
                {isManualMode ? "Submit Captcha" : "Cek Status Desil"}
              </button>
            ) : (
              <div style={{ padding: '1rem', textAlign: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', fontSize: '0.875rem', color: '#cbd5e1', marginBottom: '1rem', fontWeight: 500 }}>
                  <svg className="spinner" viewBox="0 0 50 50"><circle cx="25" cy="25" r="20" fill="none"></circle></svg>
                  Processing Auto-Bypass & Verifying...
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  <div className="skeleton rounded-2xl" style={{ height: '0.75rem', width: '100%' }}></div>
                  <div className="skeleton rounded-2xl" style={{ height: '0.75rem', width: '80%', margin: '0 auto' }}></div>
                </div>
              </div>
            )}
          </div>
        </form>

        {status.message && (
          <div className={`status-box ${status.type === 'success' ? 'status-success' : 'status-error'}`}>
            {status.message}
          </div>
        )}
      </main>
    </>
  );
}

export default App;
