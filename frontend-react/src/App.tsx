import { useEffect, useState } from 'react';

type Health = { status?: string; pipeline_ready?: boolean; device?: string; available_models?: string[]; missing_models?: string[]; inference_error?: string | null };
type ReconstructionResult = {
  generated_2d_b64?: string;
  selected_model_key?: string;
  selected_model_name?: string;
  paths?: Record<string, unknown>;
};
const API = import.meta.env.VITE_API_BASE ?? '';

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<ReconstructionResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/health`).then(r => r.json()).then(setHealth).catch(() => setHealth(null));
  }, []);

  async function reconstruct() {
    if (!file) return;
    setBusy(true); setError(''); setResult(null);
    const body = new FormData(); body.append('file', file);
    try {
      const response = await fetch(`${API}/api/process_sketch`, { method: 'POST', body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? 'Reconstruction failed');
      setResult(payload as ReconstructionResult);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not connect to the API');
    } finally { setBusy(false); }
  }

  const glbPath = result?.paths?.viewer_glb_url;
  const glbUrl = typeof glbPath === 'string' ? `${API}${glbPath}` : '';

  return <main className="app">
    <header><div><small>FORENSIC RESEARCH PROTOTYPE</small><h1>Sketch reconstruction</h1><p>Generate a 2D estimate and an approximate 3D surface from a facial sketch.</p></div>
      <span className={health?.pipeline_ready ? 'status online' : 'status'}>{health?.pipeline_ready ? 'PIPELINE ONLINE' : 'PIPELINE LIMITED'}</span>
    </header>
    <div className="notice">Research limitation: generated faces are model estimates, not verified identities. Eye-only sketches may not contain enough information for reliable full-face completion.</div>
    <section className="panel">
      <h2>1. Upload a sketch</h2>
      <input type="file" accept="image/png,image/jpeg,image/webp,image/bmp" onChange={e => { setFile(e.target.files?.[0] ?? null); setResult(null); setError(''); }} />
      {file && <p>Selected: {file.name}</p>}
      <button disabled={!file || busy || !health?.pipeline_ready} onClick={reconstruct}>{busy ? 'Processing…' : 'Run reconstruction'}</button>
      {error && <p className="error">{error}</p>}
      {!health?.pipeline_ready && <p className="muted">{health?.inference_error ?? 'API is offline or no compatible model checkpoint is loaded.'}</p>}
    </section>
    <section className="panel">
      <h2>2. Reconstruction output</h2>
      {result?.generated_2d_b64 ? <img className="result" src={`data:image/png;base64,${result.generated_2d_b64}`} alt="Generated 2D face estimate"/> : <p className="muted">Your 2D estimate will appear here after processing.</p>}
      {glbUrl && <p><a href={glbUrl} target="_blank" rel="noreferrer">Open 3D GLB model</a></p>}
      {result && <p className="muted">Selected model: {String(result.selected_model_name ?? result.selected_model_key ?? 'Not specified')}. Selection scores are not forensic confidence values.</p>}
    </section>
    <section className="panel"><h2>Model availability</h2><p className="muted">Device: {health?.device ?? 'Unknown'}</p><ul>{(health?.available_models ?? []).map(model => <li key={model}>{model}</li>)}</ul></section>
  </main>;
}