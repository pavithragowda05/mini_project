import { useEffect, useMemo, useState } from 'react';

type CheckpointStatus = {
  filename?: string;
  available?: boolean;
  loaded?: boolean;
};
type Health = {
  status?: string;
  pipeline_ready?: boolean;
  reconstruction_ready?: boolean;
  device?: string | null;
  inference_error?: string | null;
  checkpoints?: Record<string, CheckpointStatus>;
};
type ReconstructionResult = {
  status?: string;
  generated_2d_b64?: string;
  selected_model_key?: string;
  selected_model_name?: string;
  paths?: Record<string, unknown>;
  message?: string;
};

function getApiBase() {
  // Supports both .env.local and the ?api=http://127.0.0.1:8003 link.
  const queryApi = new URLSearchParams(window.location.search).get('api');
  const configuredApi = queryApi || import.meta.env.VITE_API_BASE || '';
  return configuredApi.replace(/\/+$/, '');
}

export default function App() {
  const API = useMemo(getApiBase, []);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<ReconstructionResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function refreshHealth() {
    try {
      const response = await fetch(`${API}/health`);
      if (!response.ok) throw new Error(`Backend returned HTTP ${response.status}`);
      const data = await response.json() as Health;
      setHealth(data);
      setHealthError('');
    } catch (e) {
      setHealth(null);
      setHealthError(e instanceof Error ? e.message : 'Cannot connect to backend');
    }
  }

  useEffect(() => {
    void refreshHealth();
    const timer = window.setInterval(() => void refreshHealth(), 10000);
    return () => window.clearInterval(timer);
  }, [API]);

  async function reconstruct() {
    if (!file) return;
    setBusy(true);
    setError('');
    setResult(null);
    const body = new FormData();
    body.append('file', file);
    try {
      const response = await fetch(`${API}/api/process_sketch`, { method: 'POST', body });
      const payload = await response.json() as ReconstructionResult;
      if (!response.ok) throw new Error(payload.message || (typeof payload === 'object' && 'detail' in payload ? String((payload as Record<string, unknown>).detail) : 'Reconstruction failed'));
      if (payload.status === 'preprocessing_only') {
        throw new Error(payload.message || 'The backend can preprocess this sketch, but no trained reconstruction model is loaded.');
      }
      setResult(payload);
      await refreshHealth();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not process the sketch');
    } finally {
      setBusy(false);
    }
  }

  const loadedModels = Object.entries(health?.checkpoints ?? {})
    .filter(([, checkpoint]) => checkpoint.loaded)
    .map(([key]) => key);
  const glbPath = result?.paths?.viewer_glb_url;
  const glbUrl = typeof glbPath === 'string' ? `${API}${glbPath}` : '';

  return <main className="app">
    <header>
      <div>
        <small>FORENSIC RESEARCH PROTOTYPE</small>
        <h1>Sketch reconstruction</h1>
        <p>Generate a 2D estimate and an approximate 3D surface from a facial sketch.</p>
      </div>
      <span className={health?.pipeline_ready ? 'status online' : 'status'}>
        {health?.pipeline_ready ? 'PIPELINE ONLINE' : healthError ? 'BACKEND OFFLINE' : 'PIPELINE LIMITED'}
      </span>
    </header>

    <div className="notice">
      Research limitation: generated faces are model estimates, not verified identities. Eye-only sketches may not contain enough information for reliable full-face completion.
    </div>

    <section className="panel">
      <h2>1. Upload a sketch</h2>
      <input
        type="file"
        accept="image/png,image/jpeg,image/webp,image/bmp"
        onChange={e => {
          setFile(e.target.files?.[0] ?? null);
          setResult(null);
          setError('');
        }}
      />
      {file && <p>Selected: {file.name}</p>}
      <button disabled={!file || busy || !health?.pipeline_ready} onClick={reconstruct}>
        {busy ? 'Processing…' : 'Run reconstruction'}
      </button>
      {healthError && <p className="error">Backend connection failed: {healthError}. API: {API || '(same origin)'}</p>}
      {error && <p className="error">{error}</p>}
      {!healthError && !health?.pipeline_ready && (
        <p className="muted">{health?.inference_error ?? 'Backend is reachable, but no trained reconstruction pipeline is ready.'}</p>
      )}
      {health?.pipeline_ready && <p className="muted">Loaded models: {loadedModels.join(', ') || 'Checking model list…'} · Device: {health.device ?? 'Unknown'}</p>}
    </section>

    <section className="panel">
      <h2>2. Reconstruction output</h2>
      {result?.generated_2d_b64
        ? <img className="result" src={`data:image/png;base64,${result.generated_2d_b64}`} alt="Generated 2D face estimate" />
        : <p className="muted">Your 2D estimate will appear here after processing.</p>}
      {glbUrl && <p><a href={glbUrl} target="_blank" rel="noreferrer">Open 3D GLB model</a></p>}
      {result && <p className="muted">Selected model: {result.selected_model_name ?? result.selected_model_key ?? 'Not specified'}. Selection scores are heuristic rankings, not forensic confidence values.</p>}
    </section>

    <section className="panel">
      <h2>Model availability</h2>
      <p className="muted">Device: {health?.device ?? 'Unknown'}</p>
      <ul>
        {Object.entries(health?.checkpoints ?? {}).map(([key, checkpoint]) => (
          <li key={key}>{key}: {checkpoint.loaded ? 'loaded' : checkpoint.available ? 'file found, not loaded' : 'checkpoint missing'}</li>
        ))}
      </ul>
    </section>
  </main>;
}
