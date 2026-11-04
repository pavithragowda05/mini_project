# React + TypeScript frontend

This React/Vite UI uses the existing FastAPI endpoints `/health` and `/api/process_sketch`.

## Build and run (PowerShell)
```powershell
cd frontend-react
npm install
npm run build
```

For local development, run `npm run dev`. Set `VITE_API_BASE=http://127.0.0.1:8002` in `.env.local` when Vite runs on a different port from FastAPI.

Start the API from the repository root:
```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8002
```

The `vite-env.d.ts` file imports Vite's client types so `import.meta.env` is typed. JSX conditions that inspect optional API payload fields use boolean expressions or typed string checks so `unknown` values are not passed as React children.

Outputs are estimates for research use, not verified identities.