# React frontend TypeScript build fixes

The current GitHub main branch did not contain the `frontend-react/` source folder when this branch was prepared. The React frontend exists in the separately supplied development ZIP, so the App.tsx change below must be applied to that copy (or the full frontend folder must first be committed to this repository).

## Fix 1 — Vite import.meta.env typing

This branch adds `src/vite-env.d.ts` containing:

```ts
/// <reference types="vite/client" />
```

## Fix 2 — ReactNode conditional errors

In `src/App.tsx`, change the conditional around the generated image from:

```tsx
{result?.generated_2d_b64&&
```

to:

```tsx
{Boolean(result?.generated_2d_b64) &&
```

Also change the condition before the GLB download link from:

```tsx
{result?.paths && typeof result.paths==='object' && 'viewer_glb_url' in result.paths &&
```

to:

```tsx
{Boolean(result?.paths) && typeof result.paths==='object' && 'viewer_glb_url' in result.paths &&
```

These explicit boolean conditions prevent a value typed as `unknown` from being returned as a React child.

## Build

Run from the `frontend-react` directory:

```powershell
npm install
npm run build
```
