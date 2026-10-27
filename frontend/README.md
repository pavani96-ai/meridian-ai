# Meridian frontend

React 18.3, TypeScript 5.8, Vite 5.4.21, Tailwind 3.4, Lucide, and shadcn-style Radix components.

Start FastAPI from `meridian-ai`:

```bash
uv run uvicorn api.main:app --app-dir backend --reload --port 8080
```

In a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open the Vite URL (normally http://localhost:5173). `/api` requests proxy to FastAPI on port 8080; credentials remain in the backend.

Build with `npm run build`. Restart FastAPI after building: it serves `frontend/dist` on port 8080. No separate frontend server is needed for that mode.

## Features

- Procurement request form, sample input, specialist JSON reports, CFO memorandum download.
- Multiple PDF uploads, per-file results, process-local upload history, stored-PDF reindexing.
- RAG questions with similarity, multi-query, or contextual retrieval.
- Configuration/status screen and backend connectivity indicator.

Screens show real API results only. Audits do not stream phase progress, and live AI/search calls can fail or take time. Audit results stay only in browser memory; document history resets when its component reloads and is then fetched from the backend. The backend's upload history resets on server restart. No authentication is provided by this frontend or the current API.

## Browser tests

```bash
npx playwright install chromium
npm run test:e2e
```

The tests mock API responses and cover audit submission, memo download, API failures, PDF upload, RAG retrieval, stored ingestion, system status, and mobile layout. They do not test live Gemini or backend services.
