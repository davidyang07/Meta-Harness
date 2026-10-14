# Meta-Harness dashboard

A Next.js 16 app that renders a Meta-Harness run live: the search tree of
candidates across every branch, the decision log the outer loop streams
over SSE, per-candidate diffs and test output, and cross-run memory. It
talks to the FastAPI backend directly from the browser; it has no API of
its own.

Read [`AGENTS.md`](AGENTS.md) before changing code: this Next.js version
differs from older ones (for example, `middleware.ts` is now `proxy.ts`).

## Run it

Start the backend from the repository root. Use `meta-harness serve`, not a
bare `uvicorn`, so persistence comes up on an event loop psycopg can use:

```bash
uv sync
docker compose -f infra/docker-compose.yml up -d postgres   # optional: real checkpoints
uv run meta-harness serve --port 8000
```

Then the dashboard:

```bash
cd frontend/dashboard
npm ci
cp .env.example .env.local      # optional; every variable has a default
npm run dev                     # http://localhost:3000
```

The home page lists the backend's runs and launches preset runs. The mock
proposer with the mock benchmark costs nothing; its numbers are labelled
"Mock data (synthetic)" in the status bar wherever they appear.

With no backend reachable, `/runs/demo-2026-04-25` plays a built-in
offline demo. Any other run shows an explicit "backend unavailable"
error rather than fixture data.

## Configuration

All variables are optional. See [`.env.example`](.env.example).

| Variable | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | Where the backend listens. |
| `NEXT_PUBLIC_META_HARNESS_API_TOKEN` | empty | Must match the backend's `META_HARNESS_API_TOKEN` when that is set. Sent as `Authorization: Bearer …` on API calls and as `?token=…` on the event stream. A missing or wrong token shows a 401 message in the status bar. |
| `AUTH0_DOMAIN`, `AUTH0_CLIENT_ID`, `AUTH0_CLIENT_SECRET` (or `AUTH0_CLIENT_ASSERTION_SIGNING_KEY`), `AUTH0_SECRET`, `APP_BASE_URL` | unset | Optional Auth0 login in front of every page, including `/`. Enabled only when all of them are set; otherwise the dashboard is open and shows no Login/Logout links. |

`NEXT_PUBLIC_*` values are compiled into the browser bundle when `next dev`
starts or `next build` runs, so restart after changing them. The API token
is therefore visible to anyone who can load the dashboard. It keeps other
machines off the backend; it is not a user login.

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Development server on port 3000 (React StrictMode on). |
| `npm run build` / `npm run start` | Production build and server. |
| `npm run lint` | ESLint. |
| `npx tsc --noEmit` | Type-check, including the e2e specs. |
| `npm run test:e2e` | Every Playwright project; see below. |

## End-to-end tests

Playwright starts `npm run dev` itself, or reuses a server already on the
port. Set `DASHBOARD_PORT` to use a port other than 3000 for the `mock`
project; do so whenever another checkout's dev server might be on 3000,
or the tests will run against that checkout's code. The `live-backend`
project must stay on 3000: the backend's CORS policy admits the
dashboard only from `localhost:3000` / `127.0.0.1:3000`.

```bash
npx playwright install chromium                       # once
npx playwright test --project=mock                    # no backend needed
uv run meta-harness serve --port 8000                 # then, for the live project:
npx playwright test --project=live-backend
```

- `mock` needs no backend. `e2e/dashboard.spec.ts` renders the offline
  demo with the backend deliberately unreachable. `e2e/fixture-backend.spec.ts`
  scripts every API response and drives a fake `EventSource` event by
  event, pinning down candidate verdicts, per-thread frontiers, fork
  de-duplication, the decision log and the stream's lifecycle.
- `live-backend` runs a real (mock-proposer, mock-benchmark) run through
  the real backend and checks runs, checkpoints, forks, branch isolation,
  diffs and memory. It skips itself if no backend answers at
  `NEXT_PUBLIC_API_BASE_URL`. If the backend has `META_HARNESS_API_TOKEN`
  set, export the same value as `NEXT_PUBLIC_META_HARNESS_API_TOKEN`.
