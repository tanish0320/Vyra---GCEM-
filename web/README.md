# VYRA React interface for Integration

The frontend source is `web/`. It retains the local eye film, Three.js anatomy,
readable pearl/teal design and screening workflow, with capability-gated UI for
dual capture, image colour measurements, the assistant and research tools.

**Backend connection is a separate task:** read
[the backend handoff](../docs/FRONTEND_BACKEND_HANDOFF.md). The existing Flask
backend is unchanged and does not yet expose every job/history endpoint this UI
needs. Missing capabilities remain unavailable; production includes no fixture
predictions or generated medical answers.

```sh
cd web
npm ci
npm test
npm run build:integration
```

Start the existing Flask application from its root with `python app.py`. Open
`http://127.0.0.1:5000/static/ui/index.html`. All assets are local after setup.
The integration build is generated in `person3_app/static/ui/`; keep its prefix.

For frontend development, copy `.env.example` to `.env.local`, set the Flask
proxy address and run `npm run dev`. For checking the static integration build,
run `npm run preview:integration`. API requests remain same-origin in production.

Browser tests use Playwright. Set `PLAYWRIGHT_CHROMIUM_EXECUTABLE` if necessary,
or install Chromium with `npx playwright install chromium`. New-feature tests
are fixture-driven and do not establish backend or model correctness. Runtime
screens never load those test fixtures. Browser evidence belongs outside source.

The optional live-backend suite needs the compatibility endpoints and model
files installed by the backend developer. No raw dataset is required to build
this frontend. Patient records and model weights are not copied into this source.

UI primitives are adapted from shadcn/ui (MIT); React Bits supplies the existing
motion components. Space Grotesk/IBM Plex Sans fonts and anatomical assets are
local. Eye asset attribution and modifications remain in
`public/anatomy/manifest.json` and the in-app credits. See
[asset notices](THIRD_PARTY_NOTICES.md).

Run the isolated frontend acceptance suite after building:

```sh
npm run test:browser -- --config playwright.ui.config.ts tests/browser/integration.spec.ts
```

`playwright.app.config.ts` / `playwright.live.config.ts` target an already running
backend on5000 (or `VYRA_BASE_URL`). They never start or provision Python. Use an
isolated backend test runtime before running the live suite.
