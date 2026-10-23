# Frontend integration validation

Validated on 24 October 2026 in the isolated `Integration` checkout based on
`67b3b6909965411020dc662a0e86b0fc2ca5938b`.

## Results

| Check                                 | Result                                                                    |
| ------------------------------------- | ------------------------------------------------------------------------- |
| TypeScript and Vite integration build | Passed; entry/assets under `/static/ui/`                                  |
| Frontend domain/anatomy unit cases    | 15 passed across two unchanged unit files                                 |
| Browser acceptance/regression suite   | 132 passed,0 failed,0 skipped,0 flaky; retries disabled                   |
| Viewports                             | 1440×1000 desktop,390×844 phone,768×1024 tablet                           |
| Formatting                            | Prettier passed for source, browser tests and configs                     |
| Source asset integrity                | Eye/nail GLB, manifest and poster hashes match provenance                 |
| Backend code/schema preservation      | No tracked Python, model, schema, template or backend-test changes        |
| Branch isolation                      | Original workspace stays on `vikas`; delivery checkout uses `Integration` |

Browser report: `/tmp/vyra-integration-ui-results.json`. Final run began at
`2026-10-23T20:18:45.864Z` and took 245.27 seconds. Screenshots are under
`/tmp/vyra-integration-evidence/` and the existing anatomy/workflow evidence paths.
These local report paths are not shipped as production data.

## Verified behavior

- Real browser canvas-camera frames drive brightness/detail telemetry; capture
  releases media tracks. Camera permissions/unavailability and keyboard closure
  remain covered.
- Dual uploads send both images, require valid reviewed boxes for both tissues,
  block a missing secondary image, and stay unavailable on unsupported services.
- Legacy single-image upload/retry, idempotency, crop review, cancellation, job
  restoration and stale-response handling continue to work against UI fixtures.
- Result colour measurements require real-pixel provenance. Missing/placeholder
  values are withheld; nail Hb is withheld; failed secondary results cannot show
  a combined score. Percent/fraction fields are formatted according to contract.
- Grad-CAM split/blend controls, raw/precomposited labeling, PDF URL and linked
  React stress workspace work as expected.
- English/Hindi assistant requests, source labels, safe bold/bullet formatting,
  emergency notices, service errors and stale-response cancellation are covered.
  An API error never becomes an invented fallback answer.
- The stress screen validates eight conditions, displays server units correctly,
  rejects malformed reports and leaves the original result separate.
- Research/analytics/docs navigation, missing-capability states, responsive
  layouts, local assets, film fallbacks and the existing anatomy controls pass.

## Commands

From `web/`:

```sh
npm test
node --experimental-strip-types tests/domain.test.ts
node --experimental-strip-types tests/anatomy.test.ts
npm run build:integration
npm run test:browser -- --config playwright.ui.config.ts tests/browser/integration.spec.ts tests/browser/site.spec.ts tests/browser/workflow.spec.ts tests/browser/anatomy.spec.ts tests/browser/redesign.spec.ts tests/browser/local-assets.spec.ts
npm run format:check
```

The browser suite uses explicitly test-only request fixtures. It exercises the
built production UI/assets, not a live medical backend. Browser paths use the
actual `/static/ui/` prefix so hash navigation retains the same application and
can meaningfully test delayed job responses.

Vite retains the pre-existing large-chunk advisory for the lazy Three.js renderer
and the application entry bundle. The anatomical renderer remains lazy-loaded;
this advisory is not a failed build.

## Pending on the backend computer

The following were **not** executed here: `pytest person3_app/tests/ tests/`, real
model uploads/dual inference, real PHC PDF generation, real eight-condition model
experiments, GPU/Ollama/Qwen, clinical review or raw-dataset evaluation. Backend
compatibility endpoints still need implementation. Follow the exact contracts,
identified backend issues and acceptance checklist in
[FRONTEND_BACKEND_HANDOFF.md](FRONTEND_BACKEND_HANDOFF.md).

No generated static bundle, dependencies, runtime patient data, model weights,
local browser executable paths or screenshots are committed. The generated bundle
is available locally and is reproduced with `npm run build:integration`.
