# Deployment

## Architecture

1. A push to `main` runs CI (tests, quality gate, Docker smoke test).
2. CD builds the slim Docker image and pushes it to `ghcr.io/niksi-nova/mlops`, tagged `sha-<short sha>` and `latest`.
3. CD then calls a Render deploy hook, telling Render's free web service to pull and run this run's `sha-<short sha>` image tag.
4. At container startup, the API loads the model tagged `production` from the W&B registry (`delhi-bus-delay-model`) — or a mounted `MODEL_PATH` file instead, if one's configured.
5. CD polls the live Render URL's `/health` until `model_loaded` is `true`, or fails after 12 minutes.

## One-time setup

1. **Make the GHCR package public.** After CD's first successful push, go to the package's settings on GitHub (`niksi-nova` org/user → Packages → `mlops`) and change visibility to public. Render's free tier can't pull a private image without extra credentials this project doesn't set up — see the troubleshooting note below.
2. **Create the Render web service** from an existing image: `ghcr.io/niksi-nova/mlops:latest`, free instance type.
3. **Set these environment variables on the Render service**:

   | Variable | Required? | Purpose |
   |---|---|---|
   | `WANDB_API_KEY` | yes (unless using `MODEL_PATH`) | W&B auth token. |
   | `WANDB_BASE_URL` | yes, if using a non-default W&B host (e.g. `https://api.forge.coreweave.com`) | Which W&B instance to talk to. |
   | `MODEL_ALIAS` | yes | Set to `production`. |
   | `WANDB_ENTITY` | no | Overrides the registry workspace to look the alias up in. Defaults to `anikaub20-rv-college` if unset. |
   | `WANDB_PROJECT` | no | Overrides the registry project. Defaults to `delhi-bus-delay` if unset. |
   | `MODEL_PATH` | no | Set instead of the W&B variables to load from a path inside the image/a mounted disk, skipping live W&B access entirely. |
4. **Add to this repo**: secret `RENDER_DEPLOY_HOOK` (from the Render service's Settings → Deploy Hook) and repository **variable** `RENDER_URL` (the service's public URL, e.g. `https://delhi-bus-delay-api.onrender.com`).

## Rolling back

- **Redeploy an older image**: trigger the Render deploy hook yourself with `imgURL` set to an older `ghcr.io/niksi-nova/mlops:sha-<short sha>` tag (see any past CD run's job summary for the exact tag), or use Render's dashboard to redeploy a previous build.
- **Re-promote an older registry version**: if the issue is the model itself rather than the API code, run `uv run python -m src.models.registry --promote-production --approved-by "Name"` against an earlier artifact version instead of touching the deployment at all — the API picks up whatever currently holds the `production` alias the next time it restarts (or loads it fresh at startup).

## Render free tier constraints

- **512MB RAM.** Measured peak memory in W&B mode, under `--memory=512m --memory-swap=512m`, hammering `/predict` and 500-row `/predict/batch` calls repeatedly: ~272MiB, stable, no growth, never OOM-killed. Comfortable headroom under the 512MB limit with no tuning needed.
- **Sleeps after 15 minutes of inactivity**, and takes about a minute to wake on the next request. The CD deploy's health poll accounts for this (12-minute budget, 15s between checks).

## Promotion procedure

Because `register_model()`/`promote_to_production()` silently fall back to `local_fallback` mode when no live W&B session is active — only updating local files, never the remote artifact's alias — always:

1. Run `uv run wandb login` (or otherwise confirm you're logged in) **before** running `uv run python -m src.models.registry` or `--promote-production`. If you're not logged in, the command still "succeeds" but only writes `reports/model_registry.json`/`reports/production_approvals.json` locally — the real W&B artifact never gets the alias.
2. **After promoting**, check in the W&B UI (or `wandb.Api().artifacts(...)`) that the artifact actually carries the `production` alias remotely. This project hit exactly this gap once: a local approval existed with no matching remote alias, so a container running in pure W&B mode (no `MODEL_PATH`) couldn't find anything to serve until the alias was attached to the artifact directly in W&B.

## Troubleshooting

If the CD deploy's health poll fails, check the job summary and logs first. The most common cause: `ghcr.io/niksi-nova/mlops` is a **private** GHCR package, so Render's image pull fails (Render's free tier doesn't support authenticating to a private registry) — making the package public fixes this.

### Diagnosing a model-load failure

A failed load now surfaces, instead of silently vanishing:
- **At startup**, one `model_load_config` log line shows whether `WANDB_API_KEY` is set (`true`/`false` only, never the value), the `WANDB_BASE_URL` host, the resolved entity/project/collection/alias, and whether `MODEL_PATH` is set.
- If the W&B fetch fails, a `WARNING` log line names the exact stage that failed (login, API client creation, artifact lookup, alias check, or download) along with the exception class and a redacted message.
- `GET /health`'s 503 body and the final startup log line both include a short, redacted `load_error`.

This was added after a real incident: Render logged only
`"Production model artifact not found on disk at /app/models/best_model.joblib or in the W&B registry."`
with no indication of *why* the remote fallback failed. Reproducing the same `MODEL_ALIAS=production`, env-vars-only configuration locally (pulling the published `ghcr.io/niksi-nova/mlops:latest` image, no mounted files) actually **succeeded** both with and without `WANDB_BASE_URL` set — so the failure is specific to Render's own network/runtime environment and could not be reproduced from a developer machine. The new logging is what makes the *next* occurrence diagnosable directly from Render's logs and `/health` response, without needing to reproduce it elsewhere.

### Diagnosing a stuck or failed CD `deploy` job

The `deploy` job prints one line per poll attempt (`attempt N: http_status=... model_loaded=...`, or `curl failed (exit code ...)` if the request didn't even complete) — check the job log for the *last* attempt line to see exactly how far it got. Every `curl` call in this job has `--connect-timeout 10 --max-time 20`, so no single call can hang the job; the overall poll loop is bounded to 12 minutes and the job itself times out at 20. Specific failure shapes:

- **Hook rejected**: the "Trigger Render deploy hook" step logs `Render deploy hook HTTP status: <code>` and fails if that code isn't 2xx. The hook URL itself (including its `key` query parameter) is never logged, only the status.
- **Hook call itself failed** (unreachable, timed out, malformed): logged as a nonzero curl exit code, with a reminder to check the `RENDER_DEPLOY_HOOK` secret's value directly in Render's dashboard rather than trying to debug it from CI output.
- **Timeout** (12 minutes of polling with no `http_status=200, model_loaded=true`): the last response body is printed in the final error, which is enough to tell apart a `503` (service not up yet / crashed), `model_loaded: false` with a `load_error` (see above), or an empty body (likely a wrong `RENDER_URL` — check the scheme and host, e.g. `https://your-service.onrender.com`, no trailing path).
- **Image tag**: the job reads the actual `sha-*` tag out of `docker/metadata-action`'s own JSON output (not a hand-recomputed guess), so the tag sent to Render's `imgURL` is guaranteed to be one that was actually pushed in this run. The job summary prints which image tag was deployed.
- **Confirming the *new* image is what's live**: `/model-info` exposes the deployed *model's* git commit (from the registry metadata), not the API/image's own git commit — those can legitimately differ, since a model promotion and an API code deploy are independent events. There's currently no field that identifies which image build is actually running, so this job can only confirm the service is healthy after the hook fires, not that it's specifically running the new image rather than a slow-to-replace old one. If that distinction matters, check Render's own deploy history/logs directly.
