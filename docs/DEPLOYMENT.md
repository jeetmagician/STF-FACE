# Deployment

## Local development

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
python scripts/selfcheck.py
uvicorn app.main:app --reload --port 8000
```

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

## Docker

```bash
cp .env.example .env
docker compose up --build
```

Frontend on 3000, backend on 8000. Weights are baked at build time, so the
containers need no network access at runtime.

For the InsightFace backend:

```bash
docker compose build --build-arg INSTALL_INSIGHTFACE=true backend
MODEL_BACKEND=insightface docker compose up
```

Remember the licence: those weights are non-commercial research use only.

### GPU

```bash
cd backend
docker build -f Dockerfile.gpu -t facet-backend:gpu .
docker run --gpus all -p 8000:8000 \
  -e USE_GPU=true -e MODEL_BACKEND=insightface \
  facet-backend:gpu
```

GPU helps throughput under concurrency. For a single comparison at a time, CPU
inference is already fast enough that the difference is not worth the operational
cost.

---

## Production checklist

### Must do

- [ ] **Calibrate.** An uncalibrated deployment reports assumptions as
      measurements. See [CALIBRATION.md](CALIBRATION.md).
- [ ] **Set `API_KEY`.** Without it the API is open to anyone who can reach it.
- [ ] **Set `ENVIRONMENT=production`.** Disables `/docs` and enables proxy
      header trust.
- [ ] **Restrict `CORS_ORIGINS`** to your actual frontend origin.
- [ ] **Terminate TLS.** Photographs must not cross a network in the clear.
- [ ] **Replace the rate limiter** (see below).
- [ ] **Review proxy logs** — make sure your reverse proxy is not logging more
      than you intend.

### Should do

- [ ] Set `PRIOR_SAME_PERSON` to reflect how your pairs are selected.
- [ ] Pin model hashes: `python scripts/download_models.py --write-hashes`,
      then commit `model_hashes.json`.
- [ ] Tune quality gates to your image sources.
- [ ] Set `RETAIN_FOR_FACE_SELECTION=false` if you want zero cross-request
      state at the cost of a re-upload on face selection.
- [ ] Monitor `/api/health` — it reports whether calibration is validated.

---

## The API key and the frontend

`NEXT_PUBLIC_API_KEY` is **inlined into the JavaScript bundle** and visible to
anyone who loads the page. That is fine for a local prototype and unacceptable
for a public deployment.

For anything public, proxy through a Next.js route handler that holds the key
server-side:

```ts
// app/api/analyze/route.ts
export const runtime = "nodejs";

export async function POST(request: Request) {
  const body = await request.formData();
  const upstream = await fetch(`${process.env.API_URL}/api/analyze`, {
    method: "POST",
    body,
    headers: { "X-API-Key": process.env.API_KEY! },  // server-side only
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
```

Then point `NEXT_PUBLIC_API_URL` at your own origin and drop
`NEXT_PUBLIC_API_KEY` entirely. Note that `API_URL` and `API_KEY` here have no
`NEXT_PUBLIC_` prefix, which is what keeps them out of the bundle.

---

## Rate limiting behind a load balancer

`core/security.py` implements an in-process sliding window. With N instances
behind a balancer, each keeps its own counters and the effective limit becomes
N times what you configured.

Replace with a shared store:

```python
import redis.asyncio as redis

class RedisLimiter:
    def __init__(self, url: str) -> None:
        self._client = redis.from_url(url)

    async def check(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        pipeline = self._client.pipeline()
        pipeline.incr(key)
        pipeline.expire(key, window, nx=True)
        count, _ = await pipeline.execute()
        if count > limit:
            return False, int(await self._client.ttl(key))
        return True, 0
```

Swap it into `RateLimitMiddleware`. The interface is deliberately two methods
so this is a small change.

Also set `app.state.trust_proxy_headers = True` (automatic when
`ENVIRONMENT=production`) so `X-Forwarded-For` is used for client
identification — and make sure your balancer strips client-supplied values of
that header, or callers can spoof it to evade limits.

---

## Scaling

The model loads once per process and is reused. Memory is roughly 400 MB per
worker for the OpenCV backend, 1.2 GB for InsightFace.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Each worker holds its own model copy, so size worker count against memory
rather than CPU count. With `RETAIN_FOR_FACE_SELECTION=true` and multiple
workers you also need sticky sessions, because the ephemeral store is
per-process and a face-selection resume routed to a different worker will 404.
Either enable session affinity or set the flag to `false`.

Indicative CPU latency per comparison:

| Backend | 1 photo/side | 3 photos/side |
|---|---|---|
| opencv | ~0.3 s | ~0.8 s |
| insightface | ~0.9 s | ~2.4 s |

Measure on your own hardware; these are order-of-magnitude figures.

---

## Reverse proxy

```nginx
server {
    listen 443 ssl http2;
    server_name facet.example.com;

    client_max_body_size 64M;   # 5 photos per side at 12 MB each

    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        # Overwrite rather than append: a client-supplied value must not survive.
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }

    location / {
        proxy_pass http://frontend:3000;
        proxy_set_header Host $host;
    }
}
```

---

## Monitoring

`GET /api/health` returns:

```json
{
  "status": "ok",
  "model_loaded": true,
  "calibration_validated": false,
  "active_sessions": 0,
  "notes": ["Calibration is unvalidated...", "..."]
}
```

Alert on `model_loaded: false`. Treat `calibration_validated: false` in
production as a defect, not a warning — it means your users are being shown
numbers that describe an assumption.

`active_sessions` should hover near zero. A rising count means analyses are
starting and not completing.

Worth tracking:

- Rate of `insufficient_quality` refusals — a jump means an input source
  changed.
- Rate of `multiple_faces_detected` — high values suggest users are uploading
  group photographs.
- Score distribution over time. Drift indicates your population or capture
  conditions have moved and calibration needs refitting.

---

## Backups

There is nothing to back up except:

- `backend/assets/calibration/*.json` — your fitted calibration. Losing this
  means refitting, which means re-running your validation set.
- `backend/assets/models/model_hashes.json` — your pinned weight hashes.

No photographs are stored, so there is no image backup, and no image restore
that could resurrect data a user expected to be gone.
