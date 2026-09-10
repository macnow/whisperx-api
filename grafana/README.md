# Grafana dashboard

`whisperx-dashboard.json` is a ready-to-import Grafana dashboard for the
`/metrics` endpoint exposed by whisperx-api (see README "Metrics").

## Import

1. Add a Prometheus datasource in Grafana pointed at your Prometheus server
   that scrapes `GET /metrics` from whisperx-api.
2. Grafana UI → **Dashboards → New → Import** → upload
   `whisperx-dashboard.json` (or paste its contents).
3. When prompted, select your Prometheus datasource for the `DS_PROMETHEUS`
   input.

## Example Prometheus scrape config

```yaml
scrape_configs:
  - job_name: whisperx-api
    metrics_path: /metrics
    static_configs:
      - targets: ["whisperx-api:8000"]
```

## Panels

- **Overview** — active transcriptions, free GPU memory, error rate,
  request rate, request duration p95, median transcribe speed (x realtime).
- **Requests** — request rate by endpoint/status; request duration
  (p50/p95/p99) by endpoint.
- **Transcription throughput** — `whisperx_transcribe_speed_ratio` (realtime
  factor) per model/executor thread — use this to see whether raising
  `TRANSCRIBE_CONCURRENCY`/`MAX_THREADS` is actually improving per-thread
  throughput; busy time per thread; audio seconds transcribed per second by
  model.
- **Model pools & errors** — loaded vs. idle whisper instances per model
  (`WHISPER_POOLS`); error rate by stage (`load_audio` / `processing`).
- **Capacity, cost & stage breakdown** — audio minutes processed and
  estimated GPU cost for the selected range (requires `GPU_HOURLY_COST_USD`),
  cost per audio-minute, executor saturation %, per-stage
  (transcribe/align/diarize) time-share, and whisper pool capacity
  utilization % by model.
- **Cold starts, queueing & audio quality** — model load duration (p95) and
  load/eviction rate per kind (whisper/align/diarize), pool wait time (p95)
  by model, median audio duration/upload size, median speakers detected,
  and request rate by detected language.

