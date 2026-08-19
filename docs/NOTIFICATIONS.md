# Webhook Notifications

Get notified on your phone, Discord channel, or any HTTP endpoint when a translation finishes. Useful for long-running jobs with local models — translating a full novel can take hours, and watching a terminal is not how anyone wants to spend an afternoon.

> Compatible with **ntfy**, **gotify**, **Discord**, **Slack**, **Healthchecks.io**, or any service that accepts an HTTP request.

---

## Quick start

Add a single line to your `.env` and restart the app:

```bash
NOTIFY_WEBHOOK_URL=https://ntfy.sh/your-unique-topic-here
```

Install the [ntfy](https://ntfy.sh/) app on your phone, subscribe to the same topic, and you will get a push notification every time a translation completes.

That is the minimum. Everything else below is opt-in for richer formatting, custom endpoints, or fine-grained control over which events trigger a notification.

---

## Configuration reference

All variables are read from `.env` at startup.

| Variable | Default | Description |
| --- | --- | --- |
| `NOTIFY_WEBHOOK_URL` | *(empty)* | The endpoint to call. Leaving it empty disables notifications entirely. Supports placeholders (see below). |
| `NOTIFY_WEBHOOK_METHOD` | `POST` | HTTP method. `POST` works for almost all services; use `GET` for endpoints like Healthchecks.io that just need to be pinged. |
| `NOTIFY_WEBHOOK_HEADERS` | *(empty)* | JSON object of HTTP headers to send. Useful for `Authorization`, custom API keys, etc. Header values support placeholders. |
| `NOTIFY_WEBHOOK_PAYLOAD` | *(empty)* | JSON body template. Leave empty to use the default payload (compatible with gotify). String values inside the JSON support placeholders. |
| `NOTIFY_ON_SUCCESS` | `true` | Send a notification when a translation completes successfully. |
| `NOTIFY_ON_FAILURE` | `true` | Send a notification when a translation fails (exception, fatal error). |
| `NOTIFY_ON_INTERRUPTION` | `false` | Send a notification when a translation is paused/interrupted (by the user or by an unrecoverable rate-limit). Off by default because these states are usually intentional. |
| `NOTIFY_TIMEOUT_SECONDS` | `5` | HTTP timeout for the webhook call. Increase if you observe TLS handshake timeouts on slower networks. |

### Placeholders

Wherever you write `{name}` inside `NOTIFY_WEBHOOK_URL`, `NOTIFY_WEBHOOK_HEADERS`, or string values of `NOTIFY_WEBHOOK_PAYLOAD`, the following keys are substituted at notification time:

| Placeholder | Example value |
| --- | --- |
| `{event}` | `success`, `failure`, `interruption` |
| `{file}` | `book.epub` |
| `{output}` | `book (French).epub` |
| `{duration_seconds}` | `1234.56` (float, format with `{duration_seconds:.0f}` for `1235`) |
| `{provider}` | `ollama`, `gemini`, `openai`, ... |
| `{model}` | `qwen3:14b` |
| `{source_lang}` | `English` |
| `{target_lang}` | `French` |
| `{error}` | Error message (only on `failure` events; empty otherwise) |
| `{translation_id}` | Internal job ID (web jobs only) |

Unknown placeholders are left untouched so a malformed template will not silently drop information.

### Default payload (when `NOTIFY_WEBHOOK_PAYLOAD` is empty)

```json
{
  "title": "Translation completed",
  "message": "Event: success\nFile: book.epub\nDuration: 1234.6s",
  "event": "success",
  "file": "book.epub",
  "duration_seconds": 1234.56,
  "error": null
}
```

The `title` and `message` keys are the convention used by gotify, and ntfy also displays them sensibly when no topic-in-JSON mode is needed.

---

## Recipes by service

### ntfy.sh — push to phone (recommended for personal use)

1. Install the [ntfy app](https://ntfy.sh/) on your phone (iOS / Android).
2. Subscribe to a **unique** topic name (avoid common words — topics are public, anyone who guesses the name can send notifications to it). Example: `tbl-username-K8x9p2`.
3. In `.env`:

```bash
NOTIFY_WEBHOOK_URL=https://ntfy.sh/tbl-username-K8x9p2
NOTIFY_ON_SUCCESS=true
NOTIFY_ON_FAILURE=true
```

That is enough. You will receive a notification each time a translation completes or fails.

**Richer formatting** (priority, tag emoji, tap-to-open URL): ntfy supports JSON publishing only when posting to the bare `https://ntfy.sh/` endpoint with the topic inside the body — not when the topic is in the URL. So:

```bash
NOTIFY_WEBHOOK_URL=https://ntfy.sh
NOTIFY_WEBHOOK_PAYLOAD={"topic":"tbl-username-K8x9p2","title":"{event}","message":"{file} in {duration_seconds:.0f}s","priority":4,"tags":["white_check_mark"]}
```

See the [ntfy publishing docs](https://docs.ntfy.sh/publish/) for the full list of fields (priority levels, click action, attach files, etc.).

### gotify — self-hosted notification server

The original use case from [issue #167](https://github.com/hydropix/TranslateBookWithLLM/issues/167). Run gotify in Docker:

```bash
docker run -d -p 8080:80 --name gotify gotify/server
```

In the gotify web UI, create an **Application** and copy its token. Then:

```bash
NOTIFY_WEBHOOK_URL=https://gotify.example.com/message?token=YOUR_TOKEN_HERE
```

The default payload (`title` + `message`) maps directly to gotify's expected schema — no custom payload needed.

### Discord — channel webhook

1. In your Discord server, go to **Server Settings → Integrations → Webhooks → New Webhook**.
2. Pick a channel, copy the webhook URL.
3. In `.env`:

```bash
NOTIFY_WEBHOOK_URL=https://discord.com/api/webhooks/XXXXX/YYYYY
NOTIFY_WEBHOOK_PAYLOAD={"content":"Translation **{event}**: `{file}` in {duration_seconds:.0f}s"}
```

Discord expects a `content` field. Use Markdown for formatting.

### Slack — incoming webhook

1. Create an incoming webhook in your Slack workspace (Apps → Incoming Webhooks).
2. In `.env`:

```bash
NOTIFY_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
NOTIFY_WEBHOOK_PAYLOAD={"text":"Translation {event}: {file} ({duration_seconds:.0f}s)"}
```

### Healthchecks.io — cron monitoring

If you want to track that your translation pipeline ran successfully (e.g. for automated batch jobs), Healthchecks just needs a ping:

```bash
NOTIFY_WEBHOOK_URL=https://hc-ping.com/your-uuid-here
NOTIFY_ON_SUCCESS=true
NOTIFY_ON_FAILURE=false
```

Healthchecks will alert you if it does *not* receive a ping within your expected interval. To explicitly signal failure, append `/fail`:

```bash
NOTIFY_WEBHOOK_URL=https://hc-ping.com/your-uuid-here/{event}
```

The `{event}` placeholder will resolve to `success`, `failure`, or `interruption` — Healthchecks ignores unknown suffixes and treats them as success pings, while `/fail` triggers an alert immediately.

### Custom endpoint with authentication

For any service that needs a Bearer token or custom auth header:

```bash
NOTIFY_WEBHOOK_URL=https://api.example.com/notify
NOTIFY_WEBHOOK_HEADERS={"Authorization":"Bearer YOUR_TOKEN_HERE","X-Source":"TBL"}
NOTIFY_WEBHOOK_PAYLOAD={"event":"{event}","file":"{file}","duration":{duration_seconds:.1f}}
```

Header values are also templated, so you can inject context (e.g. `"X-Job-Id":"{translation_id}"`).

---

## When notifications fire

| Event | Trigger | Notify by default |
| --- | --- | --- |
| `success` | Translation completes successfully (CLI exit or web job marked `completed`). | yes |
| `failure` | Translation fails with an exception, fatal error, or finalization error. | yes |
| `interruption` | User pauses the translation, or an unrecoverable rate-limit auto-pauses the job. | no |

Transient states like `rate_limited` (auto-resuming) do **not** trigger notifications — they would be noisy and the job will recover on its own.

---

## Testing your configuration

After editing `.env`, test the webhook without running a full translation:

```bash
python -c "from src.utils.notifier import notify, EVENT_SUCCESS; print(notify(EVENT_SUCCESS, {'file':'test.txt','duration_seconds':1.0}))"
```

- `True` printed + notification received → working
- `True` printed + no notification → the request succeeded but the service rejected the payload format (check the service's expected schema)
- `False` printed → the request failed (URL wrong, auth missing, timeout); set `DEBUG_MODE=true` in `.env` to see the underlying error

You can also override values for one-off tests via environment variables:

```bash
NOTIFY_WEBHOOK_URL=https://ntfy.sh/test-topic NOTIFY_ON_SUCCESS=true \
  python -c "from src.utils.notifier import notify, EVENT_SUCCESS; notify(EVENT_SUCCESS)"
```

---

## Troubleshooting

**Nothing arrives but the script returns `True`.**
The HTTP call succeeded but the service silently rejected the payload. Most common cause with ntfy: posting JSON to `https://ntfy.sh/<topic>` — ntfy treats the body as plain text in that case. Either remove `NOTIFY_WEBHOOK_PAYLOAD` (use default text body) or post to `https://ntfy.sh` (bare) with the topic inside the JSON body.

**Script returns `False` with a `ReadTimeout` / `handshake operation timed out`.**
Your network is slow to establish HTTPS to the webhook host. Bump `NOTIFY_TIMEOUT_SECONDS=30`. If it still fails, your firewall or antivirus is intercepting HTTPS traffic — test with `curl -d test https://your-webhook` outside Python to confirm.

**`False` with HTTP 4xx in the logs.**
Payload format does not match what the service expects. Check the service's documentation and adjust `NOTIFY_WEBHOOK_PAYLOAD`. Discord, for instance, requires a `content` field; Slack requires `text`.

**Notifications during interrupted/paused jobs.**
Set `NOTIFY_ON_INTERRUPTION=true` if you want them. Off by default because pausing is a deliberate user action.

**A failed webhook is breaking my translation.**
It cannot — the notifier swallows all HTTP and network exceptions and only logs a warning. If your translation is failing because of the webhook, please open an issue with `DEBUG_MODE=true` logs.

---

## Privacy note

Webhook URLs often contain authentication tokens (e.g. gotify's `?token=`, Discord's webhook path). Anyone with read access to your `.env` can use them to send notifications to your channel. Treat them as secrets:

- Never commit `.env` to git (it is already in `.gitignore`).
- Never paste a real webhook URL in an issue, PR, or chat message — replace the secret portion with `YOUR_TOKEN_HERE`.
- If a webhook URL leaks, rotate it: regenerate the token in the service's UI.
