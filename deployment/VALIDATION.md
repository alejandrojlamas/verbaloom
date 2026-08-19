# Docker validation

This checklist validates the VerbaLoom container build, its local-first network
boundary, and persistent runtime data.

## Validated build corrections

- `requirements.txt`, `translation_api.py`, `translate.py`, and `src/` are
  copied from the repository build context.
- The image uses Python 3.11 slim.
- `.dockerignore` excludes local environments, logs, runtime data, tests,
  development plans, Git metadata, bytecode, and `.env` files.

## Automated validation

On Windows:

```cmd
cd deployment
test_docker.bat
```

On Linux or macOS:

```bash
cd deployment
chmod +x test_docker.sh
./test_docker.sh
```

## Manual validation

1. Start Docker Desktop.
2. Build and start VerbaLoom:

   ```bash
   cd deployment
   docker compose build
   docker compose up -d
   docker compose ps
   ```

3. Confirm that the service is healthy and exposed only on loopback:

   ```bash
   curl --fail http://127.0.0.1:5000/api/health
   docker compose ps
   ```

4. Open `http://127.0.0.1:5000` in a browser.
5. Upload a small synthetic text file and verify that a translation starts.
6. Inspect logs without printing provider credentials:

   ```bash
   docker compose logs --tail=200 verbaloom
   ```

## Image contents

The image must contain:

- `/app/requirements.txt`
- `/app/src/`
- `/app/translation_api.py`
- `/app/translate.py`

It must not contain local `.env` files, Git history, tests, runtime data, logs,
or development plans.

Inspect the relevant paths with:

```bash
docker compose exec verbaloom ls -la /app/
docker compose exec verbaloom ls -la /app/src/
docker compose exec verbaloom ls -la /app/src/prompts/
```

## Persistence check

```bash
docker compose down
docker compose up -d
docker compose exec verbaloom ls -la /app/data/
```

The service should return to a healthy state and retain its checkpoint database
and job history.

## Troubleshooting

- For a missing build file, confirm that the command runs from `deployment/`
  and that the repository root contains `requirements.txt` and `src/`.
- For a restart loop, inspect `docker compose logs verbaloom`.
- For an early health-check failure, wait for the configured start period and
  retry `curl --fail http://127.0.0.1:5000/api/health`.

Validation is complete only when the image builds, the container is healthy,
the loopback endpoint responds, the browser UI loads, and runtime data survives
a container restart.
