# Деплой на сервер (GPU + публичный домен)

```
жюри → https://madagascar-ai.team → VPS (Caddy: HTTPS, домен)
                                      │ Tailscale
                                      ▼
                    RTX 5090: Forte Guard (API + YOLO на GPU) → Gemma (Ollama, локально)
```

VPS только принимает HTTPS и пересылает запросы через Tailscale; видео обрабатываются и хранятся на машине команды с GPU. Сервис на 5090 слушает только адрес Tailscale (`100.118.149.106:8000`).

## CI/CD

`.github/workflows/deploy.yml` запускается после зелёного `Tests` на `main` (или вручную, Actions → Deploy → Run workflow):

1. подключается к tailnet временным узлом (`TS_AUTHKEY`);
2. загружает коммит на 5090 в `~/forte-guard/releases/<sha>/`;
3. `deploy/gpu/activate.sh`: ждёт завершения задач в очереди (до 4 мин), переключает `~/forte-guard/app` на новый релиз, перезапускает службу, проверяет `/health` (нужный SHA) и readiness; при ошибке возвращает предыдущий релиз; хранит последние пять;
4. обновляет `deploy/vps/` на VPS и перезагружает Caddy;
5. проверяет `https://madagascar-ai.team/health` (предупреждение, а не ошибка, пока DNS или сертификат не готовы).

Секреты репозитория: `TS_AUTHKEY` (Tailscale auth key, reusable + ephemeral), `DEPLOY_GPU_SSH_KEY`, `DEPLOY_VPS_SSH_KEY`. Отпечатки серверов закреплены в `deploy/known_hosts`. Без `TS_AUTHKEY` деплой пропускается.

## Настройки сервера в GitHub

Настройки сервиса на 5090 задаются в GitHub: **Settings → Secrets and variables → Actions**.

- **Secrets** — секретные значения, например `FORTE_API_KEY` (ключ жюри). В логах скрыты.
- **Variables** — обычные настройки: `FORTE_DETECTOR`, `FORTE_MODEL_PATH`, `FORTE_LLM_MODEL`, `FORTE_ZONE_MODEL`, `FORTE_WEB_ACCESS`, `FORTE_SAMPLE_FPS`, `FORTE_LLM_ENABLED`, `FORTE_AUTO_ZONES`, `FORTE_CASHIER_PRESENCE`, `FORTE_JOB_TIMEOUT_SEC`.

При каждом деплое все имена с префиксом `FORTE_` (из Secrets и Variables; при совпадении побеждает Secret) записываются поверх `shared/.env`: совпадающие строки заменяются, новые добавляются, остальное на сервере не трогается. Служебные секреты (`DEPLOY_*`, `TS_AUTHKEY`) на сервер не попадают. В журнале деплоя печатаются только имена изменённых настроек. Если новая версия не прошла проверку, вместе с кодом возвращается и прежний `.env`.

Чтобы применить изменённую настройку без нового коммита: Actions → Deploy → Run workflow. Значения, привязанные к машине (`FORTE_HOST`, `FORTE_PORT`, `FORTE_DATA_DIR`, `FORTE_OLLAMA_URL`, `FORTE_CONFIG_DIR`), держите только в `shared/.env` на сервере: ошибка в них отключит сервис.

## Сервер с GPU (RTX 5090)

```
~/forte-guard/
  app -> releases/<sha>     активная версия
  releases/<sha>/           код релизов
  venv/                     PyTorch (cu128), ultralytics, зависимости сервиса
  shared/.env               настройки и API-ключ жюри (права 600)
  shared/models/            веса детектора (detector-v2.pt)
  shared/data/              задачи и результаты
```

Служба: `systemctl --user status|restart forte-guard` (unit — `deploy/gpu/forte-guard.service`, linger включён). Журнал: `journalctl --user -u forte-guard -f`. Ручная активация релиза: `bash ~/forte-guard/releases/<sha>/deploy/gpu/activate.sh <sha>`.

Новые веса не приезжают через git: положить файл в `shared/models/` и указать его в переменной `FORTE_MODEL_PATH` в GitHub, затем запустить Deploy.

Сайт публичный, поэтому в `shared/.env` задайте `FORTE_WEB_ACCESS=guest`: посетители загружают видео без ключа и видят только свои проверки, а история (включая видео жюри), бенчмарк и конфигурация остаются за `X-API-Key`. Режим `open` здесь не используйте — он открывает всю историю любому посетителю. Это прод: общей очереди, истории и бенчмарка на нём нет. Для команды — dev-сервис на Mac (`http://10.70.47.208:8000`, [autodeploy.md](autodeploy.md)) с общей очередью и историей.

## VPS

`~/forte-proxy/` — `deploy/vps/docker-compose.yml` и `Caddyfile`: контейнер Tailscale (порты 80/443) и Caddy в его сети, сертификат Let's Encrypt для `madagascar-ai.team` и `www`. DNS: записи `A @` и `A www` → `77.42.123.206`.
