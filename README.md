# speechParser

Telegram-бот для расшифровки записей МТС Линк и загруженных аудио/видео файлов. Бот запускается локально; загрузка записи по ссылке и распознавание выполняются на отдельном сервере, после чего бот возвращает `.txt` в чат.

## Архитектура

- `bot.py` работает на локальном компьютере, принимает сообщения Telegram и отправляет задания обработчику по HTTP.
- `worker_server.py` запускается на VPS, скачивает записи МТС Линк, выполняет транскрибацию и отдаёт статус и результат.
- Для приватных записей cookie `MTS_SESSION_ID` задаётся только на VPS.
- Локальная очередь, команда `/stop` и отправка частичного результата сохраняются.

## Обработчик на VPS

На сервере установи Docker и Docker Compose, затем клонируй репозиторий. Создай конфигурацию обработчика:

```bash
cp .env.server.example .env.server
openssl rand -hex 32
nano .env.server
```

Впиши созданный секрет в `PROCESSOR_API_KEY`. Для приватных записей добавь cookie `access` в `MTS_SESSION_ID`. Значение API-ключа потребуется также в локальном `.env`.

Собери и запусти обработчик:

```bash
docker compose up -d --build
docker compose logs -f processor
```

Compose публикует API только на `127.0.0.1:8080`. Настрой HTTPS reverse proxy на VPS с upstream `127.0.0.1:8080`, например для Caddy:

```caddy
processor.example.com {
	reverse_proxy 127.0.0.1:8080
}
```

В `PROCESSOR_URL` на локальном компьютере укажи публичный HTTPS-домен. Не открывай порт `8080` напрямую в интернет: API-ключ защищает запросы, но не шифрует передаваемые аудио и ключ.

Whisper-модель кэшируется в Docker volume `whisper_cache`, а файлы заданий — в `processor_data`.

## Локальный бот

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
nano .env
python bot.py
```

В `.env` задай `BOT_TOKEN`, `ALLOWED_USER_ID`, `PROCESSOR_URL` (адрес VPS с HTTPS) и тот же `PROCESSOR_API_KEY`, что указан на сервере. `WHISPER_MODEL` выбирает модель, которую будет использовать удалённый обработчик.

Для фоновой работы бота можно использовать `tmux`:

```bash
tmux new -s speechparser
source venv/bin/activate
python bot.py
# Ctrl+B, D — отсоединиться
# tmux attach -t speechparser — вернуться
```

## Настройки

| Переменная | Где задаётся | Назначение |
|---|---|---|
| `BOT_TOKEN` | локально | Токен Telegram-бота от [@BotFather](https://t.me/BotFather) |
| `ALLOWED_USER_ID` | локально | Telegram user ID, которому разрешён доступ |
| `PROCESSOR_URL` | локально | HTTPS-адрес API на VPS |
| `PROCESSOR_API_KEY` | локально и VPS | Общий секрет для авторизации запросов |
| `WHISPER_MODEL` | локально | Модель Whisper, по умолчанию `medium` |
| `WHISPER_WORKERS` | VPS | Параллельные части транскрипции (`auto` или верхний предел числом) |
| `MTS_SESSION_ID` | только VPS | Cookie `access` для приватных записей МТС Линк |
| `TRANSCRIPTION_TIMEOUT_SECONDS` | локально и VPS | Таймаут задания, по умолчанию 10800 секунд |
| `MAX_UPLOAD_BYTES` | VPS | Лимит загружаемого файла, по умолчанию 200 MiB |
| `SERVER_MAX_CONCURRENT_JOBS` | VPS | Число одновременных заданий, по умолчанию `1` |
| `JOB_RETENTION_SECONDS` | VPS | Срок хранения исходника и результата, по умолчанию 86400 секунд |

## Использование

1. Запусти обработчик на VPS и локальный `python bot.py`.
2. Отправь боту ссылку `https://my.mts-link.ru/...` или аудио/видео файл.
3. Бот покажет статусы и отправит готовую текстовую расшифровку без таймкодов.
4. Команда `/stop` останавливает текущую обработку и отправляет уже готовую часть текста.

Telegram Bot API ограничивает скачивание файлов размером 20 MB. Более крупные записи отправляй ссылкой МТС Линк.

## Разработка и тесты

Для локального запуска тестов установи зависимости бота и выполни:

```bash
python -m unittest discover -s tests -v
```
