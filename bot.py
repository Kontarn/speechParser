#!/usr/bin/env python3
"""
Telegram бот для транскрибации.

Поддерживает:
- Ссылки МТС Линк (https://my.mts-link.ru/...)
- Прикреплённые файлы (видео, аудио, документы mp4/mp3/wav/m4a/ogg/webm)
"""

import asyncio
import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv
from telegram import BotCommand, MenuButtonCommands, Update
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from common import (
    BASE_DIR,
    MTS_LINK_PREFIX,
    SUPPORTED_EXTENSIONS,
    TranscriptionCancelled,
    parse_mts_url,
)
from processor_client import ProcessorClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



def _env_int(name: str, default: int = 0) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        raise RuntimeError(f"Переменная окружения {name} должна быть целым числом, получено: {value!r}")


BOT_TOKEN = os.getenv("BOT_TOKEN")
ALLOWED_USER_ID = _env_int("ALLOWED_USER_ID", 0)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
PROCESSOR_URL = os.getenv("PROCESSOR_URL", "").rstrip("/")
PROCESSOR_API_KEY = os.getenv("PROCESSOR_API_KEY", "")
TRANSCRIPTION_TIMEOUT_SECONDS = _env_int("TRANSCRIPTION_TIMEOUT_SECONDS", 3 * 60 * 60)
REMOTE_POLL_INTERVAL_SECONDS = 2

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB — лимит скачивания через облачный Telegram Bot API
MAX_CONCURRENT_JOBS = 1
MAX_PENDING_JOBS = max(1, _env_int("BOT_MAX_PENDING_JOBS", 32))
ACTIVE_JOBS = 0
JOB_QUEUE = []
JOB_STATUS = {}
JOB_CONDITION = threading.Condition()


def _is_worker_unavailable(error: Exception) -> bool:
    if isinstance(error, (httpx.NetworkError, httpx.TimeoutException,
                          httpx.RemoteProtocolError, httpx.ProxyError)):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        if error.response.status_code in {502, 503, 504}:
            return True
        # Jobs are kept in the worker's memory and disappear after a restart.
        return (error.response.status_code == 404
                and error.request.method in {"GET", "DELETE"}
                and "/jobs/" in error.request.url.path)
    return False


def _log_worker_unavailable(error: Exception, job_id: str) -> None:
    reason = (f"HTTP {error.response.status_code}" if isinstance(error, httpx.HTTPStatusError)
              else type(error).__name__)
    log.warning("Воркер недоступен; локальное задание %s завершено (%s)", job_id, reason)


def _processor_client() -> ProcessorClient:
    return ProcessorClient(PROCESSOR_URL, PROCESSOR_API_KEY,
                           TRANSCRIPTION_TIMEOUT_SECONDS, REMOTE_POLL_INTERVAL_SECONDS)


def _run_pipeline_with_timeout(kind: str, args: tuple, update_status, job_id: str | None = None) -> str:
    with JOB_CONDITION:
        job = JOB_STATUS.get(job_id, {})
        output_path = Path(job.get("output_path") or get_transcript_path(kind, args))

    def cancelled():
        with JOB_CONDITION:
            return bool(JOB_STATUS.get(job_id, {}).get("cancel_requested"))

    def submitted(remote_id):
        with JOB_CONDITION:
            if job_id in JOB_STATUS:
                JOB_STATUS[job_id]["remote_job_id"] = remote_id

    return _processor_client().run(kind, args, output_path, update_status, cancelled, submitted)


def _cleanup_remote_job(job_id: str) -> None:
    with JOB_CONDITION:
        job = JOB_STATUS.get(job_id)
        remote_id = job.get("remote_job_id") if job else None
    if remote_id and _processor_client().cleanup(remote_id):
        with JOB_CONDITION:
            if job_id in JOB_STATUS:
                JOB_STATUS[job_id]["remote_job_id"] = None


def _send_processing_failure(bot, chat_id: int, status_message_id: int, text: str, main_loop) -> None:
    async def _send() -> None:
        await _edit_status(bot, chat_id, status_message_id, "❌ Обработка остановлена.")
        await bot.send_message(chat_id=chat_id, text=f"❌ Не удалось обработать запись: {text}")

    future = asyncio.run_coroutine_threadsafe(_send(), main_loop)
    future.result(timeout=30)


def _refresh_queue_positions() -> None:
    for position, job_id in enumerate(JOB_QUEUE, 1):
        JOB_STATUS[job_id]["position"] = position


def register_job(job_type: str, user_id: int, chat_id: int, message_id: int) -> str:
    with JOB_CONDITION:
        now = time.time()
        for old_id, job in list(JOB_STATUS.items()):
            if job["status"] not in {"queued", "running"} and now - job.get("finished_at", now) > 86400:
                if job.get("output_path"):
                    try:
                        Path(job["output_path"]).unlink(missing_ok=True)
                    except OSError:
                        log.warning("Не удалось удалить старый результат %s", old_id)
                JOB_STATUS.pop(old_id)
        if sum(job["status"] in {"queued", "running", "delivering"} for job in JOB_STATUS.values()) >= MAX_PENDING_JOBS:
            raise OverflowError("Очередь заполнена. Повтори запрос позже.")
        job_id = uuid.uuid4().hex
        JOB_STATUS[job_id] = {
            "job_type": job_type, "user_id": user_id, "chat_id": chat_id,
            "message_id": message_id, "status": "queued", "position": 0,
            "created_at": now, "cancel_requested": False, "slot_acquired": False,
        }
        JOB_QUEUE.append(job_id)
        _refresh_queue_positions()
        return job_id


def get_queue_position(job_id: str) -> int:
    with JOB_CONDITION:
        return JOB_STATUS.get(job_id, {}).get("position", 0)


def acquire_job_slot(job_id: str | None = None) -> bool:
    global ACTIVE_JOBS
    with JOB_CONDITION:
        job = JOB_STATUS.get(job_id)
        if job and job.get("slot_acquired"):
            return True
        if job and (job.get("cancel_requested") or job["status"] == "cancelled"):
            return False
        if ACTIVE_JOBS >= MAX_CONCURRENT_JOBS or (JOB_QUEUE and JOB_QUEUE[0] != job_id):
            return False
        if job_id in JOB_QUEUE:
            JOB_QUEUE.remove(job_id)
        ACTIVE_JOBS += 1
        if job:
            job.update(status="running", position=0, slot_acquired=True)
        _refresh_queue_positions()
        return True


def wait_for_job_slot(job_id: str) -> bool:
    with JOB_CONDITION:
        while True:
            job = JOB_STATUS.get(job_id)
            if not job or job.get("cancel_requested") or job["status"] == "cancelled":
                return False
            if acquire_job_slot(job_id):
                return True
            JOB_CONDITION.wait(timeout=0.5)


def release_job_slot(job_id: str | None = None, status: str = "done") -> None:
    global ACTIVE_JOBS
    with JOB_CONDITION:
        job = JOB_STATUS.get(job_id)
        if job_id is None or (job and job.get("slot_acquired")):
            ACTIVE_JOBS = max(0, ACTIVE_JOBS - 1)
        if job:
            job.update(status=status, position=0, slot_acquired=False, finished_at=time.time())
        if job_id in JOB_QUEUE:
            JOB_QUEUE.remove(job_id)
        _refresh_queue_positions()
        JOB_CONDITION.notify_all()


def is_mts_link(text: str) -> bool:
    return text.strip().startswith(MTS_LINK_PREFIX)


def get_transcript_path(kind: str, args: tuple, job_id: str | None = None) -> Path:
    if kind == "file":
        stem = Path(args[0]).stem
    else:
        event_session_id, _ = parse_mts_url(args[0])
        stem = event_session_id
    suffix = f"_{job_id}" if job_id else ""
    return BASE_DIR / "results" / f"transcript_{stem}{suffix}.txt"


async def _edit_status(telegram_bot, chat_id: int, message_id: int, text: str) -> None:
    try:
        await telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text)
    except TelegramError as error:
        log.debug("Не удалось обновить статус Telegram (%s)", type(error).__name__)


def _deliver_transcript(telegram_bot, chat_id, message_id, transcript_path, main_loop, *, partial):
    path = Path(transcript_path)
    content = path.read_bytes() if path.is_file() else b""
    if not partial and not path.is_file():
        raise RuntimeError("Файл расшифровки не найден")

    async def send():
        if not partial and not content:
            await _edit_status(telegram_bot, chat_id, message_id, "В записи не обнаружено речи.")
            return
        if content:
            await _edit_status(telegram_bot, chat_id, message_id,
                               "📤 Отправляю расшифрованную часть..." if partial else "📤 Отправляю файл...")
            await telegram_bot.send_document(
                chat_id=chat_id, document=content, filename=path.name,
                caption="⏹ Расшифровка остановлена. Файл содержит уже готовую часть." if partial else "✅ Готово!",
            )
        text = ("⏹ Остановлено. Частичная расшифровка отправлена выше." if content
                else "⏹ Расшифровка остановлена, готового текста пока нет.") if partial else "✅ Готово! Файл отправлен выше."
        await _edit_status(telegram_bot, chat_id, message_id, text)

    asyncio.run_coroutine_threadsafe(send(), main_loop).result(timeout=60)


def _send_partial_result_sync(telegram_bot, chat_id: int, message_id: int, transcript_path: Path, main_loop) -> None:
    _deliver_transcript(telegram_bot, chat_id, message_id, transcript_path, main_loop, partial=True)


def check_access(update: Update) -> bool:
    if update.effective_user is None:
        return False
    if update.effective_user.id != ALLOWED_USER_ID:
        log.warning(f"Отклонён запрос от user_id={update.effective_user.id}")
        return False
    return True


def make_status_updater(bot, chat_id: int, message_id: int, main_loop: asyncio.AbstractEventLoop):
    """
    Возвращает синхронную функцию update_status(text).
    Безопасно вызывается из любого потока — шлёт задачу в основной event loop.
    Троттлит вызовы: не чаще раза в 5 секунд чтобы не получить RetryAfter от Telegram.
    """
    closed = threading.Event()
    pending = set()
    lock = threading.RLock()
    last_sent = [0.0]  # mutable для захвата в closure
    min_interval = 5.0

    def update_status(text: str) -> None:
        if closed.is_set() or main_loop.is_closed():
            return
        now = time.monotonic()
        if now - last_sent[0] < min_interval:
            return
        last_sent[0] = now

        async def _edit():
            if closed.is_set():
                return
            try:
                await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text)
            except TelegramError as error:
                log.warning("Не удалось обновить статус Telegram (%s)", type(error).__name__)
        with lock:
            future = asyncio.run_coroutine_threadsafe(_edit(), main_loop)
            pending.add(future)
            future.add_done_callback(discard)

    def discard(future):
        with lock:
            pending.discard(future)

    def close():
        closed.set()
        with lock:
            for future in list(pending):
                future.cancel()

    update_status.close = close

    return update_status


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not check_access(update):
        return
    await update.message.reply_text(
        "Привет! Умею транскрибировать:\n\n"
        "• Ссылки МТС Линк — просто отправь ссылку\n"
        "• Файлы — прикрепи видео или аудио файл\n\n"
        f"Поддерживаемые форматы: {', '.join(sorted(SUPPORTED_EXTENSIONS))}\n"
        "Лимит файла через Telegram: 20 MB"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not check_access(update):
        return
    await update.message.reply_text(
        f"Текущая модель: {WHISPER_MODEL}\n"
        "Время обработки зависит от длительности записи, модели и сервера.\n"
        "Минуты в статусе показывают обработанную длительность записи.\n"
        "Результат придёт .txt файлом без таймкодов.\n\n"
        "Чтобы остановить текущую расшифровку и получить готовую часть, отправь /stop.\n\n"
        "Проверить доступность обработчика: /ping.\n\n"
        "Что принимаю:\n"
        "• Ссылку https://my.mts-link.ru/...\n"
        "• Прикреплённый аудио или видео файл (до 20 MB)"
    )


async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not check_access(update):
        return
    message = update.effective_message
    if message is None:
        return
    if not PROCESSOR_URL:
        await message.reply_text("⚠️ PROCESSOR_URL не задан в .env.")
        return

    started_at = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
            response = await client.get(f"{PROCESSOR_URL}/health")
            response.raise_for_status()
            if response.json().get("status") != "ok":
                raise ValueError("неожиданный ответ health endpoint")
    except (httpx.HTTPError, ValueError, AttributeError) as error:
        log.warning("Воркер недоступен при /ping (%s)", type(error).__name__)
        return

    elapsed_ms = round((time.monotonic() - started_at) * 1000)
    await message.reply_text(f"✅ Обработчик доступен ({elapsed_ms} мс).")


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not check_access(update):
        return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    stopped_queued = stopped_running = False
    with JOB_CONDITION:
        jobs = [(job_id, job) for job_id, job in JOB_STATUS.items()
                if job["user_id"] == user_id and job["chat_id"] == chat_id
                and job["status"] in {"queued", "running"}]
        for job_id, job in jobs:
            job["cancel_requested"] = True
            if job["status"] == "queued":
                if job_id in JOB_QUEUE:
                    JOB_QUEUE.remove(job_id)
                job["status"] = "cancelled"
                stopped_queued = True
            else:
                stopped_running = True
        _refresh_queue_positions()
        JOB_CONDITION.notify_all()
    if not jobs:
        await update.message.reply_text("Сейчас нет активной расшифровки.")
        return

    if stopped_running:
        await update.message.reply_text("⏹ Останавливаю расшифровку и подготовлю уже готовую часть...")
    elif stopped_queued:
        await update.message.reply_text("⏹ Задание в очереди отменено.")


def _start_job(job_id, kind, args, telegram_bot, chat_id, message_id, main_loop):
    output_path = get_transcript_path(kind, args, job_id)
    with JOB_CONDITION:
        JOB_STATUS[job_id]["output_path"] = str(output_path)
    update_status = make_status_updater(telegram_bot, chat_id, message_id, main_loop)

    def worker():
        acquired = False
        outcome = "failed"
        unavailable = False
        delivered = False
        try:
            acquired = wait_for_job_slot(job_id)
            if not acquired:
                outcome = "cancelled"
                return
            try:
                transcript = _run_pipeline_with_timeout(kind, args, update_status, job_id)
            finally:
                # Telegram delivery must not keep the next transcription waiting.
                update_status.close()
                release_job_slot(job_id, "delivering")
            _send_result_sync(telegram_bot, chat_id, message_id, transcript, main_loop)
            delivered = True
            outcome = "completed"
        except TranscriptionCancelled:
            outcome = "cancelled"
            if acquired:
                _send_partial_result_sync(telegram_bot, chat_id, message_id, output_path, main_loop)
                delivered = True
        except Exception as error:
            unavailable = _is_worker_unavailable(error)
            if unavailable:
                outcome = "unavailable"
                _log_worker_unavailable(error, job_id)
            else:
                log.exception("Ошибка обработки задания %s", job_id)
                _send_processing_failure(telegram_bot, chat_id, message_id, str(error), main_loop)
        finally:
            update_status.close()
            try:
                if not unavailable:
                    _cleanup_remote_job(job_id)
            finally:
                try:
                    if kind == "file":
                        Path(args[0]).unlink(missing_ok=True)
                    if delivered:
                        output_path.unlink(missing_ok=True)
                except OSError:
                    log.warning("Не удалось удалить временные файлы задания %s", job_id)
                finally:
                    release_job_slot(job_id, outcome)

    try:
        thread = threading.Thread(target=worker, daemon=True)
        with JOB_CONDITION:
            JOB_STATUS[job_id]["thread"] = thread
        thread.start()
    except Exception:
        if kind == "file":
            Path(args[0]).unlink(missing_ok=True)
        release_job_slot(job_id, "failed")
        raise


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not check_access(update):
        return

    text = update.message.text.strip()
    if not is_mts_link(text):
        await update.message.reply_text(
            "Не понял. Отправь ссылку МТС Линк или прикрепи аудио/видео файл.\n"
            "Пример: https://my.mts-link.ru/12345/67890/record-new/111222"
        )
        return

    try:
        parse_mts_url(text)
        job_id = register_job("mts_link", update.effective_user.id, update.effective_chat.id, update.message.message_id)
    except (ValueError, OverflowError) as error:
        await update.message.reply_text(str(error))
        return
    try:
        status_msg = await update.message.reply_text(
            f"⏳ Принял. Задание в очереди: {get_queue_position(job_id)}."
        )
        _start_job(job_id, "url", (text, WHISPER_MODEL), context.bot,
                   update.effective_chat.id, status_msg.message_id, asyncio.get_running_loop())
    except Exception:
        release_job_slot(job_id, "failed")
        raise


async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not check_access(update):
        return

    message = update.message
    file_obj = None
    filename = "file"

    if message.audio:
        file_obj = message.audio
        filename = message.audio.file_name or "audio.mp3"
    elif message.voice:
        file_obj = message.voice
        filename = "voice.ogg"
    elif message.video:
        file_obj = message.video
        filename = message.video.file_name or "video.mp4"
    elif message.video_note:
        file_obj = message.video_note
        filename = "video_note.mp4"
    elif message.document:
        ext = Path(message.document.file_name or "").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            await message.reply_text(
                f"Формат не поддерживается: {ext or 'неизвестный'}\n"
                f"Поддерживаю: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
            return
        file_obj = message.document
        filename = message.document.file_name or f"document{ext}"

    if not file_obj:
        await message.reply_text("Не удалось определить тип файла.")
        return

    if file_obj.file_size and file_obj.file_size > MAX_FILE_SIZE:
        size_mb = file_obj.file_size / 1024 / 1024
        await message.reply_text(
            f"Файл слишком большой: {size_mb:.1f} MB.\n"
            "Telegram Bot API ограничивает скачивание файлов до 20 MB.\n"
            "Для больших файлов используй ссылку МТС Линк."
        )
        return

    try:
        job_id = register_job("file", message.from_user.id, update.effective_chat.id, message.message_id)
    except OverflowError as error:
        await message.reply_text(str(error))
        return
    try:
        status_msg = await message.reply_text(
            f"⏳ Принял. Задание в очереди: {get_queue_position(job_id)}. Скачиваю файл..."
        )
    except Exception:
        release_job_slot(job_id, "failed")
        raise

    # Скачиваем файл в основном loop, потом передаём путь в поток
    suffix = Path(filename).suffix or ".mp4"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
    except OSError:
        release_job_slot(job_id, "failed")
        raise

    try:
        tg_file = await context.bot.get_file(file_obj.file_id)
        await tg_file.download_to_drive(tmp_path)
    except BadRequest as error:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        release_job_slot(job_id, "failed")
        if "File is too big" in str(error):
            await status_msg.edit_text(
                "Файл слишком большой для скачивания через Telegram (лимит 20 MB).\n"
                "Для больших файлов используй ссылку МТС Линк."
            )
            return
        log.exception(f"Не удалось скачать файл {filename}")
        await status_msg.edit_text("❌ Не удалось скачать файл из Telegram.")
        return
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        release_job_slot(job_id, "failed")
        log.exception(f"Не удалось скачать файл {filename}")
        await status_msg.edit_text("❌ Не удалось скачать файл из Telegram.")
        return
    log.info(f"Файл скачан: {tmp_path} ({filename})")


    _start_job(job_id, "file", (tmp_path, WHISPER_MODEL), context.bot,
               update.effective_chat.id, status_msg.message_id, asyncio.get_running_loop())


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("Необработанная ошибка при обработке обновления", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text("❌ Произошла внутренняя ошибка. Попробуйте ещё раз.")
        except Exception:
            log.exception("Не удалось отправить сообщение об ошибке")


def _send_result_sync(telegram_bot, chat_id: int, message_id: int, transcript_path: str, main_loop) -> None:
    _deliver_transcript(telegram_bot, chat_id, message_id, transcript_path, main_loop, partial=False)


BOT_COMMANDS = (
    ("start", "Начать работу с ботом", cmd_start),
    ("help", "Справка и поддерживаемые форматы", cmd_help),
    ("ping", "Проверить доступность обработчика", cmd_ping),
    ("stop", "Остановить расшифровку и получить готовую часть", cmd_stop),
)


async def configure_commands(application: Application) -> None:
    """Publish command suggestions and enable Telegram's command menu."""
    await application.bot.set_my_commands([
        BotCommand(command, description) for command, description, _ in BOT_COMMANDS
    ])
    await application.bot.set_chat_menu_button(menu_button=MenuButtonCommands())
    log.info("Меню и подсказки команд Telegram зарегистрированы")


def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN не задан в .env")
    if not ALLOWED_USER_ID:
        raise RuntimeError("ALLOWED_USER_ID не задан в .env")
    if not PROCESSOR_URL or not PROCESSOR_API_KEY:
        raise RuntimeError("PROCESSOR_URL и PROCESSOR_API_KEY должны быть заданы в .env")

    logging.getLogger("httpx").setLevel(logging.WARNING)
    app = Application.builder().token(BOT_TOKEN).post_init(configure_commands).build()

    for command, _, callback in BOT_COMMANDS:
        app.add_handler(CommandHandler(command, callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(
        filters.AUDIO | filters.VOICE | filters.VIDEO | filters.VIDEO_NOTE | filters.Document.ALL,
        handle_file,
    ))
    app.add_error_handler(error_handler)

    log.info(f"Бот запущен. Разрешён user_id={ALLOWED_USER_ID}, модель={WHISPER_MODEL}")
    try:
        app.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        log.info("Получен сигнал завершения, останавливаю бота...")
    finally:
        log.info("Бот остановлен.")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")
    main()
