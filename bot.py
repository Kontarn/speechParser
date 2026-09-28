#!/usr/bin/env python3
"""
Telegram бот для транскрибации.

Поддерживает:
- Ссылки МТС Линк (https://my.mts-link.ru/...)
- Прикреплённые файлы (видео, аудио, документы mp4/mp3/wav/m4a/ogg/webm)
"""

import asyncio
import logging
import multiprocessing
import os
import queue
import tempfile
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from pipeline import BASE_DIR, parse_mts_url, run_pipeline, run_pipeline_file

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
MTS_SESSION_ID = os.getenv("MTS_SESSION_ID", "")
TRANSCRIPTION_TIMEOUT_SECONDS = _env_int("TRANSCRIPTION_TIMEOUT_SECONDS", 3 * 60 * 60)

MTS_LINK_PREFIX = "https://my.mts-link.ru/"
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB — лимит скачивания через облачный Telegram Bot API
SUPPORTED_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mkv", ".avi", ".mov"}
MAX_CONCURRENT_JOBS = 1
ACTIVE_JOBS = 0
JOB_QUEUE = []
JOB_STATUS = {}


class TranscriptionCancelled(Exception):
    """Пайплайн остановлен пользователем, частичный результат можно отправить."""


def _pipeline_process_target(kind, args, status_queue, result_queue) -> None:
    """Запускает пайплайн в отдельном процессе, который можно принудительно остановить."""
    def process_status(text: str) -> None:
        status_queue.put(text)

    try:
        if kind == "file":
            result = run_pipeline_file(
                input_path=args[0],
                model_name=args[1],
                update_status=process_status,
            )
        else:
            result = run_pipeline(
                url=args[0],
                model_name=args[1],
                session_id=args[2],
                update_status=process_status,
            )
        result_queue.put(("result", result))
    except Exception as error:
        result_queue.put(("error", f"{type(error).__name__}: {error}"))


def _run_pipeline_with_timeout(kind: str, args: tuple, update_status, job_id: str | None = None) -> str:
    context = multiprocessing.get_context("spawn")
    status_queue = context.Queue()
    result_queue = context.Queue()
    process = context.Process(
        target=_pipeline_process_target,
        args=(kind, args, status_queue, result_queue),
        daemon=True,
    )
    process.start()
    if job_id and job_id in JOB_STATUS:
        JOB_STATUS[job_id]["process"] = process
    deadline = time.monotonic() + TRANSCRIPTION_TIMEOUT_SECONDS

    while process.is_alive():
        if job_id and JOB_STATUS.get(job_id, {}).get("cancel_requested"):
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
            raise TranscriptionCancelled
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
            raise TimeoutError(
                f"Превышено время обработки ({TRANSCRIPTION_TIMEOUT_SECONDS // 3600} ч.)"
            )
        try:
            update_status(status_queue.get(timeout=min(1, remaining)))
        except queue.Empty:
            pass

    process.join()
    if job_id and JOB_STATUS.get(job_id, {}).get("cancel_requested"):
        raise TranscriptionCancelled
    while True:
        try:
            update_status(status_queue.get_nowait())
        except queue.Empty:
            break

    try:
        result_type, result = result_queue.get(timeout=1)
    except queue.Empty as error:
        raise RuntimeError(f"Процесс обработки завершился без результата (код {process.exitcode})") from error
    if result_type == "error":
        raise RuntimeError(result)
    return result


def _send_processing_failure(bot, chat_id: int, status_message_id: int, text: str, main_loop) -> None:
    async def _send() -> None:
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text="❌ Обработка остановлена.")
        except Exception:
            pass
        await bot.send_message(chat_id=chat_id, text=f"❌ Не удалось обработать запись: {text}")

    asyncio.run_coroutine_threadsafe(_send(), main_loop)


def register_job(job_type: str, user_id: int, chat_id: int, message_id: int) -> str:
    import uuid

    job_id = uuid.uuid4().hex
    JOB_STATUS[job_id] = {
        "job_type": job_type,
        "user_id": user_id,
        "chat_id": chat_id,
        "message_id": message_id,
        "status": "queued",
        "position": 0,
        "created_at": __import__("time").time(),
        "process": None,
        "cancel_requested": False,
    }
    JOB_QUEUE.append(job_id)
    JOB_STATUS[job_id]["position"] = JOB_QUEUE.index(job_id) + 1
    return job_id


def get_queue_position(job_id: str) -> int:
    if job_id not in JOB_STATUS:
        return 0
    return JOB_STATUS[job_id].get("position", 0)


def acquire_job_slot(job_id: str | None = None) -> bool:
    global ACTIVE_JOBS
    if job_id is not None and job_id in JOB_QUEUE:
        JOB_QUEUE.remove(job_id)
    if ACTIVE_JOBS >= MAX_CONCURRENT_JOBS:
        if job_id and job_id in JOB_STATUS:
            JOB_STATUS[job_id]["status"] = "queued"
            JOB_STATUS[job_id]["position"] = JOB_QUEUE.index(job_id) + 1 if job_id in JOB_QUEUE else 1
        return False
    ACTIVE_JOBS += 1
    if job_id and job_id in JOB_STATUS:
        JOB_STATUS[job_id]["status"] = "running"
        JOB_STATUS[job_id]["position"] = 0
    return True


def release_job_slot(job_id: str | None = None) -> None:
    global ACTIVE_JOBS
    ACTIVE_JOBS = max(0, ACTIVE_JOBS - 1)
    if job_id and job_id in JOB_STATUS:
        JOB_STATUS[job_id]["status"] = "done"
        JOB_STATUS[job_id]["position"] = 0

    for queued_id in JOB_QUEUE:
        if queued_id in JOB_STATUS:
            JOB_STATUS[queued_id]["status"] = "queued"
            JOB_STATUS[queued_id]["position"] = JOB_QUEUE.index(queued_id) + 1


def is_mts_link(text: str) -> bool:
    return text.strip().startswith(MTS_LINK_PREFIX)


def get_transcript_path(kind: str, args: tuple) -> Path:
    if kind == "file":
        stem = Path(args[0]).stem
    else:
        event_session_id, _ = parse_mts_url(args[0])
        stem = event_session_id
    return BASE_DIR / "results" / f"transcript_{stem}.txt"


def _send_partial_result_sync(bot, chat_id: int, message_id: int, transcript_path: Path, main_loop) -> None:
    async def _send() -> None:
        if transcript_path.exists() and transcript_path.stat().st_size:
            try:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text="📤 Отправляю расшифрованную часть...",
                )
            except Exception:
                pass
            with transcript_path.open("rb") as transcript_file:
                await bot.send_document(
                    chat_id=chat_id,
                    document=transcript_file,
                    filename=transcript_path.name,
                    caption="⏹ Расшифровка остановлена. Файл содержит уже готовую часть.",
                )
            final_text = "⏹ Остановлено. Частичная расшифровка отправлена выше."
        else:
            final_text = "⏹ Расшифровка остановлена, готового текста пока нет."
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=final_text)
        except Exception:
            pass

    future = asyncio.run_coroutine_threadsafe(_send(), main_loop)
    future.result()


def check_access(update: Update) -> bool:
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
    import time
    last_sent = [0.0]  # mutable для захвата в closure
    min_interval = 5.0

    def update_status(text: str) -> None:
        now = time.monotonic()
        if now - last_sent[0] < min_interval:
            return
        last_sent[0] = now

        async def _edit():
            try:
                await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text)
            except Exception as e:
                log.warning(f"edit_message_text не удался: {e}")
        asyncio.run_coroutine_threadsafe(_edit(), main_loop)

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
        "Время обработки: ~40–90 мин для 2-часовой записи.\n"
        "Результат придёт .txt файлом с таймстампами.\n\n"
        "Чтобы остановить текущую расшифровку и получить готовую часть, отправь /stop.\n\n"
        "Что принимаю:\n"
        "• Ссылку https://my.mts-link.ru/...\n"
        "• Прикреплённый аудио или видео файл (до 20 MB)"
    )


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not check_access(update):
        return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    jobs = [
        (job_id, job)
        for job_id, job in JOB_STATUS.items()
        if job["user_id"] == user_id
        and job["chat_id"] == chat_id
        and job["status"] in {"queued", "running"}
    ]
    if not jobs:
        await update.message.reply_text("Сейчас нет активной расшифровки.")
        return

    stopped_queued = False
    stopped_running = False
    for job_id, job in jobs:
        if job["status"] == "queued":
            if job_id in JOB_QUEUE:
                JOB_QUEUE.remove(job_id)
            job["status"] = "cancelled"
            stopped_queued = True
            continue
        job["cancel_requested"] = True
        process = job.get("process")
        if process and process.is_alive():
            process.terminate()
        stopped_running = True

    if stopped_running:
        await update.message.reply_text("⏹ Останавливаю расшифровку и подготовлю уже готовую часть...")
    elif stopped_queued:
        await update.message.reply_text("⏹ Задание в очереди отменено.")


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

    job_id = register_job("mts_link", update.effective_user.id, update.effective_chat.id, update.message.message_id)
    if not acquire_job_slot(job_id):
        await update.message.reply_text(
            f"⏳ В данный момент уже идёт обработка другой записи. Ваша очередь: {get_queue_position(job_id)}."
        )
        return

    status_msg = await update.message.reply_text("⏳ Принял. Начинаю обработку...")
    main_loop = asyncio.get_running_loop()
    update_status = make_status_updater(context.bot, update.effective_chat.id, status_msg.message_id, main_loop)

    def _worker():
        try:
            transcript_path = _run_pipeline_with_timeout(
                "url",
                (text, WHISPER_MODEL, MTS_SESSION_ID or None),
                update_status,
                job_id,
            )
            _send_result_sync(context.bot, update.effective_chat.id, status_msg.message_id, transcript_path, main_loop)
        except TranscriptionCancelled:
            _send_partial_result_sync(
                context.bot,
                update.effective_chat.id,
                status_msg.message_id,
                get_transcript_path("url", (text, WHISPER_MODEL, MTS_SESSION_ID or None)),
                main_loop,
            )
        except Exception as e:
            log.exception(f"Ошибка пайплайна для {text}")
            _send_processing_failure(context.bot, update.effective_chat.id, status_msg.message_id, str(e), main_loop)
        finally:
            release_job_slot(job_id)

    threading.Thread(target=_worker, daemon=True).start()


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

    job_id = register_job("file", message.from_user.id, update.effective_chat.id, message.message_id)
    if not acquire_job_slot(job_id):
        await message.reply_text(
            f"⏳ В данный момент уже идёт обработка другой записи. Ваша очередь: {get_queue_position(job_id)}."
        )
        return

    status_msg = await message.reply_text("⏳ Принял. Скачиваю файл...")
    main_loop = asyncio.get_running_loop()
    update_status = make_status_updater(context.bot, update.effective_chat.id, status_msg.message_id, main_loop)

    # Скачиваем файл в основном loop, потом передаём путь в поток
    suffix = Path(filename).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        tg_file = await context.bot.get_file(file_obj.file_id)
        await tg_file.download_to_drive(tmp_path)
    except BadRequest as error:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        release_job_slot(job_id)
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
        release_job_slot(job_id)
        log.exception(f"Не удалось скачать файл {filename}")
        await status_msg.edit_text("❌ Не удалось скачать файл из Telegram.")
        return
    log.info(f"Файл скачан: {tmp_path} ({filename})")


    def _worker():
        try:
            transcript_path = _run_pipeline_with_timeout(
                "file",
                (tmp_path, WHISPER_MODEL),
                update_status,
                job_id,
            )
            _send_result_sync(context.bot, update.effective_chat.id, status_msg.message_id, transcript_path, main_loop)
        except TranscriptionCancelled:
            _send_partial_result_sync(
                context.bot,
                update.effective_chat.id,
                status_msg.message_id,
                get_transcript_path("file", (tmp_path, WHISPER_MODEL)),
                main_loop,
            )
        except Exception as e:
            log.exception(f"Ошибка при обработке файла {filename}")
            _send_processing_failure(context.bot, update.effective_chat.id, status_msg.message_id, str(e), main_loop)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            release_job_slot(job_id)

    threading.Thread(target=_worker, daemon=True).start()


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("Необработанная ошибка при обработке обновления", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text("❌ Произошла внутренняя ошибка. Попробуйте ещё раз.")
        except Exception:
            log.exception("Не удалось отправить сообщение об ошибке")


def _send_result_sync(bot, chat_id: int, message_id: int, transcript_path: str, main_loop: asyncio.AbstractEventLoop) -> None:
    """Отправляет файл с результатом из рабочего потока в основной loop."""
    async def _send():
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="📤 Отправляю файл...")
        except Exception:
            pass
        with open(transcript_path, "rb") as f:
            await bot.send_document(
                chat_id=chat_id,
                document=f,
                filename=Path(transcript_path).name,
                caption="✅ Готово!",
            )
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="✅ Готово! Файл отправлен выше.")
        except Exception:
            pass

    future = asyncio.run_coroutine_threadsafe(_send(), main_loop)
    future.result()  # ждём завершения перед выходом из потока


def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN не задан в .env")
    if not ALLOWED_USER_ID:
        raise RuntimeError("ALLOWED_USER_ID не задан в .env")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("stop", cmd_stop))
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
