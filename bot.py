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
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from pipeline import run_pipeline, run_pipeline_file

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
MTS_SESSION_ID = os.getenv("MTS_SESSION_ID", "")

MTS_LINK_PREFIX = "https://my.mts-link.ru/"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB — лимит Telegram Bot API
SUPPORTED_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mkv", ".avi", ".mov"}


def is_mts_link(text: str) -> bool:
    return text.strip().startswith(MTS_LINK_PREFIX)


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
        "Лимит файла через Telegram: 50 MB"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not check_access(update):
        return
    await update.message.reply_text(
        f"Текущая модель: {WHISPER_MODEL}\n"
        "Время обработки: ~40–90 мин для 2-часовой записи.\n"
        "Результат придёт .txt файлом с таймстампами.\n\n"
        "Что принимаю:\n"
        "• Ссылку https://my.mts-link.ru/...\n"
        "• Прикреплённый аудио или видео файл (до 50 MB)"
    )


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

    status_msg = await update.message.reply_text("⏳ Принял. Начинаю обработку...")
    main_loop = asyncio.get_running_loop()
    update_status = make_status_updater(context.bot, update.effective_chat.id, status_msg.message_id, main_loop)

    def _worker():
        try:
            transcript_path = run_pipeline(
                url=text,
                model_name=WHISPER_MODEL,
                session_id=MTS_SESSION_ID or None,
                update_status=update_status,
            )
            _send_result_sync(context.bot, update.effective_chat.id, status_msg.message_id, transcript_path, main_loop)
        except Exception as e:
            log.exception(f"Ошибка пайплайна для {text}")
            update_status(f"❌ Ошибка: {e}")

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
            "Telegram Bot API ограничивает скачивание файлов до 50 MB.\n"
            "Для больших файлов используй ссылку МТС Линк."
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

    tg_file = await context.bot.get_file(file_obj.file_id)
    await tg_file.download_to_drive(tmp_path)
    log.info(f"Файл скачан: {tmp_path} ({filename})")

    def _worker():
        try:
            update_status(f"🎙 Транскрибирую {filename}... (модель {WHISPER_MODEL})")
            transcript_path = run_pipeline_file(
                input_path=tmp_path,
                model_name=WHISPER_MODEL,
                update_status=update_status,
            )
            _send_result_sync(context.bot, update.effective_chat.id, status_msg.message_id, transcript_path, main_loop)
        except Exception as e:
            log.exception(f"Ошибка при обработке файла {filename}")
            update_status(f"❌ Ошибка: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    threading.Thread(target=_worker, daemon=True).start()


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
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(
        filters.AUDIO | filters.VOICE | filters.VIDEO | filters.VIDEO_NOTE | filters.Document.ALL,
        handle_file,
    ))

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
