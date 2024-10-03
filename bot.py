import logging
import pathlib
import json
from datetime import datetime

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
    PicklePersistence,
)

from src.deposit_helper import DepositHelper
from src.settings import GigaSettings, TG_TOKEN


class JsonLogFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": datetime.utcnow().isoformat(),
            "username": getattr(record, "username", None),
            "user_id": getattr(record, "user_id", None),
            "question": getattr(record, "question", None),
            "answer": getattr(record, "answer", None),
        }
        return json.dumps(log_record, ensure_ascii=False)


json_handler = logging.FileHandler(
    str(pathlib.Path(__file__).parent / "deposit_bot.json"),
    mode="w",
    encoding="UTF-8",
)
json_handler.setFormatter(JsonLogFormatter())

logging.basicConfig(
    level=logging.WARN,
    force=True,
    handlers=[json_handler, logging.StreamHandler()],
)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

settings = GigaSettings()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    giga = DepositHelper(settings.chat_model, settings.embeddings)
    context.chat_data["giga"] = giga
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Привет!\n"
        "Я помощник Сбера по вкладам!\n"
        "Задай мне любой вопрос по вкладам в Сбере. Я постараюсь на него ответить.\n"
        "Для сброса контекста используй команды /start или /clear\n",
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    giga = DepositHelper(settings.chat_model, settings.embeddings)
    context.chat_data["giga"] = giga
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Контекст сброшен.")


async def answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    question = update.message.text

    giga: DepositHelper = context.chat_data.get("giga") or DepositHelper(
        settings.chat_model, settings.embeddings
    )

    attempt = 0
    result = None
    while not result and attempt < 3:
        try:
            result = await giga.aget_answer(question)
        except Exception as e:
            _logger.error(
                "Ошибка при попытке %d для пользователя %s: %s",
                attempt + 1,
                user.username,
                e,
            )
            attempt += 1

    context.chat_data["giga"] = giga
    result = result or "Возникла техническая ошибка. Повторите запрос, пожалуйста."
    if attempt > 0:
        result = "Простите, что заставил ждать. " + result

    _logger.info(
        "",
        extra={
            "username": user.username,
            "user_id": user.id,
            "question": question,
            "answer": result,
        },
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=result)


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Извини, я не понимаю эту команду.\n"
        "Для сброса контекста используй команды /start или /clear",
    )


my_persistence = PicklePersistence(
    filepath=str(pathlib.Path(__file__).parent.parent.parent / "storage")
)

if __name__ == "__main__":
    application = (
        ApplicationBuilder().token(TG_TOKEN).persistence(persistence=my_persistence).build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), answer))
    application.add_handler(MessageHandler(filters.COMMAND, unknown))

    application.run_polling()
