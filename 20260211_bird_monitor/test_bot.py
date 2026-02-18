import asyncio
import os
from dotenv import load_dotenv
from telegram import Bot

load_dotenv()

async def main():
# os.getenv で安全に値を取り出す
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    if not TOKEN or not CHAT_ID:
        print("Error: .envファイルに設定が見つからない。")
        return

    bot = Bot(token=TOKEN)

    print("Sending secure message...")
    await bot.send_message(chat_id=CHAT_ID, text="安全に通信できている。")
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())