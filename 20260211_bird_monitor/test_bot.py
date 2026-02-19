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

    # print("Checking for messages...")
    # Botに送られた最新のメッセージ（updates）を取得する
    # updates = await bot.get_updates()
    # 
    # if not updates:
    #     print("Error: Botにメッセージが届いていないみたい。")
    #     print("TelegramアプリであなたのBotを探して、'START'を押すか何か送ってみて！")
    #     return
    # 
    # 最新のメッセージからChat ID（シンジくんの住所）を特定する
    # chat_id = updates[-1].message.chat.id
    # print(f"Found your Chat ID: {chat_id}")
    # 
    # 実際にメッセージを送ってみる
    # print("Sending message...")
    # await bot.send_message(chat_id=chat_id, text="シンジくん、聞こえる？テスト成功よ。")
    # print("Done! スマホを確認してみて。")

if __name__ == "__main__":
    asyncio.run(main())