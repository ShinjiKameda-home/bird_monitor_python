# Bird Monitor // タイトルは最後まで保留

## Motivation: Closing the Loop with Avian Allies

As a researcher, I have a bad habit: I can’t help but question what they call "conventional wisdom", the status quo, and the way "everyone else does it".

Not only in Japan probably but also in the other countries, "Aquaponics" is also often praised as the ultimate sustainable farming system. But is there truly a "perfect" closed loop within these systems?

While, nitrogen certainly cycles from fish waste to plant roots via bacteria, what about the rest? Potassium, phosphorus, and trace minerals like calcium or magnesium? Simply pumping water will not allow these minerals to reach the roots of vegetables.

If I were to continue buying commercial fertilizer every year, it might be many times easier to enjoy the abundance of traditional soil cultivation, but I'm not looking for an easy solution. Instead of just buying my way out, I want to thrive by harnessing what nature already provides. To me, this is more than a project; it’s a way of honoring the natural world and the wisdom of our ancestors while carving out my own path.

I found one answer in the sea. As an islander, I love seaweed—nori, kombu, wakame. These are mineral goldmines. Even my freshwater Medaka fish enjoy them when finely shredded. Through them, the ocean's minerals eventually reach my plants.

But the real mystery remained: **Who supplies the phosphorus?**

This led me down a rabbit hole into the tragic history of Nauru (known as "Naoero" in the native language) and its phosphate mines. It’s a a solemn reminder of how resource development that is meant to bring prosperity to a nation can also lead to its downfall. In Japan, we often aren't taught these inconvenient truths in school, leading to a society where people lean on academic credentials as a survival strategy, often neglecting raw, hands-on awareness. But I'd like to say: "Ignorance is not a crime itself, but it often brings a disadvantage."

Then, it occurred to me: the precious "Gem" phosphorus, once obtained from the droppings of migratory birds like albatrosses, has now run out on Nauru, while it remains in excess within Japanese farmlands. The reason? Many farmers depend on imported commercial fertilizers year after year. Furthermore, the small birds that come to my garden every morning are collecting these nutrients from the surrounding mountains and fields.

Could these birds be the missing link in my ecosystem?

My home aquaponics setup has transformed. It’s no longer just a tank and some pipes; it’s a research station for a regional circular ecosystem, fueled by the dream of reconstructing the natural perfect closed loop that was lost.

To begin this journey, I decided to repurpose my home security camera into an RTSP server. My first goal: build a system to detect and notify me whenever my little "phosphorus messengers" arrive in my garden.

## Step 1: The Gateway - Repurposing a security camera into an RTSP server

Connecting my Hiseeu W-NVR (K8216-W6) was not a walk in the park; it felt more like a desperate battle.

Initially, I fell into a deep trap. I tried to assign static IP addresses to each camera to "organize" the network, but this was a fatal mistake. The dedicated vendor app stopped responding, and for a moment, I felt a wave of absolute despair—as if I had destroyed my only eyes on the world before even starting. I was paralyzed by the fear that I had lost everything at the very first step.

After a long struggle with cold sweat, I found a breakthrough. In my desperation, I turned to my AI collaborator, Gemini (whom I call "Gem-san"). I reached out to her as if grasping at a straw, and she showed me the truth I had overlooked: "Don't touch the individual camera settings. The NVR itself is the only gateway you need."

This advice was my lifeline. It taught me something profound: AI isn't here to replace us or steal our roles; it’s here to support us when we are pushed to our limits. By embracing this collaboration, I was able to turn a moment of absolute despair into a step toward growth.

![Figure 1a: Image screenshot of "System SetUp"](./images/606.jpg)
![Figure 1b: Image screenshot of turning on RTSP server function in "Network SetUp"](./images/608.jpg)

The reality was that the NVR acts as a central hub, and I only needed its single IP address to access all cameras. Even then, the official configuration screen was riddled with misleading information—missing colons and incorrect symbols. It was (not taught in school, ) only through trial and error that I finally reached the correct URL format:

rtsp://[User_ID]:[Password]@[NVR_IP]:80/ch[Camera_Number]_[Main0_or_Sub1].264

"User_ID" is "admin" in many cases.
"NVR_IP" is often "192.168.0.000" (The last "000" is a static number you defined.)
"Camera_Number" is started not with 1 but with 0 ("CAM1" was 0 here.)
"Main0_or_Sub1" sets resolution, 0 is high but slow, 1 is low but fast.
Note that the port is 80, not the standard 554. When the video stream finally played in VLC media player, it wasn't just a technical success; I felt like I'd regained my footing in a world I thought had been usurped by muscle-bound tech giants.

![Figure 2: The first stream image in the vlc madia player](./images/vlc_player_first_stream.bmp)

## Step 2: Audience with the "BotFather" - Initializing the Telegram API

Installing the Telegram on my phone.
Searching the "BotFather", **"@BotFather"** with blue-back check mark (official)
Calling **"/newbot"** , naming the bot and being given the token.
Searching the **Username of the bot**, and starting talking.
Waiting several minutes.

Installing the package python-telegram-bot into my HomeServer.
Making dotenv file, token and chatID will be hidden in this file.


## Step 3: The watchful eye - Building an AI bird watcher with YOLO & OpenCV

## Results: Tidings from the messengers - Real-time notifications in action

## Acknowledgments
 Special thanks to "Gem"-san, my insightful AI collaborator, for helping me structure these thoughts and translating my vision into English.