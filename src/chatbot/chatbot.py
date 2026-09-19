import random
from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv('GROQ_API_KEY')
client = Groq(api_key=api_key)

GREETINGS = [
    "🤖 Hsk live!! Type 'quit' or 'stop' to exit \n",
    "🤖 Hsk has entered the chat! Ready to cause some chaos 😎 ('quit'/'stop' to leave) \n",
    "🤖 Beep boop, Hsk online! Let's get weird. ('quit'/'stop' to exit) \n",
]

FAREWELLS = [
    "🤖 Oops1! Gotta go!!..byee",
    "🤖 Hsk out! 🚀 Catch you on the flip side!",
    "🤖 Peace out, human! 👋",
]

REACTIONS = {
    "love": "💖",
    "hate": "😤",
    "sad": "😢",
    "happy": "😄",
    "joke": "😂",
    "food": "🍕",
    "python": "🐍",
    "bug": "🐛",
    "code": "💻",
}

FEWSHOT_USER = "What's the capital of France?"
FEWSHOT_ASSISTANT = (
    "Paris! 🥖 Home of the Eiffel Tower and some seriously judgmental waiters. "
    "Fun fact: it's also called the 'City of Light' — so bring shades. 😎"
)

messages = [
    {
        'role': 'system',
        'content': 'You are a helpful chatbot Hsk, the sassy bot!! you help users with general queries. '
                   'Keep it short, use humor, throw in puns occasionally',
    },
    {'role': 'user', 'content': FEWSHOT_USER},
    {'role': 'assistant', 'content': FEWSHOT_ASSISTANT},
]

print(random.choice(GREETINGS))

while True:
    user_input = input('You:')
    if user_input.lower() in ['quit', 'stop']:
        print(random.choice(FAREWELLS))
        break

    reaction = ""
    lowered = user_input.lower()
    for keyword, emoji in REACTIONS.items():
        if keyword in lowered:
            reaction = emoji + " "
            break

    messages.append({'role': 'user', 'content': user_input})

    print(f"🤖 Hsk : {reaction}", end="", flush=True)

    stream = client.chat.completions.create(
        model='openai/gpt-oss-20b',
        messages=messages,
        stream=True,
        temperature=1.1,
        top_p=0.9,
        presence_penalty=0.6,
    )

    reply = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            reply += content

    messages.append({'role': 'assistant', 'content': reply})

    print()
