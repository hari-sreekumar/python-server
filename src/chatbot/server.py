import os
import random
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

GREETINGS = [
    "🤖 Hsk live!! Type 'quit' or 'stop' to exit \n",
    "🤖 Hsk has entered the chat! Ready to cause some chaos 😎 ('quit'/'stop' to leave) \n",
    "🤖 Beep boop, Hsk online! Let's get weird. ('quit'/'stop' to exit) \n",
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

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a helpful chatbot Hsk, the sassy bot!! you help users with general queries. "
    "Keep it short, use humor, throw in puns occasionally",
}

# session_id -> message history
sessions: dict[str, list[dict]] = {}


def new_history() -> list[dict]:
    return [
        SYSTEM_PROMPT,
        {"role": "user", "content": FEWSHOT_USER},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT},
    ]


app = FastAPI(title="Hsk Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str


@app.post("/session")
def create_session() -> dict:
    session_id = str(uuid.uuid4())
    sessions[session_id] = new_history()
    return {"session_id": session_id, "greeting": random.choice(GREETINGS).strip()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid.uuid4())
    messages = sessions.setdefault(session_id, new_history())

    lowered = req.message.lower()
    reaction = ""
    for keyword, emoji in REACTIONS.items():
        if keyword in lowered:
            reaction = emoji + " "
            break

    messages.append({"role": "user", "content": req.message})

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=1.1,
        top_p=0.9,
        presence_penalty=0.6,
    )

    reply = completion.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": reply})

    return ChatResponse(session_id=session_id, reply=reaction + reply)


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict:
    sessions.pop(session_id, None)
    return {"ok": True}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
