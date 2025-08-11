import uuid

from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import json
from app.agents_call import Responder


router = APIRouter()


class Message(BaseModel):
    query: str
    conversation_id: str
    chat_history: list[dict[str, str]]

@router.get('/uuid')
async def get_uuid():
    return {'uuid': str(uuid.uuid4())}


@router.post("/chat")
async def message_agent(message: Message):

    with open('app/instruction.json') as json_file:
        load_data = json.load(json_file)



    answer = await Responder(load_data)(message.query, message.chat_history)

    add_conversion = [{
        'role': 'user',
        "content": message.query
    },
        {
            'role': 'assistant',
            'content': answer.output
        }]

    final_history = message.chat_history + add_conversion

    return JSONResponse({"answer": answer.output, "final_history": final_history, 'steps': str(answer.all_messages())})







