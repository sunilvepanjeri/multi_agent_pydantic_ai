from fastapi import APIRouter
from . import chat


router = APIRouter()

router.include_router(chat.router, tags=["chat"], prefix="/multi-agent")