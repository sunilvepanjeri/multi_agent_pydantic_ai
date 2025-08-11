from pydantic_settings import BaseSettings
import os
from dotenv import find_dotenv



class Settings(BaseSettings):

    OPENAI_API_KEY: str
    MODEL: str = 'openai:gpt-4.1'

    model_config = {'extra': 'allow'}

settings = Settings(_env_file=find_dotenv('.env'))



os.environ.update(settings.model_dump())


