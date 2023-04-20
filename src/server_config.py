from pydantic import BaseModel


class ServerConfig(BaseModel):
    num_rounds: int = 3
    round_timeout: float = None