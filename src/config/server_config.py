from pydantic import BaseModel


class Config(BaseModel):
    num_rounds: int = 5
    round_timeout: float = None


class ServerConfig(BaseModel):
    server_address: str = '0.0.0.0'
    min_available_clients: int = 3
