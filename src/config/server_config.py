from pydantic import BaseModel


class Config(BaseModel):
    num_rounds: int = 3
    round_timeout: float = 30


class ServerConfig(BaseModel):
    server_address: str = '0.0.0.0'
    min_available_clients: int = 3
    min_fit_clients: int = 3
    min_eval_clients: int = 2
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0

