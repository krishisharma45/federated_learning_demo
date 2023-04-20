from pydantic import BaseModel


class ClientConfig(BaseModel):
    server_address: str = "0.0.0.0"
    device_id: str
    epochs: int = 5
    batch_size: int = 32
    steps_per_epoch: int = 3

