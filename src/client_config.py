from pydantic import BaseModel


class ClientConfig(BaseModel):
    server_address: str = "0.0.0.0"
    device_id: str


