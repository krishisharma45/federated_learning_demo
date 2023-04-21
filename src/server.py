import flwr as fl


fl.server.start_server(server_address=f"0.0.0.0:8080",
                       config={"num_rounds": 3},
                       strategy=fl.server.strategy.FedAvg())
