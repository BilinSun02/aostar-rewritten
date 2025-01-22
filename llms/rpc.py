# Adapted from
# https://medium.com/@taraszhere/coding-remote-procedure-call-rpc-with-python-3b14a7d00ac8
import json
import socket
from abc import ABC, abstractmethod
from threading import Thread
from typing import Any

SIZE = 1024

class RPCServer(ABC):
    # [[noreturn]]
    def __init__(self, host:str='0.0.0.0', port:int=8080) -> None:
        self.host = host
        self.port = port
        self.address = (host, port)
        self.__run__()

    def __run__(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(self.address)
            sock.listen()
            while True:
                client, address = sock.accept()
                Thread(target=self.__handle__, args=[client]).start()

    def __handle__(self, client:socket.socket) -> None:
        while True:
            try: s = json.loads(client.recv(SIZE).decode())
            except: break # Client disconnected.
            try: response = self.process(s)
            except Exception as e: pass
                # Send back exeption if function called by client is not registred
                # !!TODO: implement a handling system
                #client.sendall(json.dumps(str(e)).encode())
            else: client.sendall(json.dumps(response).encode())

        client.close()

    @abstractmethod
    def process(self, s : Any):
        pass

class RPCClient:
    def __init__(self, host:str='localhost', port:int=None) -> None:
        self.__sock = None
        self.__address = (host, port)

    def connect(self):
        self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__sock.connect(self.__address)

    def disconnect(self):
        try: self.__sock.close()
        except: pass

    def process(self, s: Any):
        self.__sock.sendall(json.dumps(s).encode())
        return json.loads(self.__sock.recv(SIZE).decode())