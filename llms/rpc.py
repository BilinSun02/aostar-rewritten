# Adapted from
# https://medium.com/@taraszhere/coding-remote-procedure-call-rpc-with-python-3b14a7d00ac8

import json
import socket
from enum import Enum
from abc import ABC, abstractmethod
from threading import Thread
from typing import Any

SIZE = 1024

class RPCMsgKind(Enum):
    HANDSHAKE = 1
    MESSAGE = 2
    ERROR = 4

type RPCMessage = tuple[RPCMsgKind, Any]
# Better defined as a dependent type:
# Kinds of valid values:
# (HANDSHAKE, bool):
# - (HANDSHAKE, True): Indicates readiness
# - (HANDSHAKE, False): Indicates not-ready
# (REQUEST, Any): Request from client to process the content
# (ERROR, str): Error message

handshake_message = (RPCMsgKind.HANDSHAKE, True)

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
            try:
                assert isinstance(s, RPCMessage)
                match s[0]:
                    case RPCMsgKind.HANDSHAKE:
                        response = (RPCMsgKind.HANDSHAKE, bool(s[2]))
                    case RPCMsgKind.REQUEST:
                        response = (RPCMsgKind.MESSAGE, self.process(s))
                    case RPCMsgKind.ERROR:
                        response = (RPCMsgKind.ERROR, "Received error message")
                    case _:
                        response = (RPCMsgKind.ERROR, "Unknown message type")
            except Exception as e:
                response = (RPCMsgKind.ERROR, str(e))

            client.sendall(json.dumps(response).encode())

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
        while True:
            try:
                self.__sock.sendall(json.dumps(handshake_message).encode())
                response = json.loads(self.__sock.recv(SIZE).decode())
                if response[0] == RPCMsgKind.HANDSHAKE\
                    and response[1]:
                    break
            except:
                pass
        self.__sock.sendall(json.dumps(s).encode())
        return json.loads(self.__sock.recv(SIZE).decode())