# Adapted from
# https://medium.com/@taraszhere/coding-remote-procedure-call-rpc-with-python-3b14a7d00ac8

import json
import socket
from enum import IntEnum
from abc import ABC, abstractmethod
from threading import Thread
from typing import Any
from time import sleep
import warnings

SIZE = 1048576 # !!TODO: choose a smaller size

class RPCMsgKind(IntEnum):
    HANDSHAKE = 1
    MESSAGE = 2
    ERROR = 4

# Better defined as a dependent type:
# Kinds of valid values:
# (HANDSHAKE, bool):
# - (HANDSHAKE, True): Indicates readiness
# - (HANDSHAKE, False): Indicates not-ready
# (REQUEST, Any): Request from client to process the content
# (ERROR, str): Error message

handshake_message = (RPCMsgKind.HANDSHAKE, True)

class RPCServer(ABC):
    def __init__(self, host:str='0.0.0.0', port:int=8080) -> None:
        self.host = host
        self.port = port
        self.address = (host, port)

    def run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(self.address)
            sock.listen()
            while True:
                client, address = sock.accept()
                Thread(target=self.__handle__, args=[client]).start()

    def __handle__(self, client:socket.socket) -> None:
        while True:
            try:
                s : tuple[RPCMsgKind, Any] =\
                        json.loads(client.recv(SIZE).decode())
                # !!TODO: chop up into smaller chunks
            except: break # Client disconnected.
            try:
                match s[0]:
                    case RPCMsgKind.HANDSHAKE:
                        response = (RPCMsgKind.HANDSHAKE, bool(s[1]))
                    case RPCMsgKind.MESSAGE:
                        response = (RPCMsgKind.MESSAGE, self.process(s[1]))
                    case RPCMsgKind.ERROR:
                        response = (RPCMsgKind.ERROR, "Received error message")
                    case _:
                        response = (RPCMsgKind.ERROR, "Unknown message type: "+
                            f"{s[0]}")
            except Exception as e:
                response = (RPCMsgKind.ERROR, e.__repr__())

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

    def query(self, s: Any):
        while True:
            try:
                self.connect()
                break
            except Exception as e:
                # "Still connecting to server
                pass
            sleep(1)
        while True:
            try:
                self.__sock.sendall(json.dumps(handshake_message).encode())
                response = json.loads(self.__sock.recv(SIZE).decode())
                if response[0] == RPCMsgKind.HANDSHAKE\
                    and response[1]:
                    break
                else:
                    raise ValueError("Received invalid response from server: "\
                        +f"{response}")
            except Exception as e:
                warnings.warn(f"Pinging failed: {e.__repr__()}")
            sleep(1)
        self.__sock.sendall(json.dumps((RPCMsgKind.MESSAGE, s)).encode())
        ans = json.loads(self.__sock.recv(SIZE).decode())
        self.disconnect()
        return ans
