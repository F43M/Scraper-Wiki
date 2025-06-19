import os
import json
from queue import Queue as _Queue

try:
    import pika
except Exception:
    pika = None

class BaseQueue:
    """Abstract queue backend."""
    def publish(self, queue: str, message: dict):
        raise NotImplementedError

    def consume(self, queue: str):
        raise NotImplementedError

class InMemoryQueue(BaseQueue):
    def __init__(self):
        self.queues = {}

    def _get_queue(self, name: str):
        return self.queues.setdefault(name, _Queue())

    def publish(self, queue: str, message: dict):
        self._get_queue(queue).put(json.dumps(message))

    def consume(self, queue: str):
        q = self._get_queue(queue)
        while True:
            msg = q.get()
            yield json.loads(msg)
            q.task_done()

class RabbitMQQueue(BaseQueue):
    def __init__(self, url: str):
        if pika is None:
            raise RuntimeError("pika not installed")
        self.connection = pika.BlockingConnection(pika.URLParameters(url))
        self.channel = self.connection.channel()

    def publish(self, queue: str, message: dict):
        self.channel.queue_declare(queue=queue, durable=True)
        body = json.dumps(message).encode()
        self.channel.basic_publish(exchange='', routing_key=queue, body=body)

    def consume(self, queue: str):
        self.channel.queue_declare(queue=queue, durable=True)
        for method, _, body in self.channel.consume(queue, inactivity_timeout=1):
            if body:
                self.channel.basic_ack(method.delivery_tag)
                yield json.loads(body.decode())

_BACKEND = None

def get_backend() -> BaseQueue:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    url = os.environ.get('QUEUE_URL')
    if url and pika:
        _BACKEND = RabbitMQQueue(url)
    else:
        _BACKEND = InMemoryQueue()
    return _BACKEND


def publish(queue_name: str, message: dict):
    get_backend().publish(queue_name, message)


def consume(queue_name: str):
    yield from get_backend().consume(queue_name)
