#!/usr/bin/env python3
"""Native Unitree SDK2 DDS relay and Agent subscriber for face events."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import socket
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Protocol

try:
    from .face_dds_types import DDS_IDL_AVAILABLE, FaceRecognitionMessage
except ImportError:
    from face_dds_types import DDS_IDL_AVAILABLE, FaceRecognitionMessage


logger = logging.getLogger(__name__)
DEFAULT_TOPIC = "rt/g1/hri/vision/face_recognition"


@dataclass(frozen=True)
class FaceRecognitionEvent:
    event_id: str
    session_id: str
    sequence: int
    created_unix_ns: int
    source: str
    frame_id: str
    faces: list[dict[str, Any]]
    inference_ms: float
    model: str
    is_final: bool = True

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "FaceRecognitionEvent":
        if payload.get("event") != "face_recognition":
            raise ValueError("UDP payload is not a face_recognition event")
        source = payload.get("source") or {}
        faces = payload.get("faces")
        if not isinstance(faces, list):
            raise ValueError("faces must be a list")
        return cls(
            event_id=str(payload.get("event_id") or uuid.uuid4()),
            session_id=str(payload.get("session_id") or "legacy-face-session"),
            sequence=int(payload.get("sequence", 0)),
            created_unix_ns=int(payload.get("created_unix_ns") or time.time_ns()),
            source=str(payload.get("source_name") or "dgx_face"),
            frame_id=str(source.get("frame_id") or ""),
            faces=faces,
            inference_ms=float(payload.get("inference_ms", 0.0)),
            model=str(payload.get("model") or ""),
            is_final=bool(payload.get("is_final", True)),
        )


def initialize_unitree_dds(domain_id: int = 0, network_interface: str | None = None) -> None:
    if not DDS_IDL_AVAILABLE:
        raise RuntimeError("Unitree SDK2/CycloneDDS IDL runtime is unavailable")
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(domain_id, network_interface)


def event_to_message(event: FaceRecognitionEvent) -> FaceRecognitionMessage:
    return FaceRecognitionMessage(
        event_id=event.event_id,
        session_id=event.session_id,
        sequence=event.sequence,
        created_unix_ns=event.created_unix_ns,
        source=event.source,
        frame_id=event.frame_id,
        faces_json=json.dumps(event.faces, ensure_ascii=False, separators=(",", ":")),
        inference_ms=event.inference_ms,
        model=event.model,
        is_final=event.is_final,
    )


def message_to_event(message: FaceRecognitionMessage) -> FaceRecognitionEvent:
    faces = json.loads(message.faces_json)
    if not isinstance(faces, list):
        raise ValueError("FaceRecognition.faces_json must decode to a list")
    return FaceRecognitionEvent(
        event_id=message.event_id,
        session_id=message.session_id,
        sequence=int(message.sequence),
        created_unix_ns=int(message.created_unix_ns),
        source=message.source,
        frame_id=message.frame_id,
        faces=faces,
        inference_ms=float(message.inference_ms),
        model=message.model,
        is_final=bool(message.is_final),
    )


class FaceEventWriter(Protocol):
    def start(self) -> None: ...
    def write(self, event: FaceRecognitionEvent, timeout: float) -> bool: ...
    def close(self) -> None: ...


class UnitreeDdsFaceWriter:
    def __init__(self, topic: str = DEFAULT_TOPIC) -> None:
        self._topic = topic
        self._publisher = None

    def start(self) -> None:
        if self._publisher is not None:
            return
        from unitree_sdk2py.core.channel import ChannelPublisher

        self._publisher = ChannelPublisher(self._topic, FaceRecognitionMessage)
        self._publisher.Init()
        logger.info("DDS FaceRecognition publisher: %s", self._topic)

    def write(self, event: FaceRecognitionEvent, timeout: float) -> bool:
        if self._publisher is None:
            raise RuntimeError("DDS face publisher has not started")
        return bool(self._publisher.Write(event_to_message(event), timeout))

    def close(self) -> None:
        if self._publisher is not None:
            self._publisher.Close()
            self._publisher = None


@dataclass
class _Pending:
    event: FaceRecognitionEvent
    expires_monotonic: float


class RetryingFaceEventSink:
    """Bounded DDS outbox; retries preserve event_id for Agent deduplication."""

    def __init__(
        self,
        writer: FaceEventWriter,
        *,
        capacity: int = 64,
        write_timeout_seconds: float = 0.25,
        retry_interval_seconds: float = 0.1,
        delivery_ttl_seconds: float = 5.0,
    ) -> None:
        self._writer = writer
        self._capacity = capacity
        self._write_timeout = write_timeout_seconds
        self._retry_interval = retry_interval_seconds
        self._delivery_ttl = delivery_ttl_seconds
        self._pending: deque[_Pending] = deque()
        self._condition = threading.Condition()
        self._stop = False
        self._thread: threading.Thread | None = None
        self.delivered = 0
        self.expired = 0
        self.dropped = 0
        self.retries = 0

    @property
    def queue_size(self) -> int:
        with self._condition:
            return len(self._pending)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._writer.start()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="face-dds-outbox", daemon=True)
        self._thread.start()

    def publish(self, event: FaceRecognitionEvent) -> bool:
        with self._condition:
            if len(self._pending) >= self._capacity:
                self.dropped += 1
                return False
            self._pending.append(
                _Pending(event=event, expires_monotonic=time.monotonic() + self._delivery_ttl)
            )
            self._condition.notify()
            return True

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=self._write_timeout + 2.0)
            self._thread = None
        self._writer.close()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stop:
                    self._condition.wait(timeout=0.5)
                if self._stop:
                    return
                pending = self._pending[0]
            if time.monotonic() >= pending.expires_monotonic:
                with self._condition:
                    if self._pending and self._pending[0] is pending:
                        self._pending.popleft()
                        self.expired += 1
                logger.warning("Face event expired: %s", pending.event.event_id)
                continue
            try:
                delivered = self._writer.write(pending.event, self._write_timeout)
            except Exception:
                logger.exception("DDS face write failed; retrying event_id=%s", pending.event.event_id)
                delivered = False
            if delivered:
                with self._condition:
                    if self._pending and self._pending[0] is pending:
                        self._pending.popleft()
                        self.delivered += 1
                continue
            self.retries += 1
            with self._condition:
                if self._stop:
                    return
                self._condition.wait(timeout=self._retry_interval)


class EventDeduplicator:
    def __init__(self, *, capacity: int = 4096, ttl_seconds: float = 3600.0) -> None:
        self._capacity = capacity
        self._ttl = ttl_seconds
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def accept(self, event_id: str, *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        with self._lock:
            while self._seen:
                _, timestamp = next(iter(self._seen.items()))
                if current - timestamp <= self._ttl:
                    break
                self._seen.popitem(last=False)
            if event_id in self._seen:
                self._seen.move_to_end(event_id)
                return False
            self._seen[event_id] = current
            while len(self._seen) > self._capacity:
                self._seen.popitem(last=False)
            return True


class DdsFaceSubscriber:
    """Agent-side subscriber; callback should only enqueue an observation."""

    def __init__(
        self,
        callback: Callable[[FaceRecognitionEvent], None],
        *,
        topic: str = DEFAULT_TOPIC,
        deduplicator: EventDeduplicator | None = None,
        queue_len: int = 32,
    ) -> None:
        self._callback = callback
        self._topic = topic
        self._dedupe = deduplicator or EventDeduplicator()
        self._queue_len = queue_len
        self._subscriber = None
        self.duplicates = 0

    def start(self) -> None:
        from unitree_sdk2py.core.channel import ChannelSubscriber

        self._subscriber = ChannelSubscriber(self._topic, FaceRecognitionMessage)
        self._subscriber.Init(self._on_message, self._queue_len)
        logger.info("DDS FaceRecognition subscriber: %s", self._topic)

    def close(self) -> None:
        if self._subscriber is not None:
            self._subscriber.Close()
            self._subscriber = None

    def _on_message(self, message: FaceRecognitionMessage) -> None:
        event = message_to_event(message)
        if not self._dedupe.accept(event.event_id):
            self.duplicates += 1
            return
        self._callback(event)


class UdpToDdsRelay:
    def __init__(
        self,
        sink: RetryingFaceEventSink,
        *,
        host: str = "127.0.0.1",
        port: int = 17171,
    ) -> None:
        self._sink = sink
        self._address = (host, port)
        self._stop = threading.Event()
        self._socket: socket.socket | None = None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.bind(self._address)
        udp_socket.settimeout(0.5)
        self._socket = udp_socket
        self._sink.start()
        logger.info("Face UDP->DDS relay listening on %s:%d", *self._address)
        try:
            while not self._stop.is_set():
                try:
                    encoded, _ = udp_socket.recvfrom(65535)
                except socket.timeout:
                    continue
                try:
                    payload = json.loads(encoded.decode("utf-8"))
                    event = FaceRecognitionEvent.from_payload(payload)
                    if not self._sink.publish(event):
                        logger.warning("Face DDS outbox full; dropped event_id=%s", event.event_id)
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                    logger.exception("Invalid face event received over UDP")
        finally:
            udp_socket.close()
            self._socket = None
            self._sink.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network_interface")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=17171)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    initialize_unitree_dds(args.domain, args.network_interface)
    relay = UdpToDdsRelay(
        RetryingFaceEventSink(UnitreeDdsFaceWriter(args.topic)),
        host=args.udp_host,
        port=args.udp_port,
    )
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _signum, _frame: relay.stop())
    relay.run()


if __name__ == "__main__":
    main()
