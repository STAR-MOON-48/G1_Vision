from __future__ import annotations

import time
import unittest

from face_dds import (
    DdsFaceSubscriber,
    FaceRecognitionEvent,
    RetryingFaceEventSink,
    event_to_message,
    message_to_event,
)
from face_dds_types import DDS_IDL_AVAILABLE, FaceRecognitionMessage


def make_event(event_id: str = "face-event-1") -> FaceRecognitionEvent:
    return FaceRecognitionEvent(
        event_id=event_id,
        session_id="face-session-1",
        sequence=1,
        created_unix_ns=123,
        source="dgx_face",
        frame_id="camera_color_optical_frame",
        faces=[{"person_id": "FACE_TEAM_001", "similarity": 0.75}],
        inference_ms=12.5,
        model="buffalo_l",
    )


class FakeWriter:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.event_ids = []

    def start(self):
        pass

    def write(self, event, timeout):
        self.event_ids.append(event.event_id)
        return next(self.outcomes, True)

    def close(self):
        pass


class FaceDdsTests(unittest.TestCase):
    def test_message_round_trip(self):
        event = make_event()
        self.assertEqual(message_to_event(event_to_message(event)), event)

    def test_cyclonedds_idl_type_can_be_populated(self):
        if DDS_IDL_AVAILABLE:
            FaceRecognitionMessage.__idl__.populate()

    def test_retry_reuses_event_id(self):
        writer = FakeWriter([False, False, True])
        sink = RetryingFaceEventSink(
            writer,
            capacity=4,
            write_timeout_seconds=0.01,
            retry_interval_seconds=0.01,
            delivery_ttl_seconds=2.0,
        )
        sink.start()
        self.assertTrue(sink.publish(make_event()))
        deadline = time.monotonic() + 1.0
        while sink.delivered == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        sink.close()
        self.assertEqual(sink.delivered, 1)
        self.assertEqual(writer.event_ids, ["face-event-1"] * 3)

    def test_agent_subscriber_deduplicates_retries(self):
        received = []
        subscriber = DdsFaceSubscriber(received.append)
        message = event_to_message(make_event())
        subscriber._on_message(message)
        subscriber._on_message(message)
        self.assertEqual([event.event_id for event in received], ["face-event-1"])
        self.assertEqual(subscriber.duplicates, 1)


if __name__ == "__main__":
    unittest.main()
