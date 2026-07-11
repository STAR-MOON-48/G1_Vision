"""Native Unitree SDK2/CycloneDDS types for face-recognition events.

Do not enable postponed annotations here. CycloneDDS 0.10.2 resolves IDL field
types at runtime and requires concrete built-in/IDL annotations.
"""

from dataclasses import dataclass

try:
    from cyclonedds.idl import IdlStruct
    from cyclonedds.idl import types as idl_types

    DDS_IDL_AVAILABLE = True

    @dataclass
    class FaceRecognitionMessage(IdlStruct, typename="g1_hri.msg.FaceRecognition"):
        event_id: str
        session_id: str
        sequence: idl_types.uint64
        created_unix_ns: idl_types.uint64
        source: str
        frame_id: str
        faces_json: str
        inference_ms: idl_types.float32
        model: str
        is_final: bool

except ImportError:
    DDS_IDL_AVAILABLE = False

    @dataclass
    class FaceRecognitionMessage:
        event_id: str
        session_id: str
        sequence: int
        created_unix_ns: int
        source: str
        frame_id: str
        faces_json: str
        inference_ms: float
        model: str
        is_final: bool
