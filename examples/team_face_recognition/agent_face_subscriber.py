#!/usr/bin/env python3
"""Standalone Agent-side native DDS subscription example."""

from __future__ import annotations

import argparse
import queue
import time

from face_dds import DdsFaceSubscriber, initialize_unitree_dds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("network_interface")
    parser.add_argument("--domain", type=int, default=0)
    args = parser.parse_args()

    observations: queue.Queue = queue.Queue(maxsize=32)

    def enqueue(event) -> None:
        try:
            observations.put_nowait(event)
        except queue.Full:
            pass

    # In the real Agent, omit this call if another SDK2 component initialized DDS.
    initialize_unitree_dds(args.domain, args.network_interface)
    subscriber = DdsFaceSubscriber(enqueue)
    subscriber.start()
    try:
        while True:
            try:
                event = observations.get(timeout=1.0)
            except queue.Empty:
                continue
            print(
                f"Agent perception <- [{event.event_id}] "
                f"frame={event.frame_id} faces={event.faces}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.close()


if __name__ == "__main__":
    main()
