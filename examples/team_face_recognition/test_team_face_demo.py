import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("team_face_demo.py")
SPEC = importlib.util.spec_from_file_location("team_face_demo", MODULE_PATH)
assert SPEC and SPEC.loader
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)


class TeamFaceDemoTests(unittest.TestCase):
    def test_identity_directories_excludes_test_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            team = root / "FACE_TEAM_002"
            test_data = root / "test_data"
            team.mkdir()
            test_data.mkdir()
            (team / "member.jpg").write_bytes(b"placeholder")
            (test_data / "query.jpg").write_bytes(b"placeholder")
            self.assertEqual(DEMO.identity_directories(root), [team])

    def test_augmentation_keeps_shape_and_changes_pixels(self):
        image = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
        labels = set()
        for variant in range(8):
            label, augmented = DEMO.augment_face_crop(image, variant)
            labels.add(label)
            self.assertEqual(augmented.shape, image.shape)
            self.assertFalse(np.array_equal(augmented, image))
        self.assertEqual(len(labels), 8)

    def test_match_embedding_known_and_unknown(self):
        gallery = [
            {
                "person_id": "TEAM_001",
                "display_name": "TEAM_001",
                "centroid": DEMO.normalize(np.array([1.0, 0.0], dtype=np.float32)),
                "sample_embeddings": [DEMO.normalize(np.array([1.0, 0.0], dtype=np.float32))],
                "sample_count": 1,
            },
            {
                "person_id": "TEAM_002",
                "display_name": "TEAM_002",
                "centroid": DEMO.normalize(np.array([0.0, 1.0], dtype=np.float32)),
                "sample_embeddings": [DEMO.normalize(np.array([0.0, 1.0], dtype=np.float32))],
                "sample_count": 1,
            },
        ]
        known = DEMO.match_embedding(np.array([0.99, 0.05]), gallery, threshold=0.4, min_margin=0.05)
        unknown = DEMO.match_embedding(np.array([-1.0, -1.0]), gallery, threshold=0.4, min_margin=0.05)
        self.assertEqual(known["status"], "KNOWN")
        self.assertEqual(known["person_id"], "TEAM_001")
        self.assertEqual(unknown["status"], "UNKNOWN")
        self.assertIsNone(unknown["person_id"])

    def test_database_round_trip_and_gallery(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "faces.sqlite3"
            with DEMO.FaceDatabase(database_path) as database:
                database.upsert_person("TEAM_001")
                database.add_sample("TEAM_001", "one.jpg", np.array([3.0, 4.0]), 0.9, 100.0, "test")
            with DEMO.FaceDatabase(database_path) as database:
                samples = database.samples()
            gallery = DEMO.build_gallery(samples)
            self.assertEqual(len(samples), 1)
            self.assertEqual(gallery[0]["person_id"], "TEAM_001")
            np.testing.assert_allclose(np.linalg.norm(samples[0]["embedding"]), 1.0, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
