"""Manifest drift guard: verify that ARTIFACTS.json entries match local source artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SIZE_CAP_BYTES = 50 * 1024 * 1024  # hash files under 50MB directly in local tests


class TestArtifactsManifest(unittest.TestCase):
    def test_artifacts_manifest_matches_sources(self) -> None:
        manifest_path = REPO_ROOT / "artifacts" / "ARTIFACTS.json"
        self.assertTrue(manifest_path.exists(), f"missing manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["repo_id"], "heman10x/rlcd-modernbert-151m")

        for key, meta in manifest["files"].items():
            src = REPO_ROOT / meta["source_file"]
            self.assertTrue(src.exists(), f"missing source for {key}: {meta['source_file']}")
            size = os.path.getsize(src)
            self.assertEqual(size, meta["bytes"], f"{key}: byte size mismatch ({size} != {meta['bytes']})")
            if size <= SIZE_CAP_BYTES:
                digest = hashlib.sha256(src.read_bytes()).hexdigest()
                self.assertEqual(digest, meta["sha256"], f"{key}: sha256 mismatch ({digest} != {meta['sha256']})")


if __name__ == "__main__":
    unittest.main()
