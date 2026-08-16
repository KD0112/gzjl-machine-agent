from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import ValidationError

from agent_graph import (
    build_graph,
    create_sqlite_checkpointer,
    resume_image_confirmation,
    start_graph_agent,
)
from handoff_repository import HandoffRepository
from image_evidence import (
    ImageEvidenceRepository,
    ImagePolicy,
    ImageValidationError,
    validate_image_upload,
)
from memory_repository import MemoryRepository
from schemas import ImageInspectionResult


def make_test_image(
    *,
    image_format: str = "PNG",
    size: tuple[int, int] = (900, 600),
    text: str = "KOMATSU PC200 708-2L-00300",
) -> bytes:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 120, 820, 480), outline="black", width=8)
    draw.text((130, 250), text, fill="black")
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def fake_good_inspector(**kwargs):
    return {
        "evidence_id": kwargs["evidence_id"],
        "inspection": {
            "image_type": "nameplate",
            "extracted_text": ["KOMATSU", "PC200", "708-2L-00300"],
            "brand": "小松",
            "machine_model": "PC200",
            "part_name_candidate": "液压泵",
            "part_number": "708-2L-00300",
            "visible_damage": [],
            "observed_features": ["金属铭牌"],
            "image_quality": "good",
            "confidence": 0.94,
            "warnings": ["适配关系仍需人工核对"],
            "required_followups": [],
            "safe_for_auto_merge": True,
        },
        "model_runtime": {
            "route": {"provider": "fake", "model": "fake-vision"},
            "calls": 1,
            "attempts": 1,
            "successes": 1,
            "failures": 0,
        },
    }


def fake_poor_inspector(**kwargs):
    return {
        "evidence_id": kwargs["evidence_id"],
        "inspection": {
            "image_type": "unknown",
            "extracted_text": [],
            "brand": None,
            "machine_model": None,
            "part_name_candidate": None,
            "part_number": None,
            "visible_damage": [],
            "observed_features": [],
            "image_quality": "poor",
            "confidence": 0.2,
            "warnings": ["图片模糊"],
            "required_followups": ["请重拍铭牌"],
            "safe_for_auto_merge": False,
        },
        "model_runtime": {
            "route": {"provider": "fake", "model": "fake-vision"},
            "calls": 1,
            "attempts": 1,
            "successes": 1,
            "failures": 0,
        },
    }


class ImageValidationTests(unittest.TestCase):
    def test_accepts_and_reencodes_supported_png(self) -> None:
        validated = validate_image_upload(
            make_test_image(),
            filename="nameplate.png",
            claimed_mime_type="image/png",
        )
        self.assertEqual(validated.image_format, "PNG")
        self.assertEqual(validated.mime_type, "image/png")
        self.assertEqual(validated.width, 900)
        self.assertEqual(len(validated.sha256), 64)

    def test_rejects_extension_mime_and_content_mismatch(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_image_upload(
                make_test_image(),
                filename="fake.jpg",
                claimed_mime_type="image/jpeg",
            )

    def test_rejects_corrupt_or_tiny_images(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_image_upload(
                b"not an image",
                filename="broken.png",
                claimed_mime_type="image/png",
            )
        with self.assertRaises(ImageValidationError):
            validate_image_upload(
                make_test_image(size=(32, 32)),
                filename="tiny.png",
                claimed_mime_type="image/png",
            )

    def test_rejects_files_over_byte_budget(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_image_upload(
                make_test_image(),
                filename="large.png",
                claimed_mime_type="image/png",
                policy=ImagePolicy(max_bytes=100),
            )


class ImageEvidenceRepositoryTests(unittest.TestCase):
    def test_customer_isolation_and_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = ImageEvidenceRepository(
                db_path=root / "evidence.sqlite3",
                evidence_dir=root / "files",
            )
            validated = validate_image_upload(
                make_test_image(),
                filename="nameplate.png",
                claimed_mime_type="image/png",
            )
            metadata = repository.store(
                validated,
                customer_id="customer-a",
                session_id="session-a",
            )
            content, loaded = repository.read_content(
                metadata["evidence_id"],
                customer_id="customer-a",
            )

            self.assertTrue(content)
            self.assertEqual(loaded["sha256"], validated.sha256)
            with self.assertRaises(KeyError):
                repository.read_content(
                    metadata["evidence_id"],
                    customer_id="customer-b",
                )
            self.assertTrue(
                repository.delete(
                    metadata["evidence_id"],
                    customer_id="customer-a",
                )
            )


class ImageSchemaTests(unittest.TestCase):
    def test_schema_forbids_unknown_fields_and_invalid_confidence(self) -> None:
        payload = {
            "image_type": "nameplate",
            "extracted_text": [],
            "visible_damage": [],
            "observed_features": [],
            "image_quality": "good",
            "confidence": 1.2,
            "warnings": [],
            "required_followups": [],
            "safe_for_auto_merge": False,
            "invented_field": "not allowed",
        }
        with self.assertRaises(ValidationError):
            ImageInspectionResult.model_validate(payload)


class MultimodalGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.image_repository = ImageEvidenceRepository(
            db_path=root / "images.sqlite3",
            evidence_dir=root / "evidence",
        )
        self.handoff_repository = HandoffRepository(root / "handoff.sqlite3")
        self.memory_repository = MemoryRepository(root / "memory.sqlite3")
        self.checkpointer = create_sqlite_checkpointer(root / "checkpoint.sqlite3")
        validated = validate_image_upload(
            make_test_image(),
            filename="nameplate.png",
            claimed_mime_type="image/png",
        )
        self.attachment = self.image_repository.store(
            validated,
            customer_id="customer-image",
            session_id="session-image",
        )

    def tearDown(self) -> None:
        self.checkpointer.conn.close()
        self.temp_dir.cleanup()

    def _graph(self, inspector):
        return build_graph(
            checkpointer=self.checkpointer,
            handoff_repository=self.handoff_repository,
            memory_repository=self.memory_repository,
            image_repository=self.image_repository,
            vision_inspector=inspector,
        )

    def test_confirmed_image_fields_enter_slots_after_interrupt(self) -> None:
        graph = self._graph(fake_good_inspector)
        thread_id = "image-confirm-thread"
        started = start_graph_agent(
            "这个液压泵多少钱？",
            thread_id=thread_id,
            approval_mode="auto",
            handoff_mode="off",
            parser_mode="rules",
            customer_id="customer-image",
            session_id="session-image",
            attachments=[self.attachment],
            graph=graph,
        )
        self.assertEqual(started["status"], "waiting_image_confirmation")
        self.assertEqual(
            started["image_confirmation_request"]["candidate_fields"]["part_number"],
            "708-2L-00300",
        )

        resumed = resume_image_confirmation(
            thread_id,
            "confirm",
            graph=graph,
        )
        slots = resumed["parse_result"]["slots"]
        self.assertEqual(slots["brand"], "小松")
        self.assertEqual(slots["machine_model"], "PC200")
        self.assertEqual(slots["part_name"], "液压泵")
        self.assertEqual(slots["part_number"], "708-2L-00300")
        self.assertEqual(
            resumed["parse_result"]["slot_sources"]["part_number"],
            "confirmed_image",
        )

    def test_customer_rejection_routes_to_human(self) -> None:
        graph = self._graph(fake_good_inspector)
        thread_id = "image-reject-thread"
        started = start_graph_agent(
            "帮我看看这个配件",
            thread_id=thread_id,
            approval_mode="auto",
            handoff_mode="manual",
            parser_mode="rules",
            customer_id="customer-image",
            session_id="session-image",
            attachments=[self.attachment],
            graph=graph,
        )
        self.assertEqual(started["status"], "waiting_image_confirmation")

        resumed = resume_image_confirmation(
            thread_id,
            "reject",
            graph=graph,
        )
        self.assertEqual(resumed["status"], "waiting_human")
        self.assertEqual(
            resumed["handoff_reason"]["reason_code"],
            "vision_customer_rejected",
        )
        self.assertEqual(resumed["confirmed_visual_slots"], {})

    def test_image_only_request_returns_confirmed_fields_without_handoff(self) -> None:
        graph = self._graph(fake_good_inspector)
        thread_id = "image-only-thread"
        started = start_graph_agent(
            "请识别图片上的品牌、配件名称和零件号，先不要查询价格。",
            thread_id=thread_id,
            approval_mode="auto",
            handoff_mode="manual",
            parser_mode="rules",
            customer_id="customer-image",
            session_id="session-image",
            attachments=[self.attachment],
            graph=graph,
        )
        self.assertEqual(started["status"], "waiting_image_confirmation")

        resumed = resume_image_confirmation(
            thread_id,
            "confirm",
            graph=graph,
        )
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["parse_result"]["intents"], ["image_inspection"])
        self.assertEqual(
            resumed["parse_result"]["slots"]["part_number"],
            "708-2L-00300",
        )
        self.assertFalse(resumed.get("handoff_id"))
        self.assertIn("708-2L-00300", resumed["customer_reply"])

    def test_low_quality_image_is_not_merged_and_handoffs(self) -> None:
        graph = self._graph(fake_poor_inspector)
        result = start_graph_agent(
            "帮我看看这个配件",
            thread_id="image-poor-thread",
            approval_mode="auto",
            handoff_mode="manual",
            parser_mode="rules",
            customer_id="customer-image",
            session_id="session-image",
            attachments=[self.attachment],
            graph=graph,
        )
        self.assertEqual(result["status"], "waiting_human")
        self.assertEqual(result["vision_status"], "needs_better_image")
        self.assertEqual(result["confirmed_visual_slots"], {})
        self.assertEqual(
            result["handoff_reason"]["reason_code"],
            "vision_low_quality",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
