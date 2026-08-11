from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor

from rag_system.memory import ConversationMemory


class ConversationMemoryTests(unittest.TestCase):
    def test_sessions_are_private_and_clear_is_scoped(self) -> None:
        memory = ConversationMemory()
        memory.add_turn("session-a", "甲的问题", "仅属于甲的答案")
        memory.add_turn("session-b", "乙的问题", "仅属于乙的答案")

        summary_a = memory.summarize("session-a")
        summary_b = memory.summarize("session-b")
        self.assertIn("仅属于甲", summary_a)
        self.assertNotIn("仅属于乙", summary_a)
        self.assertIn("仅属于乙", summary_b)
        self.assertNotIn("仅属于甲", summary_b)

        self.assertTrue(memory.clear("session-a"))
        self.assertFalse(memory.clear("session-a"))
        self.assertEqual(memory.history("session-a"), ())
        self.assertEqual(len(memory.history("session-b")), 1)

    def test_capacity_uses_lru_order(self) -> None:
        memory = ConversationMemory(max_sessions=2)
        memory.add_turn("old", "问题", "旧会话")
        memory.add_turn("recent", "问题", "新会话")
        memory.history("old")  # Reading refreshes TTL and LRU position.
        memory.add_turn("third", "问题", "第三个会话")

        self.assertTrue(memory.has_session("old"))
        self.assertFalse(memory.has_session("recent"))
        self.assertTrue(memory.has_session("third"))

    def test_ttl_expires_at_exact_boundary(self) -> None:
        now = [0.0]
        memory = ConversationMemory(ttl_seconds=10, clock=lambda: now[0])
        memory.add_turn("session", "问题", "答案")
        now[0] = 9.99
        self.assertTrue(memory.has_session("session"))

        # The preceding read refreshes activity at 9.99.
        now[0] = 19.99
        self.assertFalse(memory.has_session("session"))
        self.assertEqual(memory.stats()["sessions"], 0)

    def test_round_character_and_summary_bounds(self) -> None:
        memory = ConversationMemory(
            max_rounds=2,
            max_characters=40,
            summary_max_characters=35,
        )
        memory.add_turn("s", "第一问", "第一答")
        memory.add_turn("s", "第二问", "第二答")
        memory.add_turn("s", "第三问" * 20, "第三答" * 20)

        history = memory.history("s")
        self.assertLessEqual(len(history), 2)
        self.assertLessEqual(sum(turn.character_count for turn in history), 40)
        self.assertLessEqual(len(memory.summarize("s")), 35)
        self.assertLessEqual(len(memory.summarize("s", max_characters=7)), 7)

    def test_contextualized_query_has_deterministic_labels(self) -> None:
        memory = ConversationMemory()
        self.assertEqual(memory.contextualize_query("new", " 当前问题 "), "当前问题")
        memory.add_turn("s", "上一问", "上一答")
        combined = memory.contextualize_query("s", "下一问")
        self.assertIn("历史对话（仅作为上下文）", combined)
        self.assertIn("问题：上一问", combined)
        self.assertTrue(combined.endswith("当前问题：下一问"))

    def test_invalid_boundaries_are_rejected(self) -> None:
        for kwargs in (
            {"max_sessions": 0},
            {"ttl_seconds": 0},
            {"max_rounds": 0},
            {"max_characters": 1},
            {"summary_max_characters": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ConversationMemory(**kwargs)

        memory = ConversationMemory()
        with self.assertRaises(ValueError):
            memory.add_turn("", "问题", "答案")
        with self.assertRaises(ValueError):
            memory.add_turn("s", "", "答案")
        with self.assertRaises(ValueError):
            memory.summarize("s", max_characters=0)

    def test_basic_concurrent_writes_remain_consistent(self) -> None:
        memory = ConversationMemory(max_rounds=50, max_characters=10_000)

        def write_turn(number: int) -> None:
            memory.add_turn("shared", f"问题-{number}", f"答案-{number}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write_turn, range(200)))

        history = memory.history("shared")
        sequences = [turn.sequence for turn in history]
        self.assertEqual(len(history), 50)
        self.assertEqual(sequences, sorted(sequences))
        self.assertEqual(len(set(sequences)), len(sequences))
        self.assertLessEqual(memory.stats()["characters"], 10_000)


if __name__ == "__main__":
    unittest.main()
