import unittest

from rag_system.domain import AnswerResult, Citation, Route, RouteDecision
from rag_system.ui import _file_paths, _sources_markdown


class InterfaceFormattingTests(unittest.TestCase):
    def test_file_paths_normalize_none_single_and_multiple_values(self) -> None:
        self.assertEqual(_file_paths(None), [])
        self.assertEqual(_file_paths("a.txt"), ["a.txt"])
        self.assertEqual(_file_paths(["a.txt", None, "b.md"]), ["a.txt", "b.md"])

    def test_sources_escape_untrusted_markdown_and_canonicalize_links(self) -> None:
        result = AnswerResult(
            answer="answer",
            decision=RouteDecision(Route.WEB, 0.5, "test"),
            citations=(
                Citation(
                    "W1",
                    "<script>title</script>",
                    "line 1\n<script>line 2</script>",
                    "https://example.com/a_(b)?utm_source=x",
                    0.75,
                ),
            ),
        )
        rendered = _sources_markdown(result)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("https://example.com/a_%28b%29", rendered)
        self.assertIn("相关度 75%", rendered)

    def test_empty_citations_have_explicit_message(self) -> None:
        result = AnswerResult("answer", RouteDecision(Route.REFUSED, 0.0, "none"))
        self.assertEqual(_sources_markdown(result), "没有可展示的来源。")


if __name__ == "__main__":
    unittest.main()
