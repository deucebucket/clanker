"""
Tests for the Clank-Lang decoder.

Verifies that decoding works correctly for English, Chinese, and Python dictionaries.
"""

import os
import sys
import pytest

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clank_decoder import decode, ClankDecoder, DictionaryLoader


@pytest.fixture
def loader():
    """Create a loader pointing to the project dictionaries."""
    project_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    return DictionaryLoader(search_paths=[os.path.join(project_root, "dictionaries")])


@pytest.fixture
def decoder(loader):
    """Create a decoder with the test loader."""
    return ClankDecoder(loader)


class TestDictionaryLoader:
    """Tests for the DictionaryLoader."""

    def test_load_english(self, loader):
        d = loader.load("en")
        assert d["language"] == "en"
        assert d["name"] == "English"
        assert "opcodes" in d

    def test_load_chinese(self, loader):
        d = loader.load("zh")
        assert d["language"] == "zh"
        assert "opcodes" in d

    def test_load_python(self, loader):
        d = loader.load("python")
        assert d["language"] == "python"
        assert d["kind"] == "code"

    def test_load_rust(self, loader):
        d = loader.load("rust")
        assert d["language"] == "rust"

    def test_load_javascript(self, loader):
        d = loader.load("javascript")
        assert d["language"] == "javascript"

    def test_load_pseudocode(self, loader):
        d = loader.load("pseudocode")
        assert d["language"] == "pseudocode"
        assert d["kind"] == "other"

    def test_load_nonexistent(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load("klingon")

    def test_available_languages(self, loader):
        langs = loader.available_languages()
        assert "en" in langs
        assert "zh" in langs
        assert "python" in langs
        assert "rust" in langs
        assert "javascript" in langs

    def test_caching(self, loader):
        d1 = loader.load("en")
        d2 = loader.load("en")
        assert d1 is d2  # Same object from cache

    def test_clear_cache(self, loader):
        loader.load("en")
        loader.clear_cache()
        assert "en" not in loader._cache


class TestDecoder:
    """Tests for the ClankDecoder."""

    def test_nop_done_english(self, decoder):
        result = decoder.decode("@ 0x00", "en")
        assert result.strip() == "done"

    def test_nop_done_with_message_english(self, decoder):
        clank = '@ 0x00 $_ $_ 01 {message: "all finished"}'
        result = decoder.decode(clank, "en")
        assert 'done: "all finished"' in result

    def test_set_english(self, decoder):
        clank = '@ 0x02 $0 $_ 02 {name: "x"} {value: "42"}'
        result = decoder.decode(clank, "en")
        assert "set x to 42" in result

    def test_endpoint_english(self, decoder):
        clank = '@ 0xC0 $0 $1 02 {method: "GET"} {path: "/hello"}'
        result = decoder.decode(clank, "en")
        assert "define GET endpoint at /hello" in result

    def test_respond_english(self, decoder):
        clank = '@ 0xC1 $1 $2 01 {status: 200}'
        result = decoder.decode(clank, "en")
        assert "respond with status 200" in result

    def test_endpoint_chinese(self, decoder):
        clank = '@ 0xC0 $0 $1 02 {method: "GET"} {path: "/hello"}'
        result = decoder.decode(clank, "zh")
        assert "GET" in result
        assert "/hello" in result
        assert "定义" in result

    def test_done_chinese(self, decoder):
        result = decoder.decode("@ 0x00", "zh")
        assert "完成" in result

    def test_set_chinese(self, decoder):
        clank = '@ 0x02 $0 $_ 02 {name: "x"} {value: "42"}'
        result = decoder.decode(clank, "zh")
        assert "设置" in result
        assert "x" in result
        assert "42" in result

    def test_endpoint_python(self, decoder):
        clank = '@ 0xC0 $0 $1 02 {method: "GET"} {path: "/hello"}'
        result = decoder.decode(clank, "python")
        assert "@app.route" in result
        assert '"/hello"' in result

    def test_respond_python(self, decoder):
        clank = '@ 0xC1 $1 $2 01 {status: 200}'
        result = decoder.decode(clank, "python")
        assert "return" in result
        assert "200" in result

    def test_multiline_script_english(self, decoder):
        clank = """@ 0xC0 $0 $1 02 {method: "GET"} {path: "/health"}
@ 0xC1 $1 $2 01 {status: 200}
@ 0x00"""
        result = decoder.decode(clank, "en")
        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert "define GET endpoint at /health" in lines[0]
        assert "respond with status 200" in lines[1]
        assert "done" in lines[2]

    def test_indentation_with_blocks(self, decoder):
        clank = """@ 0xE0 $_ $_ 01 {condition: "x > 0"}
@ 0x03 $_ $_ 01 {value: "positive"}
@ 0x0F"""
        result = decoder.decode(clank, "en")
        lines = result.strip().split("\n")
        assert lines[0].startswith("when")
        # The emit line should be indented
        assert lines[1].startswith("  ")
        assert "emit" in lines[1]
        assert lines[2] == "end"

    def test_unknown_opcode(self, decoder):
        clank = "@ 0xFF $_ $_ 00"
        result = decoder.decode(clank, "en")
        assert "unknown opcode" in result.lower()

    def test_skip_comments_and_blanks(self, decoder):
        clank = """# This is a comment
@ 0x00

# Another comment"""
        result = decoder.decode(clank, "en")
        assert result.strip() == "done"

    def test_query_data_english(self, decoder):
        clank = '@ 0xD0 $0 $_ 01 {source: "users"}'
        result = decoder.decode(clank, "en")
        assert "query users" in result

    def test_repeat_english(self, decoder):
        clank = '@ 0xE2 $_ $_ 01 {count: 5}'
        result = decoder.decode(clank, "en")
        assert "repeat 5 times" in result

    def test_wait_english(self, decoder):
        clank = '@ 0x01 $_ $_ 01 {duration: "5s"}'
        result = decoder.decode(clank, "en")
        assert "wait 5s" in result

    def test_log_english(self, decoder):
        clank = '@ 0x04 $_ $_ 02 {level: "info"} {message: "server started"}'
        result = decoder.decode(clank, "en")
        assert "log [info] server started" in result


class TestConvenienceFunction:
    """Tests for the module-level decode() function."""

    def test_decode_english(self, loader):
        result = decode("@ 0x00", "en", loader=loader)
        assert result.strip() == "done"

    def test_decode_chinese(self, loader):
        result = decode("@ 0x00", "zh", loader=loader)
        assert "完成" in result

    def test_decode_python(self, loader):
        clank = '@ 0x02 $0 $_ 02 {name: "x"} {value: "42"}'
        result = decode(clank, "python", loader=loader)
        assert "x = 42" in result


class TestFullScript:
    """Tests for complete Clank scripts end-to-end."""

    def test_hello_world_english(self, decoder):
        clank = """@ 0x10 $_ $_ 01 {text: "Hello World example"}
@ 0x02 $0 $_ 02 {name: "greeting"} {value: "Hello, World!"}
@ 0x03 $0 $_ 01 {value: "greeting"}
@ 0x00"""
        result = decoder.decode(clank, "en")
        assert "Hello World example" in result
        assert "set greeting to Hello, World!" in result
        assert "emit greeting" in result
        assert "done" in result

    def test_web_endpoint_multilang(self, decoder):
        clank = """@ 0xC0 $0 $1 02 {method: "GET"} {path: "/api/status"}
@ 0xC1 $1 $2 01 {status: 200}
@ 0x0F
@ 0x00"""

        en_result = decoder.decode(clank, "en")
        assert "define GET endpoint at /api/status" in en_result

        zh_result = decoder.decode(clank, "zh")
        assert "定义" in zh_result
        assert "/api/status" in zh_result

        py_result = decoder.decode(clank, "python")
        assert "@app.route" in py_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
