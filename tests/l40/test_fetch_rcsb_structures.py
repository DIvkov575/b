import json
from unittest.mock import MagicMock

import pytest

from src.l40.fetch_rcsb_structures import (
    fetch_one,
    load_existing_ids,
    parse_fasta_response,
    parse_structure_id,
)


class TestParseStructureId:
    def test_splits_pdbid_and_chain_from_npz_filename(self):
        pdb_id, chain_id = parse_structure_id("8u3n_a.npz")
        assert pdb_id == "8U3N"
        assert chain_id == "A"

    def test_handles_multi_char_chain(self):
        pdb_id, chain_id = parse_structure_id("1abc_aa.npz")
        assert pdb_id == "1ABC"
        assert chain_id == "AA"

    def test_rejects_filename_without_underscore(self):
        with pytest.raises(ValueError):
            parse_structure_id("nounderscore.npz")


class TestParseFastaResponse:
    def test_strips_header_and_joins_sequence_lines(self):
        text = ">8U3N.A|Cytochrome P450-SU1|Micromonospora sp.\nMGSSHHHHHH\nSSGLVPRGSH\n"
        assert parse_fasta_response(text) == "MGSSHHHHHHSSGLVPRGSH"

    def test_raises_on_empty_response(self):
        with pytest.raises(ValueError):
            parse_fasta_response("")

    def test_raises_on_response_with_no_sequence_lines(self):
        with pytest.raises(ValueError):
            parse_fasta_response(">8U3N.A|header only\n")


class TestFetchOne:
    def test_returns_sequence_on_200(self):
        session = MagicMock()
        session.get.return_value.status_code = 200
        session.get.return_value.text = ">8U3N.A|desc\nMGSS\n"

        result = fetch_one(session, "8U3N", "A")

        assert result == "MGSS"
        session.get.assert_called_once()
        assert "8U3N.A" in session.get.call_args[0][0]

    def test_raises_on_non_200(self):
        session = MagicMock()
        session.get.return_value.status_code = 404
        session.get.return_value.text = "not found"

        with pytest.raises(RuntimeError):
            fetch_one(session, "8U3N", "A")


class TestLoadExistingIds:
    def test_returns_empty_set_when_file_absent(self, tmp_path):
        assert load_existing_ids(tmp_path / "missing.jsonl") == set()

    def test_returns_structure_ids_from_existing_jsonl(self, tmp_path):
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps({"structure_id": "8u3n_a", "sequence": "MGSS"}) + "\n"
            + json.dumps({"structure_id": "1abc_a", "sequence": "AAAA"}) + "\n"
        )
        assert load_existing_ids(path) == {"8u3n_a", "1abc_a"}
