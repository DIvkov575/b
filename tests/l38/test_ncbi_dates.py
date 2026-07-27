"""Tests for ncbi_dates.py's FASTA-with-accessions parsing and date-cutoff
split logic, plus fetch_createdates' retry-on-failure control flow (mocked
network calls -- this is real logic worth testing directly, unlike a bare
network call, since a real run already crashed once on an unretried
timeout and lost all progress; see fetch_ncbi_dates_full.py)."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.l38.ncbi_dates import fetch_createdates, parse_fasta_with_accessions, split_by_date_cutoff


def test_parse_fasta_with_accessions_extracts_accession_and_sequence(tmp_path):
    fasta = tmp_path / "test.fa"
    fasta.write_text(
        ">NP_536675.1 coat protein [Xanthomonas phage Cf1c]\nMKRL\nRPSD\n"
        ">YP_010115145.1 adsorption tail protein [Enterococcus phage]\nAACC\n"
    )

    records = parse_fasta_with_accessions(fasta)

    assert records == [
        ("NP_536675.1", "MKRLRPSD"),
        ("YP_010115145.1", "AACC"),
    ]


def test_parse_fasta_with_accessions_handles_header_with_no_description(tmp_path):
    fasta = tmp_path / "test.fa"
    fasta.write_text(">NP_536675.1\nMKRL\n")

    records = parse_fasta_with_accessions(fasta)

    assert records == [("NP_536675.1", "MKRL")]


def test_parse_fasta_with_accessions_empty_file(tmp_path):
    fasta = tmp_path / "empty.fa"
    fasta.write_text("")

    assert parse_fasta_with_accessions(fasta) == []


def test_split_by_date_cutoff_separates_by_createdate():
    pairs = [("A", "SEQA"), ("B", "SEQB"), ("C", "SEQC")]
    dates = {"A": "2019/01/01", "B": "2021/06/15", "C": "2020/11/30"}

    train, test, missing = split_by_date_cutoff(pairs, dates, cutoff="2020/12/01")

    assert train == ["SEQA", "SEQC"]
    assert test == ["SEQB"]
    assert missing == []


def test_split_by_date_cutoff_boundary_is_inclusive_on_test_side():
    pairs = [("A", "SEQA")]
    dates = {"A": "2020/12/01"}  # exactly on the cutoff

    train, test, missing = split_by_date_cutoff(pairs, dates, cutoff="2020/12/01")

    assert train == []
    assert test == ["SEQA"]


def test_split_by_date_cutoff_reports_missing_accessions_separately():
    pairs = [("A", "SEQA"), ("B", "SEQB")]
    dates = {"A": "2019/01/01"}  # B has no resolvable date

    train, test, missing = split_by_date_cutoff(pairs, dates, cutoff="2020/12/01")

    assert train == ["SEQA"]
    assert test == []
    assert missing == ["B"]


def test_split_by_date_cutoff_string_comparison_works_for_yyyy_mm_dd_format():
    # sanity check that plain string comparison correctly orders YYYY/MM/DD
    # dates without needing datetime parsing (relies on zero-padding, which
    # NCBI's createdate format guarantees)
    pairs = [("A", "SEQA"), ("B", "SEQB")]
    dates = {"A": "2020/09/05", "B": "2020/11/05"}  # both before cutoff, different months

    train, test, missing = split_by_date_cutoff(pairs, dates, cutoff="2020/12/01")

    assert train == ["SEQA", "SEQB"]


def _mock_response(uids, records):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"result": {"uids": uids, **records}}
    return resp


def test_fetch_createdates_retries_on_timeout_then_succeeds():
    good_response = _mock_response(
        ["1"], {"1": {"accessionversion": "NP_1.1", "createdate": "2020/01/01"}}
    )

    with patch("src.l38.ncbi_dates.requests.get") as mock_get, patch("src.l38.ncbi_dates.time.sleep"):
        mock_get.side_effect = [requests.exceptions.ReadTimeout("boom"), good_response]

        result = fetch_createdates(["NP_1.1"], batch_size=200, max_retries=3)

    assert result == {"NP_1.1": "2020/01/01"}
    assert mock_get.call_count == 2  # first call failed, second succeeded


def test_fetch_createdates_raises_after_exhausting_retries():
    with patch("src.l38.ncbi_dates.requests.get") as mock_get, patch("src.l38.ncbi_dates.time.sleep"):
        mock_get.side_effect = requests.exceptions.ReadTimeout("boom")

        with pytest.raises(requests.exceptions.ReadTimeout):
            fetch_createdates(["NP_1.1"], batch_size=200, max_retries=3)

    assert mock_get.call_count == 3


def test_fetch_createdates_calls_on_progress_after_each_batch():
    resp1 = _mock_response(["1"], {"1": {"accessionversion": "A.1", "createdate": "2020/01/01"}})
    resp2 = _mock_response(["2"], {"2": {"accessionversion": "B.1", "createdate": "2021/01/01"}})

    progress_calls = []

    def on_progress(done, total, dates_so_far):
        progress_calls.append((done, total, dict(dates_so_far)))

    with patch("src.l38.ncbi_dates.requests.get") as mock_get, patch("src.l38.ncbi_dates.time.sleep"):
        mock_get.side_effect = [resp1, resp2]

        fetch_createdates(["A.1", "B.1"], batch_size=1, on_progress=on_progress)

    assert len(progress_calls) == 2
    assert progress_calls[0] == (1, 2, {"A.1": "2020/01/01"})
    assert progress_calls[1] == (2, 2, {"A.1": "2020/01/01", "B.1": "2021/01/01"})
