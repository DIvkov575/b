import csv
import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def write_results_csv(results: list, output_path: str) -> str:
    """Write all design results to CSV."""
    if not results:
        logger.warning("No results to write")
        return output_path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["design_id", "path", "contacts_a", "contacts_b",
                  "n_residues", "pae_a", "pae_b", "n_clashes",
                  "passes_contact", "passes_af2", "passes_clash", "passes_all"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i, r in enumerate(results):
            row = {
                "design_id": i,
                "path": r.get("path", ""),
                "contacts_a": r.get("contacts_a", ""),
                "contacts_b": r.get("contacts_b", ""),
                "n_residues": r.get("n_residues", ""),
                "pae_a": r.get("pae_a", ""),
                "pae_b": r.get("pae_b", ""),
                "n_clashes": r.get("n_clashes", ""),
                "passes_contact": r.get("contacts_a", 0) >= 5 and r.get("contacts_b", 0) >= 5,
                "passes_af2": r.get("pae_a", 99) < 10 and r.get("pae_b", 99) < 10,
                "passes_clash": r.get("n_clashes", 99) <= 5,
                "passes_all": r.get("passes", False),
            }
            writer.writerow(row)

    logger.info(f"Results written to {output_path}")
    return output_path


def write_summary(results: list, output_path: str) -> str:
    """Write a summary JSON with statistics."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total = len(results)
    pass_contact = sum(1 for r in results
                       if r.get("contacts_a", 0) >= 5 and r.get("contacts_b", 0) >= 5)
    pass_all = sum(1 for r in results if r.get("passes", False))

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_designs": total,
        "pass_contact_filter": pass_contact,
        "pass_all_filters": pass_all,
        "hit_rate_contact": pass_contact / max(total, 1),
        "hit_rate_final": pass_all / max(total, 1),
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary written to {output_path}")
    return output_path


def rank_designs(results: list) -> list:
    """Rank designs by composite score (lower pae = better, more contacts = better)."""
    scored = []
    for r in results:
        pae_a = r.get("pae_a", 99)
        pae_b = r.get("pae_b", 99)
        contacts = r.get("contacts_a", 0) + r.get("contacts_b", 0)
        # Lower score = better. Penalize high PAE, reward contacts.
        score = (pae_a + pae_b) - 0.1 * contacts
        scored.append({**r, "composite_score": score})
    return sorted(scored, key=lambda x: x["composite_score"])
