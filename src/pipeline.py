import yaml
import logging
from pathlib import Path
from src.prepare import (place_targets, generate_contig, format_hotspots,
                         clean_pdb, place_targets_multi, validate_hotspots)
from src.generate import (build_rfdiffusion_cmd, run_rfdiffusion,
                          parse_output_pdbs, generate_mock_designs)
from src.filter import analyze_design, filter_designs_by_contacts
from src.design import build_proteinmpnn_cmd, run_proteinmpnn, pdb_to_fasta, write_fasta
from src.evaluate import check_ternary_clash, evaluate_design, build_af2_command
from src.results import write_results_csv, write_summary, rank_designs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(config_path: str = "configs/targets.yaml",
                 defaults_path: str = "configs/defaults.yaml",
                 dry_run: bool = False,
                 mock: bool = False,
                 skip_af2: bool = False) -> list:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(defaults_path) as f:
        defaults = yaml.safe_load(f)

    all_results = []

    for pair in config["pairs"]:
        logger.info(f"=== Processing: {pair['name']} ===")

        # Stage 0: Validate hotspots
        valid_a = validate_hotspots(pair["target_a"]["pdb"], pair["target_a"]["chain"],
                                    pair["target_a"]["hotspots"])
        valid_b = validate_hotspots(pair["target_b"]["pdb"], pair["target_b"]["chain"],
                                    pair["target_b"]["hotspots"])
        if len(valid_a) < 3 or len(valid_b) < 3:
            logger.warning(f"Too few valid hotspots: A={len(valid_a)}, B={len(valid_b)}. Skipping pair.")
            continue

        # Stage 1: Place targets
        placement = place_targets(
            pdb_a=pair["target_a"]["pdb"],
            pdb_b=pair["target_b"]["pdb"],
            separation=pair["separation"]
        )
        logger.info(f"Targets placed at {pair['separation']}Å separation")

        contig = generate_contig(
            chain_a_length=placement["chain_a_length"],
            chain_b_length=placement["chain_b_length"],
            binder_min=pair["binder_length"][0],
            binder_max=pair["binder_length"][1]
        )
        hotspots = format_hotspots(valid_a, valid_b)
        logger.info(f"Contig: {contig}")
        logger.info(f"Hotspots: {hotspots}")

        # Stage 2: Generate backbones
        output_dir = f"outputs/{pair['name']}/designs"
        num_designs = defaults["generation"]["num_designs"]

        if mock:
            n = min(num_designs, 20)
            logger.info(f"MOCK MODE: generating {n} random designs")
            design_pdbs = generate_mock_designs(
                output_dir, placement["combined_pdb"], num_designs=n
            )
        elif dry_run:
            cmd = build_rfdiffusion_cmd(
                input_pdb=placement["combined_pdb"],
                contig=contig,
                hotspots=hotspots,
                output_dir=output_dir,
                num_designs=num_designs,
                diffusion_steps=defaults["generation"]["diffusion_steps"]
            )
            logger.info(f"DRY RUN command:\n  {cmd}")
            return []
        else:
            cmd = build_rfdiffusion_cmd(
                input_pdb=placement["combined_pdb"],
                contig=contig,
                hotspots=hotspots,
                output_dir=output_dir,
                num_designs=num_designs,
                diffusion_steps=defaults["generation"]["diffusion_steps"]
            )
            run_rfdiffusion(cmd)
            design_pdbs = parse_output_pdbs(output_dir)

        logger.info(f"Generated {len(design_pdbs)} designs")

        # Stage 3: Filter by contacts
        min_contacts = defaults["filtering"]["min_contacts_per_target"]
        contact_threshold = defaults["filtering"]["contact_distance_threshold"]

        contact_results = []
        for pdb in design_pdbs:
            try:
                result = analyze_design(
                    pdb, placement["combined_pdb"],
                    chain_a_id="A", chain_b_id="B",
                    contact_threshold=contact_threshold
                )
                contact_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to analyze {pdb}: {e}")

        passed_contact = filter_designs_by_contacts(contact_results, min_contacts=min_contacts)
        contact_rate = len(passed_contact) / max(len(contact_results), 1) * 100
        logger.info(f"Contact filter: {len(passed_contact)}/{len(contact_results)} pass "
                    f"(>={min_contacts} contacts per target) [{contact_rate:.1f}%]")

        # Stage 4: Sequence design (ProteinMPNN) — skipped in mock if no real backbones
        mpnn_dir = f"outputs/{pair['name']}/sequences"
        if passed_contact and not mock:
            for design in passed_contact:
                cmd = build_proteinmpnn_cmd(
                    input_pdb=design["path"],
                    output_dir=mpnn_dir,
                    chains_to_design="C",
                    num_sequences=4
                )
                run_proteinmpnn(cmd, dry_run=dry_run)
            logger.info(f"ProteinMPNN: designed sequences for {len(passed_contact)} backbones")
        elif passed_contact and mock:
            logger.info(f"MOCK: skipping ProteinMPNN (no real backbones)")

        # Stage 5: AF2-Multimer scoring — skipped if --skip-af2 or mock
        if passed_contact and not skip_af2 and not mock:
            af2_dir = f"outputs/{pair['name']}/af2"
            Path(af2_dir).mkdir(parents=True, exist_ok=True)
            for design in passed_contact:
                binder_seq = pdb_to_fasta(design["path"], "C")
                # Score vs target A
                fasta_a = f"{af2_dir}/{Path(design['path']).stem}_vs_A.fasta"
                write_fasta({"binder": binder_seq,
                             "target_A": pdb_to_fasta(pair["target_a"]["pdb"],
                                                      pair["target_a"]["chain"])},
                            fasta_a)
                # Score vs target B
                fasta_b = f"{af2_dir}/{Path(design['path']).stem}_vs_B.fasta"
                write_fasta({"binder": binder_seq,
                             "target_B": pdb_to_fasta(pair["target_b"]["pdb"],
                                                      pair["target_b"]["chain"])},
                            fasta_b)
                cmd_a = build_af2_command(fasta_a, f"{af2_dir}/pred_A")
                cmd_b = build_af2_command(fasta_b, f"{af2_dir}/pred_B")
                logger.info(f"AF2 commands prepared for {Path(design['path']).stem}")
            logger.info(f"AF2: prepared {len(passed_contact) * 2} predictions")
        elif not skip_af2 and not mock and not passed_contact:
            logger.info("No designs passed contact filter — skipping AF2")

        # Stage 6: Steric clash check (placeholder — needs AF2 predicted structures)
        # In real mode, we'd load AF2 predicted complexes and check if targets clash.
        # For now, compute clash between target placements (static check).
        import numpy as np
        from src.utils import get_ca_coords, load_structure
        target_struct = load_structure(placement["combined_pdb"])
        model = target_struct[0]
        coords_a = np.array([r["CA"].get_vector().get_array()
                             for r in model["A"] if "CA" in r])
        coords_b = np.array([r["CA"].get_vector().get_array()
                             for r in model["B"] if "CA" in r])
        static_clashes = check_ternary_clash(coords_a, coords_b, clash_dist=2.0)
        logger.info(f"Static target clash check: {static_clashes} clashes "
                    f"({'PASS' if static_clashes <= 5 else 'FAIL'})")

        # Stage 7: Results aggregation
        for r in passed_contact:
            r["passes"] = True  # passed contact filter; AF2/clash TBD in real mode

        all_results.extend(contact_results)

        # Write outputs
        results_dir = f"outputs/{pair['name']}"
        write_results_csv(contact_results, f"{results_dir}/results.csv")
        write_summary(contact_results, f"{results_dir}/summary.json")

        if passed_contact:
            ranked = rank_designs(passed_contact)
            logger.info(f"Top designs:")
            for d in ranked[:5]:
                logger.info(f"  {Path(d['path']).name}: "
                            f"contacts_A={d['contacts_a']} contacts_B={d['contacts_b']} "
                            f"score={d['composite_score']:.1f}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-target binder design pipeline")
    parser.add_argument("--mock", action="store_true", help="Mock generation (no GPU)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--skip-af2", action="store_true", help="Skip AF2 evaluation")
    parser.add_argument("--config", default="configs/targets.yaml")
    parser.add_argument("--defaults", default="configs/defaults.yaml")
    args = parser.parse_args()
    results = run_pipeline(
        config_path=args.config,
        defaults_path=args.defaults,
        dry_run=args.dry_run,
        mock=args.mock,
        skip_af2=args.skip_af2
    )
    total_pass = sum(1 for r in results if r.get("passes", False))
    print(f"\nTotal designs: {len(results)}, Passing: {total_pass}")
