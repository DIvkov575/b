import yaml
import logging
from pathlib import Path
from src.prepare import place_targets, generate_contig, format_hotspots
from src.generate import (build_rfdiffusion_cmd, run_rfdiffusion,
                          parse_output_pdbs, generate_mock_designs)
from src.filter import analyze_design, filter_designs_by_contacts

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(config_path: str = "configs/targets.yaml",
                 defaults_path: str = "configs/defaults.yaml",
                 dry_run: bool = False,
                 mock: bool = False) -> list:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(defaults_path) as f:
        defaults = yaml.safe_load(f)

    all_passed = []

    for pair in config["pairs"]:
        logger.info(f"=== Processing: {pair['name']} ===")

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
        hotspots = format_hotspots(
            pair["target_a"]["hotspots"],
            pair["target_b"]["hotspots"]
        )
        logger.info(f"Contig: {contig}")
        logger.info(f"Hotspots: {hotspots}")

        # Stage 2: Generate
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

        results = []
        for pdb in design_pdbs:
            try:
                result = analyze_design(
                    pdb, placement["combined_pdb"],
                    chain_a_id="A", chain_b_id="B",
                    contact_threshold=contact_threshold
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to analyze {pdb}: {e}")

        passed = filter_designs_by_contacts(results, min_contacts=min_contacts)

        # Stage 4: Report
        hit_rate = len(passed) / max(len(results), 1) * 100
        logger.info(f"Contact filter: {len(passed)}/{len(results)} pass "
                    f"(>={min_contacts} contacts per target at {contact_threshold}Å)")
        logger.info(f"Hit rate: {hit_rate:.1f}%")

        for d in passed[:10]:
            logger.info(f"  PASS: {Path(d['path']).name} "
                        f"contacts_A={d['contacts_a']} contacts_B={d['contacts_b']}")

        all_passed.extend(passed)

    return all_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-target binder design pipeline")
    parser.add_argument("--mock", action="store_true", help="Mock generation (no GPU)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--config", default="configs/targets.yaml")
    parser.add_argument("--defaults", default="configs/defaults.yaml")
    args = parser.parse_args()
    results = run_pipeline(
        config_path=args.config,
        defaults_path=args.defaults,
        dry_run=args.dry_run,
        mock=args.mock
    )
    print(f"\nTotal designs passing: {len(results)}")
