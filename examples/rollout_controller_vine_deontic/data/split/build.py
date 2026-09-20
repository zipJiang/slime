"""Freeze task/difficulty-stratified case-family splits without success filtering.

Build from the audited expanded-evaluation source manifest. The resulting bundle
includes raw cases, statutes, unique question lists, and explicit sampling weights.
The sample subcommand emits a balanced root-question schedule for future runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DOMAINS = ("airline", "sara_numeric", "sara_binary")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def family_map(catalog):
    """Join identical normalized facts globally and explicit binary siblings."""
    parent = {key: key for key in catalog}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    seen = {}
    for key, row in sorted(catalog.items()):
        domain, case_id = key.split("/", 1)
        signatures = [("facts", " ".join(row["facts"].casefold().split()))]
        if domain == "sara_binary":
            signatures.append((domain, re.sub(r"_(pos|neg)$", "", case_id)))
        for signature in signatures:
            if not signature[1]:
                raise ValueError(f"Empty facts: {key}")
            if signature in seen:
                a, b = sorted((find(key), find(seen[signature])))
                parent[b] = a
            else:
                seen[signature] = key
    return {key: find(key) for key in catalog}


def choose_test(catalog, hard, families, locked, fraction, seed):
    """Stratify family counts; minimize case-count imbalance using labels only.

    Binary families can span hard/normal and contain many cases. Search 2048
    deterministic candidate assignments with fixed family quotas per stratum.
    No model predictions or rewards enter selection.
    """
    groups = defaultdict(list)
    for key, family in families.items():
        groups[family].append(key)
    chosen = set()
    for domain in DOMAINS:
        domain_groups = {
            f: sorted(keys)
            for f, keys in groups.items()
            if keys[0].startswith(domain + "/")
        }
        strata = defaultdict(list)
        for family, keys in sorted(domain_groups.items()):
            if any(k.split("/", 1)[0] != domain for k in keys):
                raise ValueError("Cross-domain family requires joint stratification")
            complexity = ""
            if domain == "airline":
                complexity = re.search(r"complexity_(\d+)_", keys[0]).group(1)
            strata[bool(set(keys) & hard), complexity].append(family)
        target = round(len(domain_groups) * fraction)
        quotas = {s: math.floor(len(fs) * fraction) for s, fs in strata.items()}
        order = sorted(strata, key=lambda s: (-(len(strata[s]) * fraction % 1), s))
        for s in order[: target - sum(quotas.values())]:
            quotas[s] += 1
        eligible = {s: sorted(set(fs) - locked) for s, fs in strata.items()}
        if any(len(eligible[s]) < n for s, n in quotas.items()):
            raise ValueError(f"Cannot meet test quota with locked families: {domain}")

        def counts(fs, domain_groups=domain_groups, domain=domain):
            result = Counter()
            for f in fs:
                for k in domain_groups[f]:
                    result["hard" if k in hard else "normal"] += 1
                    if domain == "sara_binary":
                        result[catalog[k]["answer"]] += 1
            return result

        totals = counts(domain_groups)
        expected = {k: v * fraction for k, v in totals.items()}
        contributions = {f: counts([f]) for f in domain_groups}
        best = None
        for trial in range(2048):
            rng = random.Random(f"{seed}/{domain}/{trial}")
            candidate = set()
            for s in sorted(strata):
                candidate.update(rng.sample(eligible[s], quotas[s]))
            actual = Counter()
            for f in sorted(candidate):
                actual.update(contributions[f])
            error = sum(((actual[k] - n) / max(1, n)) ** 2 for k, n in expected.items())
            rank = (error, sorted(candidate))
            if best is None or rank < best:
                best = rank
            if error == 0:
                break
        chosen.update(best[1])
    return chosen


def weighted_questions(keys, hard, families):
    """Equal task/difficulty mass, then equal families, then equal sibling cases."""
    groups = defaultdict(lambda: defaultdict(list))
    for key in sorted(keys):
        groups[key.split("/", 1)[0], key in hard][families[key]].append(key)
    if len(groups) != 6:
        raise ValueError("All six task/difficulty cells must be nonempty")
    rows = []
    for (domain, is_hard), fs in sorted(groups.items()):
        for family, members in sorted(fs.items()):
            rows.extend(
                {
                    "prompt": [{"role": "user", "content": key}],
                    "metadata": {
                        "case_key": key,
                        "case_id": key.split("/", 1)[1],
                        "domain": domain,
                        "hard": is_hard,
                        "family": family,
                        "sampling_weight": 1 / (6 * len(fs) * len(members)),
                    },
                }
                for key in members
            )
    return sorted(rows, key=lambda r: r["metadata"]["case_key"])


def balanced_schedule(rows, draws, seed):
    """Six draws per block, one per cell; sample within cells with replacement."""
    if draws <= 0 or draws % 6:
        raise ValueError("Draw count must be a positive multiple of six")
    groups = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=lambda r: r["metadata"]["case_key"]):
        m = row["metadata"]
        groups[m["domain"], m["hard"]][m["family"]].append(row)
    if len(groups) != 6:
        raise ValueError("Expected all six task/difficulty cells")
    rng = random.Random(seed)
    result = []
    for _ in range(draws // 6):
        cells = sorted(groups)
        rng.shuffle(cells)
        for cell in cells:
            fs = groups[cell]
            result.append(rng.choice(fs[rng.choice(sorted(fs))]))
    return result


def build(source, output, fraction, seed):
    if output.exists():
        raise ValueError("Frozen output exists; choose a new version directory")
    if not 0 < fraction < 1:
        raise ValueError("Test fraction must be between zero and one")
    manifest = json.loads(source.read_text())
    catalog, origins, hard = {}, {}, set()
    for record in manifest["sources"]:
        path = Path(record["path"])
        if digest(path) != record["sha256"]:
            raise ValueError(f"Source changed: {path}")
        domain = path.parent.name
        for row in read_rows(path):
            key = domain + "/" + row["id"]
            normalized = {
                k: " ".join(v.split()) if isinstance(v, str) else v
                for k, v in row.items()
            }
            if key in catalog:
                prior = {
                    k: " ".join(v.split()) if isinstance(v, str) else v
                    for k, v in catalog[key].items()
                }
                if normalized != prior:
                    raise ValueError(f"Conflicting case: {key}")
            else:
                catalog[key], origins[key] = row, path.parent
            if path.stem == "hard":
                hard.add(key)
    families = family_map(catalog)
    trained = set()
    for record in manifest["training_sources"]:
        path = Path(record["path"])
        if digest(path) != record["sha256"]:
            raise ValueError(f"Training source changed: {path}")
        for row in read_rows(path):
            m = row["metadata"]
            trained.add(
                m["domain"] + "/" + m["case_id"] if "case_id" in m else m["family"]
            )
    # Keep earlier model-selection questions in train, too. The new test was
    # previously evaluated in the expanded sweep; it is not an untouched test.
    old_val = set(manifest["previous_validation_cases"])
    locked = {families[k] for k in trained | old_val}
    test_families = choose_test(catalog, hard, families, locked, fraction, seed)
    splits = {
        "test": {k for k, f in families.items() if f in test_families},
        "train": {k for k, f in families.items() if f not in test_families},
    }
    assert not splits["train"] & splits["test"]
    assert splits["train"] | splits["test"] == set(catalog)
    assert not test_families & locked
    output.mkdir(parents=True)
    (output / "source-manifest.json").write_bytes(source.read_bytes())
    (output / "build.py").write_bytes(Path(__file__).read_bytes())
    statistics = {}
    for split, keys in splits.items():
        questions = weighted_questions(keys, hard, families)
        write_rows(output / f"{split}.jsonl", questions)
        (output / f"{split}-case-keys.json").write_text(
            json.dumps(sorted(keys), indent=2) + "\n"
        )
        statistics[split] = {
            "cases": len(keys),
            "families": len({families[k] for k in keys}),
            "domains": {
                d: {
                    "cases": sum(k.startswith(d + "/") for k in keys),
                    "hard": sum(k.startswith(d + "/") and k in hard for k in keys),
                    "families": len(
                        {families[k] for k in keys if k.startswith(d + "/")}
                    ),
                }
                for d in DOMAINS
            },
        }
        for domain in DOMAINS:
            directory = output / "corpus" / domain
            directory.mkdir(parents=True, exist_ok=True)
            write_rows(
                directory / f"{split}.jsonl",
                [catalog[k] for k in sorted(keys) if k.startswith(domain + "/")],
            )
    statute_sources = {}
    for key, row in sorted(catalog.items()):
        domain = key.split("/", 1)[0]
        for statute in row["statutes"]:
            src = origins[key] / statute
            dst = output / "corpus" / domain / statute
            if not dst.resolve().is_relative_to((output / "corpus").resolve()):
                raise ValueError("Statute path leaves corpus")
            if dst.exists() and digest(dst) != digest(src):
                raise ValueError(f"Conflicting statute: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            statute_sources[str(src)] = digest(src)
    write_rows(
        output / "inventory.jsonl",
        [
            {
                "case_key": k,
                "family": families[k],
                "hard": k in hard,
                "previously_trained": k in trained,
                "previous_validation": k in old_val,
                "split": "test" if k in splits["test"] else "train",
            }
            for k in sorted(catalog)
        ],
    )
    report = {
        "format": "deontic_balanced_family_split_v1",
        "seed": seed,
        "test_fraction": fraction,
        "statistics": statistics,
        "locked_training_families": sorted(locked),
        "selection": (
            "Task/difficulty family quotas; airline complexity; binary label counts. "
            "No model outcomes."
        ),
        "sampling": (
            "Uniform task, hard/normal 50/50, uniform family within cell, "
            "uniform case within family/cell."
        ),
        "test_status": (
            "Previously evaluated cases; disjoint from historical training "
            "and 13-case validation families."
        ),
        "source_manifest_sha256": digest(source),
        "statute_sources": statute_sources,
        "outputs": {
            str(p.relative_to(output)): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(statistics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--source-manifest", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--test-fraction", type=float, default=0.5)
    create.add_argument("--seed", type=int, default=20260912)
    sample = commands.add_parser("sample")
    sample.add_argument("--split-dir", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--draws", type=int, required=True)
    sample.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.source_manifest, args.output, args.test_fraction, args.seed)
    else:
        path = args.split_dir / "train.jsonl"
        manifest = json.loads((args.split_dir / "manifest.json").read_text())
        if digest(path) != manifest["outputs"]["train.jsonl"]:
            raise ValueError("Training list differs from the frozen manifest")
        if args.output.exists():
            raise ValueError("Schedule output already exists")
        write_rows(
            args.output, balanced_schedule(read_rows(path), args.draws, args.seed)
        )


if __name__ == "__main__":
    main()
