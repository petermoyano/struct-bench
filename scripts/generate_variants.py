"""Generate a synthetic FASTA file of single-point protein variants.

Uses avGFP (Aequorea victoria green fluorescent protein, 238 residues) as the
base sequence and creates N single-point mutations at random positions with a
fixed seed for reproducibility.
"""

import argparse
import random
from pathlib import Path

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# avGFP (UniProt P42212) — 238 residues
GFP_SEQUENCE = (
    'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL'
    'VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVN'
    'RIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHY'
    'QQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
)


def generate_variants(base_seq: str, n_variants: int, seed: int) -> list[tuple[str, str]]:
    """Return a list of (header, sequence) tuples including wild-type."""
    rng = random.Random(seed)
    variants = [('variant_000_WT', base_seq)]
    used_mutations: set[tuple[int, str]] = set()

    i = 1
    while len(variants) - 1 < n_variants:
        pos = rng.randint(0, len(base_seq) - 1)
        original = base_seq[pos]
        substitutes = [aa for aa in AMINO_ACIDS if aa != original]
        mutant_aa = rng.choice(substitutes)

        if (pos, mutant_aa) in used_mutations:
            continue
        used_mutations.add((pos, mutant_aa))

        mutated = base_seq[:pos] + mutant_aa + base_seq[pos + 1:]
        header = f'variant_{i:03d}_{original}{pos + 1}{mutant_aa}'
        variants.append((header, mutated))
        i += 1

    return variants


def write_fasta(variants: list[tuple[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for header, seq in variants:
            f.write(f'>{header}\n{seq}\n')


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic protein variant FASTA')
    parser.add_argument('--output', type=str, default='data/variants.fasta',
                        help='Output FASTA file path')
    parser.add_argument('--n-variants', type=int, default=50,
                        help='Number of single-point mutations to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    output_path = Path(args.output)
    variants = generate_variants(GFP_SEQUENCE, args.n_variants, args.seed)
    write_fasta(variants, output_path)
    print(f'Wrote {len(variants)} sequences ({1} WT + {len(variants) - 1} mutants) to {output_path}')
    print(f'Base protein: avGFP ({len(GFP_SEQUENCE)} residues)')


if __name__ == '__main__':
    main()
