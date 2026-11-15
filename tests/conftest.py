"""Mutation testing hook: with SWASTHYA_MUTANT=<name> set, one guard mutant is applied before any test module is
imported (see src/swasthya/mutation.py and scripts/mutation_test.py). Without the variable nothing changes."""
import os

_name = os.environ.get("SWASTHYA_MUTANT")
if _name:
    from swasthya.mutation import MUTANTS, apply

    if _name not in MUTANTS:
        raise SystemExit(f"unknown mutant {_name!r}; known: {sorted(MUTANTS)}")
    apply(_name)
