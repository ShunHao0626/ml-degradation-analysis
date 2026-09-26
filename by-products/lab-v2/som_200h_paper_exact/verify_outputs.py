#!/usr/bin/env python3
"""Verify the generated 200 h analysis and refresh its checksum manifest."""

from run_pipeline import verify_outputs, write_manifest


if __name__ == "__main__":
    report = verify_outputs()
    write_manifest()
    print("PASS" if report["passed"] else "FAIL")
    for name, passed in report["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    raise SystemExit(0 if report["passed"] else 1)
