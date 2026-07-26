"""
scripts/junit_to_summary.py

Turns pytest's --junitxml output into a short GitHub-flavored markdown
summary, printed to stdout. Piped into $GITHUB_STEP_SUMMARY in
.github/workflows/daily-tests.yml so pass/fail counts and any failing
test names show up directly on the Actions run page -- no separate
dashboard needed for the "what happened today" question.

Usage:
    python scripts/junit_to_summary.py reports/junit.xml
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET


def main(junit_path: str) -> None:
    tree = ET.parse(junit_path)
    root = tree.getroot()

    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]

    total = failures = errors = skipped = 0
    time = 0.0
    failed_names = []

    for suite in suites:
        total += int(suite.get("tests", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))
        time += float(suite.get("time", 0.0))

        for case in suite.findall("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                classname = case.get("classname", "")
                name = case.get("name", "")
                failed_names.append(f"{classname}::{name}")

    passed = total - failures - errors - skipped
    status = "PASS" if (failures == 0 and errors == 0) else "FAIL"

    print(f"## Test Report -- {status}\n")
    print("| Metric | Count |")
    print("|---|---|")
    print(f"| Total | {total} |")
    print(f"| Passed | {passed} |")
    print(f"| Failed | {failures} |")
    print(f"| Errors | {errors} |")
    print(f"| Skipped | {skipped} |")
    print(f"| Duration | {time:.1f}s |")

    if failed_names:
        print("\n### Failed tests\n")
        for name in failed_names:
            print(f"- `{name}`")

    print("\nFull Allure report (with trend history) is published to GitHub Pages once this run finishes.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/junit_to_summary.py <path-to-junit.xml>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])