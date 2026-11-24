"""Tests for isolating manual macrobenchmark runs from BigQuery ingestion.

Manual runs (default ``_RUN_SOURCE=manual``) must upload under a ``manual/``
prefix that neither ``macrobenchmarks_schema.json`` nor ``ingest.sql`` ingests,
even when executed on ``BRANCH_NAME=main``. Only ``_RUN_SOURCE=scheduled``
uploads to the ingested ``branch=<branch>/...`` prefix.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[4]
MACRO_DIR = REPO / "cloudbuild/macrobenchmarks"
CLOUDBUILD = yaml.safe_load((MACRO_DIR / "macrobenchmarks-cloudbuild.yaml").read_text())
STEPS = {step["id"]: step for step in CLOUDBUILD["steps"]}

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _stub(path: Path, body: str) -> None:
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755)


def _run_init_variables(tmp_path: Path, **overrides):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _stub(
        bin_dir / "gcloud",
        'case "$*" in\n'
        '  "storage buckets describe"*"value(location)"*) echo "US-CENTRAL1" ;;\n'
        '  "storage buckets describe"*) echo "{}" ;;\n'
        "esac\n"
        "exit 0",
    )
    build_vars = tmp_path / "build_vars.env"
    values = {
        **CLOUDBUILD["substitutions"],
        "PROJECT_ID": "test-project",
        "BUILD_ID": "12345678-aaaa-bbbb-cccc-1234567890ab",
        "BRANCH_NAME": "main",
        "_INFRA_PREFIX": "pfx",
        "_ZONE": "us-central1-a",
        "_GKE_SERVICE_ACCOUNT": "sa@example.com",
        "_DATASET_PATH": "gs://src-bucket/parquet",
        "_REQUIREMENTS": "gcsfs",
        **overrides,
    }
    step = STEPS["init-variables"]
    run_env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "WORKSPACE_DIR": str(tmp_path),
        "FAILED_FILE": str(tmp_path / "FAILED"),
        "BUILD_VARS_FILE": str(build_vars),
    }
    for entry in step["env"]:
        name, template = entry.split("=", 1)
        m = re.fullmatch(r"\$\{([A-Z0-9_]+)\}", template)
        assert m, f"unexpected env template: {entry}"
        run_env[name] = str(values.get(m.group(1), ""))

    proc = subprocess.run(
        ["bash", step["args"][0]],
        cwd=REPO,
        env=run_env,
        capture_output=True,
        text=True,
    )
    exports = (
        dict(re.findall(r"^export (\w+)=(.*)$", build_vars.read_text(), re.M))
        if build_vars.exists()
        else {}
    )
    return proc, exports


def test_cloudbuild_declares_and_wires_run_source_default_manual():
    assert CLOUDBUILD["substitutions"]["_RUN_SOURCE"] == "manual"
    assert "_RUN_SOURCE=${_RUN_SOURCE}" in STEPS["init-variables"]["env"]


@pytest.mark.parametrize(
    "override, expected",
    [
        ({}, "manual"),
        ({"_RUN_SOURCE": ""}, "manual"),
        ({"_RUN_SOURCE": "manual"}, "manual"),
        ({"_RUN_SOURCE": "scheduled"}, "scheduled"),
    ],
)
def test_init_variables_exports_run_source(tmp_path, override, expected):
    proc, exports = _run_init_variables(tmp_path, **override)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert exports["RUN_SOURCE"] == expected


def test_init_variables_rejects_unknown_run_source(tmp_path):
    proc, _ = _run_init_variables(tmp_path, _RUN_SOURCE="bogus")
    assert proc.returncode != 0
    assert "_RUN_SOURCE" in proc.stdout + proc.stderr


@pytest.mark.parametrize(
    "run_source, expected_prefix",
    [
        ("scheduled", ""),
        ("manual", "manual/"),
        (None, "manual/"),
    ],
)
def test_scrape_metrics_routes_manual_runs_under_manual_prefix(
    tmp_path, run_source, expected_prefix
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir / "python3", "exit 0")
    _stub(bin_dir / "sleep", "exit 0")

    (tmp_path / "start_time.txt").write_text("2026-11-24T00:00:00Z")
    (tmp_path / "end_time.txt").write_text("2026-11-24T01:00:00Z")
    build_vars = tmp_path / "build_vars.env"
    run_source_line = f"export RUN_SOURCE={run_source}\n" if run_source else ""
    build_vars.write_text(
        "export BRANCH_NAME=main\n"
        "export RUN_ID=buildid-12345678\n"
        "export CLUSTER_NAME=pfx-gke-12345678\n"
        "export CHECKPOINT_BUCKET=pfx-macrobench-checkpoint-12345678\n"
        "export DATASET_BUCKET=pfx-macrobench-dataset-12345678\n"
        "export RESULTS_BUCKET=pfx-macrobench-results\n"
        "export REGION=us-central1\n"
        f"{run_source_line}"
    )

    step = STEPS["scrape-metrics"]
    values = {
        **CLOUDBUILD["substitutions"],
        "PROJECT_ID": "test-project",
        "_ZONE": "us-central1-a",
        "_DATASET_PATH": "gs://src-bucket/parquet",
        "_REQUIREMENTS": "gcsfs",
    }
    run_env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "WORKSPACE_DIR": str(tmp_path),
        "FAILED_FILE": str(tmp_path / "FAILED"),
        "BUILD_VARS_FILE": str(build_vars),
    }
    for entry in step["env"]:
        name, template = entry.split("=", 1)
        m = re.fullmatch(r"\$\{([A-Z0-9_]+)\}", template)
        assert m, f"unexpected env template: {entry}"
        run_env[name] = str(values.get(m.group(1), ""))

    proc = subprocess.run(
        ["bash", step["args"][0]],
        cwd=REPO,
        env=run_env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    exports = dict(re.findall(r"^export (\w+)=(.*)$", build_vars.read_text(), re.M))
    summary_uri = exports["SUMMARY_URI"]
    assert re.fullmatch(
        rf"gs://pfx-macrobench-results/{expected_prefix}branch=main/\d{{8}}/buildid-12345678/\d{{8}}-\d{{6}}\.csv",
        summary_uri,
    )


def test_macro_schema_and_ingest_sql_exclude_manual_prefix():
    schema = json.loads((MACRO_DIR / "macrobenchmarks_schema.json").read_text())
    (uri_template,) = schema["sourceUris"]
    assert uri_template == "gs://@INFRA_PREFIX@-macrobench-results/branch=*.csv"
    assert uri_template.count("*") == 1

    # BigQuery single-* wildcard matches across path separators.
    glob_re = re.compile(
        "^"
        + re.escape(uri_template.replace("@INFRA_PREFIX@", "pfx")).replace(r"\*", ".*")
        + "$"
    )
    scheduled_uri = "gs://pfx-macrobench-results/branch=main/20261124/buildid-12345678/20261124-010203.csv"
    manual_uri = "gs://pfx-macrobench-results/manual/branch=main/20261124/buildid-12345678/20261124-010203.csv"
    branch_named_manual_uri = "gs://pfx-macrobench-results/branch=manual/20261124/buildid-12345678/20261124-010203.csv"
    assert glob_re.match(scheduled_uri)
    assert not glob_re.match(manual_uri)
    assert glob_re.match(branch_named_manual_uri)

    ingest_sql = (MACRO_DIR / "ingest.sql").read_text()
    pattern_match = re.search(
        r"NOT REGEXP_CONTAINS\(s\._FILE_NAME,\s*r'([^']+)'\)", ingest_sql
    )
    assert pattern_match, "ingest.sql must exclude manual/ files as defense-in-depth"
    sql_re = re.compile(pattern_match.group(1))
    assert sql_re.search(manual_uri)
    assert not sql_re.search(scheduled_uri)
    assert not sql_re.search(branch_named_manual_uri)
