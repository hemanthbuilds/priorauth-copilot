"""Policy lookup: which policy covers a procedure, exact diagnosis-code checks, and loud failure on a broken setup."""

import subprocess
import sys

from app import config, policy_store


def test_policy_lookup():
    assert policy_store.get_policy("72148").policy_key == "lumbar_mri"
    assert policy_store.get_policy("CPT:27447").policy_key == "total_knee"
    assert policy_store.get_policy("99999") is None and policy_store.get_policy(None) is None
    assert set(policy_store.procedures()) == {"72148", "27447", "95810"}


def test_code_check():
    assert policy_store.code_check(["m54.16"], "72148") == "M54.16 is on the accepted list for CPT 72148."
    assert "M54.50 is NOT on the accepted list for CPT 95810" in policy_store.code_check(["M54.50"], "95810")
    both = policy_store.code_check(["G47.33", "M54.50"], "95810")
    assert "G47.33 is on" in both and "M54.50 is NOT" in both
    assert policy_store.code_check([], "72148") is None and policy_store.code_check(["M54.16"], None) is None


def test_missing_policy_file_fails_loudly(tmp_path):
    (tmp_path / "policies").mkdir()
    (tmp_path / "procedures.yaml").write_text(
        'procedures:\n  - cpt: "1"\n    name: x\n    policy_key: x\n    title: x\n    policy_id: X1\n'
        '    files: [missing.pdf]\n    diagnoses: {}\n')
    code = "import app.policy_store as p; p.main()"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=config.ROOT,
                          env={"DATA_DIR": str(tmp_path), "PATH": ""})
    assert proc.returncode == 1 and "missing.pdf" in proc.stderr
