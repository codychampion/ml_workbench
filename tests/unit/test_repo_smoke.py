from pathlib import Path


def test_core_workbench_files_are_present():
    root = Path(__file__).resolve().parents[2]
    expected = [
        "docker-compose.yml",
        "docker-compose.gpu.yml",
        ".env.example",
        "README.md",
        "README_LLM.md",
        "requirements.txt",
    ]

    missing = [path for path in expected if not (root / path).exists()]
    assert missing == []


def test_compose_profiles_are_documented():
    readme = Path(__file__).resolve().parents[2] / "README.md"
    text = readme.read_text(encoding="utf-8")

    for profile in ["llm", "jupyter", "api", "tracking", "pipeline"]:
        assert f"`{profile}`" in text
