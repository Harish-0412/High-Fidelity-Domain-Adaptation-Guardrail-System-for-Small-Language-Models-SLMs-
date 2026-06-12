import os

replacements = {
    "api": "api",
    "ingestion": "ingestion",
    "retrieval": "retrieval",
    "services.critic": "services.critic",
    "services.training": "services.training",
    "services.evaluation": "services.evaluation",
    "services.core": "services.core",
    "services": "services" # Fallback if any plain imports exist
}

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)

    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

for root, _, files in os.walk("."):
    if ".git" in root or ".venv" in root or ".pytest_cache" in root or "__pycache__" in root:
        continue
    for file in files:
        if file.endswith(".py") or file.endswith(".md"):
            process_file(os.path.join(root, file))
