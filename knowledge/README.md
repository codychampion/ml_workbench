# Knowledge Vault

Obsidian-compatible knowledge base. Open this folder in Obsidian.

## Structure

```
knowledge/
├── topics/          # General topic notes
├── papers/
│   ├── pdfs/        # PDF storage (gitignored)
│   └── notes/       # Paper notes
├── datasets/
│   ├── cards/       # Dataset cards
│   └── manifests/   # Dataset manifests (json)
├── experiments/
│   ├── plans/       # Experiment plans
│   └── runs/        # Run summaries (auto-generated)
├── models/registry/ # Model registry notes
└── templates/       # Note templates
```

## Khoj Integration

Start Khoj to chat with this vault:
```bash
docker compose --profile chat up -d
# Open http://localhost:42110
```

Khoj indexes: notes, code, outputs.
