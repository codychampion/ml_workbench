# MLOps Knowledge Graph

This is an Obsidian-compatible knowledge base for tracking ML research, experiments, papers, and results. All files use YAML frontmatter for structured metadata that Khoj can index and search.

## Directory Structure

```
knowledge/
├── templates/       # Metadata templates (copy these to create new entries)
├── papers/          # Research papers and literature
├── experiments/     # AIM experiment records
├── datasets/        # FiftyOne datasets and data sources
├── models/          # Trained models and checkpoints
├── results/         # Evaluation results and benchmarks
├── projects/        # High-level project plans
└── logs/            # Daily research logs
```

## Entity Relationships

```
┌─────────────┐     references      ┌─────────────┐
│   Papers    │◄───────────────────►│  Projects   │
└──────┬──────┘                     └──────┬──────┘
       │ informs                           │ contains
       ▼                                   ▼
┌─────────────┐     uses            ┌─────────────┐
│ Experiments │────────────────────►│  Datasets   │
└──────┬──────┘                     └─────────────┘
       │ produces                          ▲
       ▼                                   │ evaluated_on
┌─────────────┐     stored_as       ┌─────────────┐
│   Models    │────────────────────►│  Results    │
└─────────────┘                     └─────────────┘
```

## Quick Start

1. Copy a template from `templates/` to the appropriate folder
2. Fill in the YAML frontmatter metadata
3. Add notes, observations, and links to related entities
4. Use `[[wikilinks]]` to connect entities

## Tags Convention

- `#paper` - Research papers
- `#experiment` - ML experiments
- `#dataset` - Datasets
- `#model` - Trained models
- `#result` - Evaluation results
- `#project` - Projects
- `#idea` - Research ideas
- `#todo` - Action items
- `#question` - Open questions

## Khoj Integration

Khoj indexes this vault for AI-powered search. Ask questions like:
- "What papers discuss attention mechanisms?"
- "Show me experiments with accuracy > 90%"
- "What datasets have I used for image classification?"
- "Summarize my research on transformers"
