---
type: project
name: "{{project_name}}"
status: active  # planning, active, paused, completed, abandoned
priority: high  # low, medium, high, critical

# Timeline
start_date: {{date}}
target_date:
completed_date:
last_updated: {{date}}

# Goals
primary_goal: "{{goal}}"
success_criteria:
  -
metrics_target:
  accuracy:
  latency_ms:

# Scope
domain: "{{domain}}"  # cv, nlp, ml, etc.
task: "{{task}}"
approach: "{{approach}}"

# Resources
estimated_hours:
actual_hours:
compute_budget:
gpu_hours_used:

# Team
owner: "{{owner}}"
collaborators: []

# Connections
related_papers: []
experiments: []
datasets: []
models: []
results: []
parent_project: ""
sub_projects: []

tags:
  - project
  - {{status}}
  - {{domain}}
---

# Project: {{project_name}}

## Overview
<!-- Brief description of the project -->


## Motivation
<!-- Why is this project important? What problem does it solve? -->


## Goals & Success Criteria

### Primary Goal
{{goal}}

### Success Metrics
| Metric | Target | Current | Status |
|--------|--------|---------|--------|
|        |        |         | 🔴/🟡/🟢 |

### Non-Goals
<!-- What is explicitly out of scope -->
-

## Background & Related Work

### Key Papers
- [[]]

### Prior Art
-

### Our Approach
<!-- How does our approach differ? -->


## Technical Approach

### Architecture
<!-- High-level architecture diagram or description -->


### Key Components
1.
2.
3.

### Data Pipeline
<!-- Describe data flow -->


### Training Strategy
<!-- Describe training approach -->


## Milestones

### Phase 1: {{phase1_name}}
- [ ] Task 1
- [ ] Task 2
- **Target:** {{date}}

### Phase 2: {{phase2_name}}
- [ ] Task 1
- [ ] Task 2
- **Target:** {{date}}

## Progress Log

### {{date}}
-

## Experiments

| Experiment | Status | Key Result | Link |
|------------|--------|------------|------|
|            |        |            | [[]] |

## Datasets Used
- [[]]

## Models Produced
- [[]]

## Key Results
- [[]]

## Open Questions
- [ ]

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
|      |            |        |            |

## Lessons Learned
<!-- Fill in as project progresses -->
-

## Next Steps
- [ ]

## References
-

