---
type: experiment-plan
exp_id: "cuad-maker-001"
created: "2025-12-22"
status: planned
domain: legal-ai
task: contract-review
method: MAKER

# Connections
dataset: "[[cuad_dataset_card]]"
paper: "[[maker_paper]]"
related_experiments: []

tags:
  - maker
  - multi-agent
  - legal
  - contracts
  - zero-errors
  - decomposition
  - voting
---

# MAKER-Based Contract Review Experiment Design

**Dataset:** [[cuad_dataset_card|CUAD]] (Contract Understanding Atticus Dataset)
**Paper:** [[maker_paper|"Solving a Million-Step LLM Task with Zero Errors"]] (arXiv:2511.09030)
**Date:** 2025-12-22

## Executive Summary

This experiment applies the **MAKER framework** (Maximal Agentic decomposition, first-to-ahead-by-K Error correction, and Red-flagging) from [[maker_paper]] to automated contract review using the [[cuad_dataset_card|CUAD dataset]]. The goal is to achieve high-precision identification of 41 clause categories across commercial legal contracts by decomposing the review process into minimal subtasks and applying error correction at each step.

## Background

### CUAD Dataset
- **Size:** 84,325 samples across 510 commercial legal contracts
- **Task:** Identify 41 important clause categories
- **Categories:** Include both binary classification (33 categories) and entity extraction (8 categories)
- **Challenge:** Long documents (up to 6.97k characters), complex legal language, overlapping clause contexts

### MAKER Framework Key Principles

1. **Maximal Agentic Decomposition (MAD):**
   - Break tasks into the smallest possible subtasks
   - Each agent focuses on a single micro-task
   - Reduces context burden and improves reliability

2. **First-to-ahead-by-K Voting:**
   - Independent sampling from multiple agents
   - Vote until one candidate is k votes ahead
   - Provides error correction at each step

3. **Red-Flagging:**
   - Discard responses showing signs of unreliability
   - Flags: overly long responses, formatting errors, uncertainty markers
   - Reduces correlated errors

## Experimental Design

### Phase 1: Single-Clause Classification (Baseline)

**Objective:** Establish baseline accuracy for clause identification

**Method:**
- Decompose each contract into 41 separate clause identification tasks
- For each clause category, create a focused micro-agent with the prompt:
  ```
  "Given this contract text, does it contain a [CLAUSE_CATEGORY] clause?
   Answer YES or NO with a brief explanation."
  ```

**Decomposition Level:** `m = 1` (one clause category per agent call)

**Agent Configuration:**
- Base Model: GPT-4.1-mini (based on paper's cost-effectiveness analysis)
- Temperature: 0 for first vote, 0.1 for subsequent votes
- Max tokens: 500 (red-flag threshold based on paper's findings)
- Voting threshold: k = 3 (start conservatively)

**Metrics:**
- Per-step success rate (p)
- Per-clause-category accuracy
- False positive/negative rates
- Cost per contract review

### Phase 2: Multi-Level Hierarchical Decomposition

**Objective:** Apply hierarchical decomposition for complex contracts

**Decomposition Strategy:**

```
Level 1: Contract Segmentation
├── Agent: Split contract into logical sections (intro, obligations, termination, etc.)
│
Level 2: Section Analysis
├── Agent: For each section, identify potentially relevant clause categories
│
Level 3: Clause Verification
├── Agent: For each identified category, verify presence with evidence extraction
│
Level 4: Conflict Resolution
└── Agent: Resolve any conflicting classifications via voting
```

**Agent Roles:**
1. **Segmentation Agent:** Breaks contract into manageable sections
2. **Relevance Agent:** Identifies which clauses might be present in each section
3. **Verification Agent:** Confirms clause presence with textual evidence
4. **Conflict Resolution Agent:** Resolves disagreements between multiple verifications

**Voting at Each Level:**
- Level 1: k = 2 (low risk, structural task)
- Level 2: k = 3 (moderate risk, content understanding)
- Level 3: k = 5 (high risk, final classification)
- Level 4: k = 3 (tiebreaker scenarios only)

### Phase 3: Error Correction via Voting

**Objective:** Determine optimal k value for contract review task

**Experimental Matrix:**

| k value | Expected Pr[Correct] | Expected Cost Multiplier |
|---------|---------------------|-------------------------|
| 1       | baseline            | 1x                      |
| 2       | baseline²           | ~2-3x                   |
| 3       | baseline³           | ~3-5x                   |
| 5       | baseline⁵           | ~5-8x                   |
| 7       | baseline⁷           | ~7-11x                  |

**Process:**
1. Estimate p (per-step success rate) on a sample of 100 contracts
2. Use Equation 18 from paper to project cost: E[cost] = Θ(s ln s)
3. Select k that achieves target accuracy (95%+) at acceptable cost
4. Validate on hold-out set

### Phase 4: Red-Flagging Implementation

**Red Flag Criteria:**

1. **Length-based flags:**
   - Response > 750 tokens → discard and resample
   - Indicates agent confusion or over-analysis

2. **Format-based flags:**
   - Missing required answer format (YES/NO)
   - Malformed JSON output
   - Indicates processing errors

3. **Confidence-based flags:**
   - Uncertainty markers ("maybe", "possibly", "unclear")
   - Hedging language suggests low confidence
   - Multiple contradictory statements

4. **Legal-specific flags:**
   - Generic legal boilerplate without specifics
   - Circular reasoning
   - Failure to cite specific contract text

**Implementation:**
```python
def is_red_flagged(response: str, max_tokens: int = 750) -> bool:
    # Length check
    if len(response.split()) > max_tokens:
        return True

    # Format check
    if not has_required_format(response):
        return True

    # Confidence check
    uncertainty_markers = ["maybe", "possibly", "unclear", "might", "could be"]
    if any(marker in response.lower() for marker in uncertainty_markers):
        return True

    # Legal-specific check
    if is_generic_boilerplate(response):
        return True

    return False
```

### Phase 5: Scaling Laws Validation

**Objective:** Validate paper's scaling laws for contract review domain

**Hypothesis:**
- Cost scales as Θ(s ln s) where s = number of clause identification steps
- Per-step error rate remains stable as contract length increases
- Voting converges exponentially as predicted by Equation 9

**Experiments:**
1. Vary contract length (short: <2k chars, medium: 2-5k chars, long: >5k chars)
2. Measure per-step error rate across contract lengths
3. Compare actual cost to predicted cost from Equation 18
4. Validate exponential convergence of voting process

## Success Metrics

### Primary Metrics

1. **Accuracy:** F1 score for each of 41 clause categories (target: >95%)
2. **Reliability:** Zero-error rate across full contract reviews (target: >90%)
3. **Cost-Effectiveness:** Cost per contract vs. baseline single-agent approach
4. **Scalability:** Maintain accuracy as contract length increases

### Secondary Metrics

1. **Per-step Success Rate (p):** Measure base agent reliability
2. **Voting Efficiency:** Average votes required per decision
3. **Red-Flag Rate:** Percentage of responses flagged and resampled
4. **Error Correlation:** Measure decorrelation of errors across agents

## Implementation Plan

### Step 1: Data Preparation
```bash
# Download CUAD dataset
python -m pipelines.collect.collect_cuad --split train --limit 1000

# Prepare for MAKER experiment
python -m pipelines.collect.collect_cuad --prepare-maker
```

### Step 2: Baseline Estimation
```python
# Estimate per-step success rate on sample
from pipelines.evaluate.cuad_maker import estimate_success_rate

p = estimate_success_rate(
    model="gpt-4.1-mini",
    dataset="cuad",
    sample_size=100,
    clause_categories=["Anti-Assignment", "Audit Rights", ...],
)

print(f"Base success rate: {p:.4f}")
```

### Step 3: Optimize k Value
```python
# Project costs for different k values
from pipelines.evaluate.cuad_maker import project_costs

s = 41  # number of clause categories per contract
target_accuracy = 0.95

optimal_k, projected_cost = find_optimal_k(
    p=p,
    s=s,
    target_accuracy=target_accuracy,
)

print(f"Optimal k: {optimal_k}, Projected cost: ${projected_cost:.2f}")
```

### Step 4: Run Full Experiment
```python
# Execute MAKER-based contract review
from pipelines.evaluate.cuad_maker import run_maker_experiment

results = run_maker_experiment(
    dataset="cuad",
    split="test",
    k=optimal_k,
    red_flagging=True,
    max_tokens=750,
)

print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Zero-error rate: {results['zero_error_rate']:.2%}")
print(f"Total cost: ${results['total_cost']:.2f}")
```

## Key Insights from Paper Applied to Contract Review

### 1. Extreme Decomposition Benefits

**Paper Finding:** Breaking tasks into minimal subtasks (m=1) enables efficient error correction

**Contract Review Application:**
- Instead of "review entire contract," decompose into 41 separate clause identification tasks
- Each micro-agent focuses on identifying a single clause type
- Reduces cognitive load and improves focus

### 2. Small Models Suffice

**Paper Finding:** Non-reasoning models like GPT-4.1-mini achieve comparable per-step accuracy to larger reasoning models

**Contract Review Application:**
- Use GPT-4.1-mini for cost-effectiveness
- Each clause identification is a focused task suitable for smaller models
- Reserve larger models only if per-step accuracy is insufficient

### 3. Red-Flagging Reduces Correlated Errors

**Paper Finding:** Overly long responses and format errors correlate with incorrect reasoning

**Contract Review Application:**
- Flag responses > 750 tokens (agent is over-analyzing)
- Flag responses with uncertainty markers (agent is confused)
- Flag generic responses without specific contract citations

### 4. Voting Converges Exponentially

**Paper Finding:** With sufficient k, error rate drops exponentially (Equation 9)

**Contract Review Application:**
- Start with k=3 for most clause categories
- Increase k=5 for critical clauses (e.g., liability, termination)
- Expect exponential reduction in errors with modest k values

### 5. Cost Scales Log-Linearly

**Paper Finding:** Expected cost grows as Θ(s ln s) with extreme decomposition

**Contract Review Application:**
- For 41 clauses per contract, expect ~2-3x cost increase vs. single-agent
- Parallelizable: 41 clause checks can run concurrently
- Amortized cost: Share contract preprocessing across all clause checks

## Expected Outcomes

### Quantitative Goals

1. **Accuracy:** >95% F1 score on all 41 clause categories
2. **Reliability:** >90% of contracts reviewed with zero errors
3. **Cost:** <$5 per contract review (vs. ~$50-100 for human review)
4. **Scalability:** Maintain accuracy on contracts up to 10k characters

### Qualitative Goals

1. **Explainability:** Each clause identification backed by specific textual evidence
2. **Audibility:** Full trace of agent decisions for review and validation
3. **Modularity:** Easy to add new clause categories or modify existing ones
4. **Confidence Scores:** Provide uncertainty estimates for each classification

## Risk Mitigation

### Potential Challenges

1. **Challenge:** Legal domain complexity may require higher k values
   - **Mitigation:** Start with conservative k=5, adjust based on validation results

2. **Challenge:** Long contracts may exceed context windows
   - **Mitigation:** Implement sliding window approach with overlap

3. **Challenge:** Ambiguous clause boundaries
   - **Mitigation:** Add "Uncertain" category, use higher k for ambiguous cases

4. **Challenge:** Dataset imbalance (some clauses rare)
   - **Mitigation:** Stratified sampling, overweight rare categories in evaluation

## Future Extensions

1. **Multi-Contract Analysis:** Apply MAKER to comparative contract analysis
2. **Entity Extraction:** Extend to extract specific dates, amounts, parties
3. **Clause Generation:** Use decomposition for contract drafting assistance
4. **Risk Scoring:** Aggregate clause presence into overall contract risk scores
5. **Cross-Domain Transfer:** Adapt framework to other document types (patents, regulations)

## References

- Paper: Meyerson et al., "Solving a Million-Step LLM Task with Zero Errors" (arXiv:2511.09030)
- Dataset: CUAD v1 (https://huggingface.co/datasets/theatticusproject/cuad)
- Original CUAD Paper: Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review"

## Conclusion

The MAKER framework provides a principled approach to achieving high-precision contract review through:
1. **Extreme decomposition** into focused clause identification tasks
2. **Error correction** via multi-agent voting at each step
3. **Red-flagging** to filter unreliable responses

This approach addresses the fundamental challenge in AI-assisted legal review: achieving near-perfect accuracy on long, complex documents. By decomposing contracts into minimal subtasks and applying systematic error correction, we can scale LLM-based review to production-grade reliability.

The expected outcome is a contract review system that:
- Achieves >95% accuracy across all clause categories
- Reviews contracts with zero errors >90% of the time
- Costs <$5 per contract (vs. $50-100 for human review)
- Maintains reliability as document complexity increases

This represents a significant step toward trustworthy AI-assisted legal document analysis.
