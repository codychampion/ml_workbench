---
type: paper
title: "Solving a Million-Step LLM Task with Zero Errors"
authors: ["Elliot Meyerson", "Giuseppe Paolo", "Roberto Dailey", "Hormoz Shahrzad", "Olivier Francon", "Conor F. Hayes", "Xin Qiu", "Babak Hodjat", "Risto Miikkulainen"]
year: 2025
venue: "arXiv"
arxiv_id: "2511.09030"
pdf_path: "[[cuad_paper_2511.09030.pdf]]"
url: "https://arxiv.org/abs/2511.09030"
read_date: 2025-12-22
status: read
rating: 5/5

# Categorization
domain: ["multi-agent-systems", "llm-reasoning", "error-correction"]
task: ["long-horizon-reasoning", "multi-step-tasks", "zero-error-execution"]
methods: ["maximal-decomposition", "voting", "error-correction"]
application: ["towers-of-hanoi", "contract-review", "legal-ai"]

# Connections
related_papers: []
related_datasets: ["[[cuad_dataset_card]]"]
experiments: ["[[cuad_maker_plan]]"]
implements: ["MAKER", "MDAPs"]

tags:
  - llm
  - multi-agent
  - error-correction
  - decomposition
  - voting
  - reliability
  - zero-errors
---

# Solving a Million-Step LLM Task with Zero Errors

**Authors:** Meyerson et al. (Cognizant AI Lab & UT Austin)
**Year:** 2025
**ArXiv:** [2511.09030](https://arxiv.org/abs/2511.09030)

## Summary

This paper introduces **MAKER** (Maximal Agentic decomposition, first-to-ahead-by-K Error correction, and Red-flagging), the first system to successfully solve a task with over **1 million LLM steps with zero errors**. The key insight is that instead of building ever-larger models, we should use **Massively Decomposed Agentic Processes (MDAPs)** with extreme task decomposition and error correction.

**Key Achievement:** Solved Towers of Hanoi with 20 disks (1,048,575 steps) with zero errors using GPT-4.1-mini.

## Core Contributions

### 1. Maximal Agentic Decomposition (MAD)

**Problem:** LLMs have persistent error rates that compound exponentially over long tasks.

**Solution:** Break tasks into **minimal subtasks** (m=1 step per agent)

**Benefits:**
- Each agent focuses on a single micro-task
- Reduces context burden
- Enables error correction at every step
- Allows use of smaller, cheaper models

**Formalization:**
```
p(success on full task) = p^s  (without decomposition)
                        vs
p(success per step)^s  (with MAD + error correction)
```

### 2. First-to-Ahead-by-K Voting

**Method:** For each subtask, sample multiple independent solutions and vote

**Voting Rule:** Continue sampling until one candidate is k votes ahead of all others

**Probability of Correct Decision:**
```
p(correct) = 1 / (1 + ((1-p)/p)^k)
```

Where p = base per-step success rate

**Key Insight:** With k=3 and p=0.998, achieves >99.9% per-step reliability

**Scaling Laws:**
- k grows as Θ(ln s) where s = number of steps
- Cost grows as Θ(s ln s) - log-linear scaling!
- Can parallelize k votes for O(s) time complexity

### 3. Red-Flagging

**Idea:** Discard responses showing signs of unreliability

**Red Flags Implemented:**
1. **Length-based:** Response > 750 tokens → agent is over-analyzing
2. **Format-based:** Malformed output → processing error
3. **Confidence-based:** Uncertainty markers → low confidence

**Impact:**
- Reduces error rate from 0.4% to 0.2%
- More importantly: **reduces correlated errors**
- Prevents pathological cases where same error occurs repeatedly

## Experimental Results

### Towers of Hanoi Benchmark

| Model | Cost $/MTok | Per-Step Error | Max Error-Free Steps |
|-------|-------------|----------------|---------------------|
| GPT-4.1-nano | 0.4 | 35.71% | ~3 |
| GPT-4.1-mini | 1.6 | 0.40% | ~250 |
| o3-mini | 4.4 | 0.18% | ~555 |
| **MAKER** | - | **0.00%** | **>1,000,000** |

**Result:** MAKER achieved **zero errors** on 1,048,575 steps (20-disk Hanoi)

**Model Used:** GPT-4.1-mini with k=3, max_tokens=750
**Cost:** Estimated $3.5K (vs. $4.9K for k=4, $9.4K for o3-mini)

### Key Findings

1. **Small models suffice:** GPT-4.1-mini outperforms larger reasoning models on cost-effectiveness
2. **Per-step error rate stable:** Doesn't degrade as problem size increases
3. **Voting converges exponentially:** Sharp decrease in undecided steps after k rounds
4. **Red-flagging crucial:** Reduces correlated errors significantly

## Theoretical Framework

### Scaling Laws

**Without decomposition (m=s):**
```
p(success) = p^s  → exponential decay
```

**With MAD (m=1) and voting:**
```
p_sub = 1 / (1 + ((1-p)/p)^k)
p_full = p_sub^s

k_min = ⌈ln(t^(-1/s) - 1) / ln((1-p)/p)⌉ = Θ(ln s)

E[cost] = Θ(cs ln s) / v(2p-1)
```

Where:
- s = total steps
- p = base per-step success rate
- k = voting threshold
- v = probability of valid response (not red-flagged)
- t = target probability of full success

**Critical Insight:** Cost scales **log-linearly** O(s ln s), not exponentially!

### AALPs Analysis

Paper performs Asymptotic Analysis with LLM Primitives:
- Cost measured in LLM calls
- Shows m=1 (maximal decomposition) is optimal
- Proves voting overhead is logarithmic

## Implementation Details

### Agent Design

**Prompt Structure:**
- System prompt: Overall task description + strategy
- User prompt: Current state + previous move
- Output format: Structured (move + next_state)

**Agent Configuration:**
- Temperature: 0 for first vote, 0.1 for subsequent
- Max tokens: 750 (red-flag threshold)
- Voting: k=3 (determined empirically)

### Parser Design

**Repairing Parser (Section 4.2):**
- Attempts to fix common formatting errors
- Extracts intended answer despite issues

**Red-Flagging Parser (Section 4.4):**
- Strict format enforcement
- Discards malformed responses
- Validates output structure

**Tradeoff:** Red-flagging reduces v but increases p (and reduces correlated errors)

## Applications to Contract Review

See [[cuad_maker_plan]] for detailed application to [[cuad_dataset_card]].

**Key Adaptations:**
1. **Task Decomposition:** 41 clause identification subtasks per contract
2. **Voting Thresholds:**
   - k=2 for structural tasks
   - k=3 for content understanding
   - k=5 for final classification
3. **Red Flags:**
   - Generic legal boilerplate
   - Circular reasoning
   - No specific contract citations
   - Uncertainty markers

**Expected Results:**
- Accuracy: >95% F1 per clause
- Reliability: >90% zero-error contract reviews
- Cost: <$5 per contract

## Key Insights

### 1. Orthogonal Scaling Direction

**Traditional Approach:** Build smarter base LLMs
**MAKER Approach:** Decompose + error correct with existing LLMs

**Result:** MAKER solves tasks that even the smartest LLMs cannot

### 2. Multi-Agent Advantage

Similar to quantum advantage - demonstrates capability **impossible** for single-agent systems

**Key Requirement:** Decomposition granularity must be fine enough that:
- Correct solution is likely to be sampled
- No incorrect solution is more likely than correct

### 3. Error Decorrelation is Critical

**Problem:** If same errors occur repeatedly across agents, voting fails

**Solutions:**
1. Red-flagging pathological responses
2. Temperature > 0 for diversity
3. Prompt paraphrasing (future work)

### 4. Cost-Effectiveness

**Counterintuitive Finding:** Smaller models + voting < larger models alone

**Reasoning:**
- GPT-4.1-mini with k=3: $3.5K
- o3-mini alone: $9.4K (and still makes errors!)
- Parallelization makes wall-clock time O(s) not O(s ln s)

## Limitations & Future Work

### Addressed in Paper

1. **Domain-specific:** Currently requires clear strategy (execution not planning)
2. **Correlated errors:** Need better decorrelation methods
3. **Semantic matching:** Exact matches required; need semantic equivalence

### Future Extensions

1. **Insight + Execution:** Apply to planning/strategy generation
2. **Unknown step counts:** Handle variable-length decompositions
3. **Semantic voting:** Use LLMs to judge equivalence
4. **Cross-domain:** Test on other long-horizon tasks

### Proposed in Paper

- Multiplication experiments (Appendix F) - shows promise
- Prompt paraphrasing for decorrelation
- Different LLMs for different subtask types

## Connections to Other Work

### Related Concepts

- **Microservices:** Similar benefits (modularity, independent scaling)
- **Error correction codes:** Shannon's information theory
- **Quantum error correction:** Scaling quantum computers
- **Biological systems:** DNA repair mechanisms, cancer resistance

### Cited Work

- Towers of Hanoi benchmark (Shojaee et al., 2025)
- Small language models for agents (Belcak et al., 2025)
- Multi-agent LLM systems (Guo et al., 2024)
- LLM execution failures (Sinha et al., 2025)

## Reproducibility

**Code:** Not released (as of paper publication)
**Benchmark:** Towers of Hanoi (publicly available)
**Models:** OpenAI API (GPT-4.1-mini), Anthropic, together.ai

**Our Implementation:** See [[cuad_maker_plan]] for contract review application

## Personal Notes

### Why This Matters

This is a **paradigm shift** in scaling AI:
- Don't just make models smarter → orchestrate them better
- Error correction is possible for linguistic computing
- Opens path to reliable AI for safety-critical applications

### Application to Our Work

Perfect fit for contract review ([[cuad_maker_plan]]):
1. Contracts = long documents requiring zero errors
2. 41 clauses = natural decomposition
3. Legal domain = high reliability requirement
4. Cost-sensitive = smaller models preferred

### Questions/Ideas

- [ ] Can we apply to other legal documents? (patents, regulations)
- [ ] What about multi-modal contracts (PDFs with tables/images)?
- [ ] Can we use multiple model types for different clause types?
- [ ] How to handle ambiguous clauses that lawyers disagree on?

## Citation

```bibtex
@article{meyerson2025maker,
  title={Solving a Million-Step LLM Task with Zero Errors},
  author={Meyerson, Elliot and Paolo, Giuseppe and Dailey, Roberto and
          Shahrzad, Hormoz and Francon, Olivier and Hayes, Conor F and
          Qiu, Xin and Hodjat, Babak and Miikkulainen, Risto},
  journal={arXiv preprint arXiv:2511.09030},
  year={2025}
}
```

## Figures & Key Visuals

- **Figure 1:** Orthogonal scaling - MAKER achieves 1M+ steps vs. <1K for best LLMs
- **Figure 3:** Scaling laws - probability curves for different p values
- **Figure 4:** Cost scaling - log-linear growth O(s ln s)
- **Figure 6:** Per-step error rates - stable across problem sizes
- **Figure 9:** Red-flagging impact - reduces correlated errors

## Related Notes

- [[cuad_dataset_card]] - Dataset for contract review experiment
- [[cuad_maker_plan]] - Our implementation of MAKER for legal contracts
