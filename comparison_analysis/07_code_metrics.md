# Code Metrics Analysis

**Quantitative comparison of codebase size, complexity, and quality**

---

## 1. Lines of Code (LOC)

### Kayba ace-framework

**Total Core LOC:** ~6,185 lines

**Breakdown by Module:**

| File | LOC | Purpose |
|------|-----|---------|
| `prompts_v2_1.py` | 1,586 | State-of-the-art prompt templates (v2.1) |
| `prompts_v2.py` | 1,023 | Enhanced prompt templates (v2.0 - DEPRECATED) |
| `roles.py` | 713 | Generator, Reflector, Curator implementations |
| `llm_providers/litellm_client.py` | 712 | LiteLLM integration (100+ providers) |
| `adaptation.py` | 582 | OfflineAdapter, OnlineAdapter, SimpleEnvironment |
| `observability/opik_integration.py` | 348 | Opik tracing and cost tracking |
| `playbook.py` | 283 | Playbook and Bullet data structures |
| `llm_providers/langchain_client.py` | 230 | LangChain integration |
| `llm.py` | 174 | LLMClient interface, DummyLLMClient, TransformersLLMClient |
| Other files | ~534 | Delta, features, tracers, __init__ |

**LOC Distribution:**
- **Prompts (v2.1 + v2.0):** 2,609 lines (42%)
- **LLM Integration:** 1,116 lines (18%) - LiteLLM + LangChain
- **Core ACE Logic:** 1,578 lines (26%) - Roles + Adaptation + Playbook
- **Observability:** 348 lines (6%)
- **Infrastructure:** 534 lines (8%)

### OpenCE

**Total Core LOC:** ~2,195 lines (ACE implementation + interfaces + core)

**Breakdown by Module:**

| File | LOC | Purpose |
|------|-----|---------|
| `methods/ace/roles.py` | 270 | Generator, Reflector, Curator (ACE-specific) |
| `methods/ace/playbook.py` | 245 | Playbook data structures |
| `methods/ace/adaptation.py` | 223 | OfflineAdapter, OnlineAdapter |
| `models/clients.py` | 210 | LLMClient interface, DeepseekLLMClient, TransformersLLMClient |
| `models/providers.py` | 136 | BaseModelProvider, OpenAIModelProvider, etc. |
| `core/orchestrator.py` | 90 | ClosedLoopOrchestrator (5-pillar pipeline) |
| `methods/ace/prompts.py` | 89 | ACE prompt templates |
| `methods/base.py` | 78 | BaseMethod, MethodRegistry |
| `components/evolvers/ace_curator.py` | 76 | ACE Curator wrapper (IEvolver) |
| `components/evaluators/ace_reflector.py` | 75 | ACE Reflector wrapper (IEvaluator) |
| `methods/ace/delta.py` | 67 | Delta operations |
| `methods/ace/deduplication.py` | 64 | Semantic deduplication |
| `methods/ace_closed_loop.py` | 59 | ACE closed-loop method |
| `interfaces/data_models.py` | 56 | Canonical data models |
| Other interfaces/components | ~457 | Acquirer, Processor, Constructor interfaces + implementations |

**LOC Distribution:**
- **Core ACE Logic:** 1,001 lines (46%) - ACE methods only
- **Interfaces:** 457 lines (21%) - 5-pillar architecture
- **Orchestrator:** 90 lines (4%)
- **LLM Integration:** 346 lines (16%) - Clients + Providers
- **Components:** 151 lines (7%) - Wrappers for ACE in 5-pillar system
- **Methods:** 137 lines (6%) - Method registry + ACE closed loop

### Comparison

| Metric | Kayba | OpenCE |
|--------|-------|--------|
| **Total LOC** | 6,185 | 2,195 |
| **Core ACE Logic** | 1,578 (26%) | 1,001 (46%) |
| **Prompts** | 2,609 (42%) | 89 (4%) |
| **LLM Integration** | 1,116 (18%) | 346 (16%) |
| **Observability** | 348 (6%) | 0 (0%) |
| **Framework/Interfaces** | 0 (0%) | 457 (21%) |

**Key Insights:**
- Kayba is **2.8x larger** than OpenCE
- Kayba's size driven by **prompt versions** (42% of codebase)
- OpenCE's architecture is **more modular** (46% core ACE vs 26%)
- OpenCE invests **21% in abstractions**, Kayba invests **6% in observability**

---

## 2. File Count

### Kayba

**Total Python Files:** 16

```
ace/
├── __init__.py
├── adaptation.py
├── delta.py
├── features.py
├── llm.py
├── playbook.py
├── prompts.py
├── prompts_v2.py
├── prompts_v2_1.py
├── roles.py
├── llm_providers/
│   ├── __init__.py
│   ├── langchain_client.py
│   └── litellm_client.py
└── observability/
    ├── __init__.py
    ├── opik_integration.py
    └── tracers.py
```

**Modularity:** Low (16 files, some large files like `prompts_v2_1.py` = 1,586 LOC)

### OpenCE

**Total Python Files:** 38

```
src/opence/
├── interfaces/ (6 files)
│   ├── __init__.py
│   ├── acquisition.py
│   ├── construction.py
│   ├── data_models.py
│   ├── evaluation.py
│   ├── evolution.py
│   └── processing.py
├── components/ (9 files)
│   ├── acquirers/
│   ├── constructors/
│   ├── evaluators/
│   ├── evolvers/
│   └── processors/
├── methods/ (8 files)
│   ├── ace/ (6 files)
│   ├── ace_closed_loop.py
│   └── base.py
├── models/ (4 files)
├── adapters/ (2 files)
└── core/ (2 files)
```

**Modularity:** High (38 files, average ~60 LOC per file)

**Verdict:**
- Kayba favors **fewer, larger files** (easier navigation, harder to extend)
- OpenCE favors **many small files** (easier to extend, more imports)

---

## 3. Complexity Metrics

### Cyclomatic Complexity (Estimated)

**Kayba:**

| Function | Complexity | Notes |
|----------|------------|-------|
| `Generator.generate()` | ~8 | Retry loop + JSON parsing + markdown stripping |
| `Reflector.reflect()` | ~12 | Multi-round refinement + early exit logic |
| `Curator.curate()` | ~6 | Retry loop + delta parsing |
| `OfflineAdapter.run()` | ~10 | Nested epoch/sample loops + checkpointing |
| `_safe_json_loads()` | ~7 | Markdown stripping + truncation detection |

**Average:** ~8.6 (moderate complexity)

**OpenCE:**

| Function | Complexity | Notes |
|----------|------------|-------|
| `Generator.generate()` | ~6 | Retry loop + JSON parsing |
| `Reflector.reflect()` | ~10 | Multi-round refinement |
| `Curator.curate()` | ~5 | Retry loop + delta parsing |
| `OfflineAdapter.run()` | ~8 | Nested loops + deduplication |
| `_safe_json_loads()` | ~3 | Basic JSON parsing |
| `ClosedLoopOrchestrator.run()` | ~4 | Linear pipeline execution |

**Average:** ~6.0 (low-moderate complexity)

**Verdict:** OpenCE has **lower complexity** (simpler functions, more modular).

### Nesting Depth

**Kayba:**
- Max nesting: 4 levels (OfflineAdapter with checkpointing logic)
- Average: 2-3 levels

**OpenCE:**
- Max nesting: 3 levels
- Average: 2 levels

**Verdict:** OpenCE is **flatter** (better readability).

---

## 4. Code Duplication

### Kayba

**Duplicate Code:**
- Retry logic (Generator, Reflector, Curator): ~30 lines duplicated 3x
- JSON parsing (Shared via `_safe_json_loads()`, but retry messages differ)
- Observability decorators (`@maybe_track` used consistently)

**Duplication Score:** ~5% (low)

### OpenCE

**Duplicate Code:**
- Retry logic (Generator, Reflector, Curator): ~25 lines duplicated 3x
- JSON parsing (Shared via `_safe_json_loads()`)
- Interface boilerplate (5 interfaces with similar structure)

**Duplication Score:** ~8% (low-moderate)

**Verdict:** Both have **low duplication**, Kayba slightly better.

---

## 5. Type Hint Coverage

### Kayba

**Coverage:** ~95%

**Examples:**
```python
def generate(
    self,
    *,
    question: str,
    context: Optional[str],
    playbook: Playbook,
    reflection: Optional[str] = None,
    **kwargs: Any,
) -> GeneratorOutput:
```

**Missing Types:**
- Some `**kwargs` not fully typed
- Return types in `__init__` methods sometimes omitted

### OpenCE

**Coverage:** ~98%

**Examples:**
```python
def generate(
    self,
    *,
    question: str,
    context: Optional[str],
    playbook: Playbook,
    reflection: Optional[str] = None,
    **kwargs: Any,
) -> GeneratorOutput:
```

**Missing Types:**
- Minimal gaps

**Verdict:** Both have **excellent type hint coverage**, OpenCE slightly better.

---

## 6. Docstring Coverage

### Kayba

**Coverage:** ~60%

**Well-Documented:**
- Public API methods (Generator, Reflector, Curator)
- Playbook operations
- LLM clients

**Poorly Documented:**
- Internal helper functions
- Some private methods

**Example (Good):**
```python
class Generator:
    """
    Produces answers using the current playbook of strategies.

    The Generator is one of three core ACE roles. It takes a question and
    uses the accumulated strategies in the playbook to produce reasoned answers.

    Args:
        llm: The LLM client to use for generation
        prompt_template: Custom prompt template (uses GENERATOR_PROMPT by default)
        max_retries: Maximum attempts if JSON parsing fails (default: 3)
        retry_prompt: Additional instruction appended on retry

    Example:
        >>> from ace import Generator, LiteLLMClient, Playbook
        >>> client = LiteLLMClient(model="gpt-3.5-turbo")
        >>> generator = Generator(client)
        >>> result = generator.generate(...)
    """
```

### OpenCE

**Coverage:** ~30%

**Well-Documented:**
- Interfaces (abstract methods have docstrings)
- Some data models

**Poorly Documented:**
- Most concrete implementations
- Internal functions
- No usage examples in docstrings

**Example (Typical):**
```python
class Generator:
    """Produces trajectories using the current playbook."""

    def __init__(self, llm: LLMClient, prompt_template: str = GENERATOR_PROMPT, *, max_retries: int = 3) -> None:
        # No docstring
```

**Verdict:** Kayba has **2x better docstring coverage** and higher quality (includes examples).

---

## 7. Test Metrics

### Kayba

**Test Files:** 6

**Test Count:** ~40 tests

**Coverage Areas:**
- Playbook CRUD: 8 tests
- Delta operations: 6 tests
- Roles (Generator/Reflector/Curator): 10 tests
- Adaptation loops: 8 tests
- LLM clients: 5 tests
- Integration tests: 10+ tests

**Test LOC:** ~1,200 lines

**Test-to-Code Ratio:** 1:5.1 (1 line test per 5.1 lines code)

### OpenCE

**Test Files:** 6

**Test Count:** ~30 tests

**Coverage Areas:**
- ACE components: 8 tests
- Adaptation: 6 tests
- Deduplication: 5 tests
- Methods: 4 tests
- Orchestrator: 7 tests

**Test LOC:** ~800 lines

**Test-to-Code Ratio:** 1:2.7 (1 line test per 2.7 lines code)

**Verdict:**
- Kayba has **more tests** (40 vs 30)
- OpenCE has **better test-to-code ratio** (1:2.7 vs 1:5.1)
- Both have good coverage

---

## 8. Dependency Count

### Kayba

**Core Dependencies:** 5
```
langchain-openai, litellm, pydantic, python-dotenv, tenacity
```

**Optional Dependencies:** 15+
```
[observability]: opik
[demos]: rich, datasets, pyyaml, browser-use, pandas, openpyxl, playwright
[langchain]: langchain-litellm
[transformers]: transformers, torch, accelerate
[dev]: pytest, black, mypy, pre-commit, git-changelog
```

**Total Dependency Tree:** ~150 packages (with all extras)

**Install Size:** ~500MB (with all extras)

### OpenCE

**Core Dependencies:** 2
```
python-dotenv, pydantic
```

**Optional Dependencies:** 10+
```
[ace]: sentence-transformers, scikit-learn
[local-llm]: transformers, torch, accelerate
[cli]: typer
[api]: openai
[rwkv]: rwkv
[dev]: pytest, ruff
```

**Total Dependency Tree:** ~50 packages (with all extras)

**Install Size:** ~200MB (with all extras)

**Verdict:**
- Kayba has **3x more dependencies** (batteries-included approach)
- OpenCE has **minimal core** (users add what they need)

---

## 9. Code Quality Metrics Summary

| Metric | Kayba | OpenCE | Winner |
|--------|-------|--------|--------|
| **Total LOC** | 6,185 | 2,195 | OpenCE (leaner) |
| **File Count** | 16 | 38 | Tie (different philosophies) |
| **Avg LOC/File** | 387 | 58 | OpenCE (more modular) |
| **Cyclomatic Complexity** | ~8.6 | ~6.0 | OpenCE (simpler) |
| **Code Duplication** | ~5% | ~8% | Kayba (less duplication) |
| **Type Hint Coverage** | 95% | 98% | OpenCE |
| **Docstring Coverage** | 60% | 30% | Kayba (2x better) |
| **Test Count** | 40 | 30 | Kayba |
| **Test-to-Code Ratio** | 1:5.1 | 1:2.7 | OpenCE (better ratio) |
| **Core Dependencies** | 5 | 2 | OpenCE (lighter) |
| **Total Dependencies** | ~150 | ~50 | OpenCE (3x fewer) |

---

## 10. Code Maintainability

### Kayba

**Strengths:**
- ✅ Better docstrings (60% coverage with examples)
- ✅ Fewer files to navigate (16 vs 38)
- ✅ Less code duplication (5%)

**Weaknesses:**
- ❌ Larger files (avg 387 LOC/file)
- ❌ Higher complexity (avg ~8.6)
- ❌ More dependencies to manage (~150)

**Maintainability Score:** 7/10

### OpenCE

**Strengths:**
- ✅ More modular (avg 58 LOC/file)
- ✅ Lower complexity (avg ~6.0)
- ✅ Better test coverage ratio (1:2.7)
- ✅ Fewer dependencies (50 vs 150)
- ✅ Higher type hint coverage (98%)

**Weaknesses:**
- ❌ Poor docstrings (30% coverage, no examples)
- ❌ More files to navigate (38 vs 16)

**Maintainability Score:** 8/10

**Verdict:** OpenCE is **more maintainable** (cleaner architecture, better tests), but Kayba has **better documentation**.

---

## 11. Code Evolution

### Kayba: Rapid Feature Addition

**Version History:**
- v0.1.0: Core ACE (2,000 LOC)
- v0.2.0: +1,000 LOC (LiteLLM integration)
- v0.3.0: +500 LOC (Opik observability)
- v0.4.0: +2,685 LOC (Prompt versioning v2.1, checkpointing)

**Growth Rate:** +110% LOC over 4 months

**Growth Pattern:** Additive (new features added, old code kept for compatibility)

### OpenCE: Architectural Refactoring

**Version History:**
- v0.1.0: Refactor from flat ACE to 5-pillar architecture

**Growth Rate:** N/A (single release)

**Growth Pattern:** Refactoring (reorganized existing code into interfaces)

---

## 12. Technical Debt

### Kayba

**Potential Debt:**
- ⚠️ Large prompt files (1,586 LOC for v2.1)
- ⚠️ Retry logic duplicated in 3 roles
- ⚠️ Hardcoded Chinese prompts in OpenCE code (if they copied base)

**Mitigation:**
- ✅ Prompt files are mostly templates (low cognitive load)
- ✅ Retry messages now configurable (v0.4.1)
- ✅ Good test coverage prevents regressions

**Technical Debt Score:** 3/10 (low debt)

### OpenCE

**Potential Debt:**
- ⚠️ Hardcoded Chinese retry messages
- ⚠️ Interface boilerplate (5 interfaces, some overlap)
- ⚠️ No observability (hard to add later without breaking changes)

**Mitigation:**
- ✅ Clean abstractions reduce coupling
- ✅ Small files easy to refactor

**Technical Debt Score:** 4/10 (low-moderate debt)

---

## Bottom Line

**Quantitative Winner: OpenCE**
- Leaner codebase (2,195 vs 6,185 LOC)
- More modular (58 vs 387 LOC/file)
- Lower complexity (6.0 vs 8.6)
- Better test ratio (1:2.7 vs 1:5.1)

**Qualitative Winner: Kayba**
- Better documentation (60% vs 30%)
- More features (observability, checkpointing)
- Production-ready

**For maintainability:** OpenCE's architecture ages better
**For usability:** Kayba's documentation makes it easier to learn

**Ideal setup:** OpenCE's architecture + Kayba's documentation = perfect codebase.
