# ACE Framework - Implementation TODOs

**Source:** OpenCE comparison analysis (see `comparison_analysis/` folder)
**Date:** November 16, 2025
**Status:** Pending prioritization

These are features from OpenCE that were evaluated for alignment with our production-first philosophy.

---

## ❌ REJECTED: Simple Context Enhancers

**Priority:** Was HIGH
**Status:** REJECTED - Out of scope

### Why We're NOT Adding This

**Original idea:** Add RAG retrieval, compression, etc. as "context enhancers" in the ACE framework.

**Why it's out of scope:**
1. **ACE's job:** Learn strategies from experience (Generator → Reflector → Curator → Playbook)
2. **RAG's job:** Retrieve external knowledge from vector stores
3. **These are separate concerns** - Users should own RAG integration
4. **Already works:** Users can pass RAG results via the `context` parameter

**The right architecture:**
```
┌─────────────────────────────────────┐
│  User's Application Layer           │
│  - RAG retrieval (LangChain, etc.)  │
│  - Document processing              │
│  - Context assembly                 │
└──────────────┬──────────────────────┘
               │ Pass as context string
               ↓
┌─────────────────────────────────────┐
│  ACE Framework (This Repo)          │
│  - Generator (uses context+playbook)│
│  - Reflector (learns from mistakes) │
│  - Curator (updates playbook)       │
└─────────────────────────────────────┘
```

**Users who need RAG + ACE:**
```python
# User owns RAG integration (outside ACE)
from langchain.vectorstores import FAISS
from ace import Generator, Playbook

vectorstore = FAISS.from_texts(documents, embeddings)
playbook = Playbook()
generator = Generator(llm)

def my_rag_ace_pipeline(question):
    # 1. User's RAG logic
    docs = vectorstore.similarity_search(question, k=5)
    context = "\n".join([doc.page_content for doc in docs])

    # 2. ACE generation with learned strategies
    result = generator.generate(
        question=question,
        context=context,  # RAG results here
        playbook=playbook  # Learned strategies here
    )
    return result
```

**Conclusion:** Keep ACE focused on learning from experience. RAG integration is user's responsibility.

---

## ✅ COMPLETED: Lite Distribution 📦

**Priority:** Was MEDIUM
**Status:** COMPLETED - Clean dependency structure implemented

### Final Structure

**Core Dependencies (~100MB):**
```bash
pip install ace-framework
```
Includes:
- `litellm` - Universal LLM client (100+ providers) - **Required for Reflector/Curator**
- `pydantic` - Data validation
- `python-dotenv` - Environment config
- `tenacity` - Retry logic

**Optional Dependencies:**
```bash
# Enterprise LangChain integration
pip install ace-framework[langchain]

# Production monitoring
pip install ace-framework[observability]

# Local model support
pip install ace-framework[transformers]

# Everything combined (~500MB)
pip install ace-framework[all]

# Mix and match
pip install ace-framework[observability,langchain]
```

**Developer Dependencies (NOT distributed to PyPI):**
- Moved to `[dependency-groups]` per PEP 735
- Automatically installed for contributors via `uv sync`
- Includes: pytest, black, mypy, pre-commit, git-changelog
- NOT included when users `pip install ace-framework`

### Key Decisions

**✅ LiteLLM stays in core** because:
- Reflector and Curator ALWAYS need an LLM
- Users don't rebuild these complex components
- Even custom generator users need the learning system

**❌ Removed [demos] entirely** because:
- Browser-use, rich, datasets, etc. are for repo examples only
- These should not be part of package distribution
- Demo dependencies belong in repo dev environment

**✅ LangChain moved to [all] extra** because:
- Most users don't need it (LiteLLM covers 100+ providers)
- Enterprise users can add explicitly with `[langchain]`
- Reduces default install size

### Why This Works

- ✅ "5 minutes to agent" - default install just works
- ✅ Clean separation - runtime vs dev vs demo dependencies
- ✅ Flexible - users can mix extras as needed
- ✅ Production-ready - proper dependency groups via PEP 735

---

## 🔄 LOW PRIORITY: Code Organization Improvements 🗂️

**Priority:** 🟢 LOW
**Effort:** 2-4 hours
**Value:** LOW (internal only, easier maintenance)
**Status:** Not started

### What It Does

Splits large files into smaller, more navigable modules. No user-facing changes.

### Current vs Proposed Structure

**Current:**
```
ace/
├── prompts_v2_1.py (1,586 lines - too big!)
├── roles.py (713 lines)
└── ...
```

**Proposed:**
```
ace/prompts/
├── __init__.py
├── v1.py (old prompts)
├── v2.py (deprecated prompts)
└── v2_1/
    ├── __init__.py
    ├── generator.py (~500 lines)
    ├── reflector.py (~500 lines)
    ├── curator.py (~500 lines)
    └── manager.py (PromptManager class)
```

### Why This Could Help

- ✅ Easier for contributors to navigate
- ✅ Matches Python best practices (smaller modules)
- ✅ Clearer version organization

### Why It Might Not Matter

- ⚠️ Purely internal improvement (users don't care)
- ⚠️ Current structure works fine
- ⚠️ More files = more imports

### Implementation Plan

**Files to refactor:**
- `ace/prompts_v2_1.py` → split into `ace/prompts/v2_1/` directory
- Maintain backwards compatibility:
  ```python
  # ace/prompts_v2_1.py (deprecated)
  import warnings
  from ace.prompts.v2_1 import *
  warnings.warn("ace.prompts_v2_1 is deprecated, use ace.prompts.v2_1", DeprecationWarning)
  ```

**Files to update:**
- All examples importing from `prompts_v2_1`
- All tests importing from `prompts_v2_1`
- Update docs to reference new import paths

### Migration Notes

- Add deprecation warnings for old import paths
- Keep old imports working for 1-2 versions
- Document new structure in CLAUDE.md

### Success Metrics

- No test failures
- All examples still work
- Contributors find code easier to navigate

---

## Decision Framework

### When to implement these:

**#1 Context Enhancers:**
- Implement when: Users request RAG integration or you see multiple forks adding this
- Skip if: Pure ACE (no RAG) remains the core use case

**#2 Lite Distribution:**
- Implement when: You get complaints about install size or CI/CD times
- Skip if: All users are fine with batteries-included approach

**#3 Code Organization:**
- Implement when: Contributors complain about navigating large files
- Skip if: Current structure isn't causing issues

### Recommendation

**Start with #1 (Context Enhancers)** if you see user demand for RAG + ACE.

**Do #2 (Lite Distribution)** as a quick win to broaden appeal.

**Skip #3 (Code Organization)** unless it becomes a contributor pain point.

---

## Not Recommended (From OpenCE)

These features **do not align** with our production-first philosophy:

### ❌ Full 5-Pillar Architecture
- **Why skip:** Over-engineering for ACE-specific use cases
- **Violates:** "5 minutes to agent" principle
- **Better:** Our simple context enhancers (#1 above)

### ❌ Method Registry
- **Why skip:** Adds indirection without clear benefit
- **Violates:** Pragmatic simplicity
- **Better:** Direct component instantiation (current approach)

### ❌ RWKV Dedicated Client
- **Why skip:** Niche use case, already support via transformers
- **Violates:** Batteries-included for mainstream providers
- **Better:** Document RWKV usage with existing transformers client

### ❌ Provider Pattern (Lazy Loading)
- **Why skip:** Low user-facing value, adds abstraction
- **Violates:** Simplicity over optimization
- **Better:** Direct instantiation (current approach)

---

## Next Steps

1. Review these 3 TODOs as a team
2. Decide which (if any) to prioritize
3. For #1 (Context Enhancers):
   - Draft full implementation
   - Create example with real vector store
   - Get user feedback on API design
4. For #2 (Lite Distribution):
   - Test minimal install in clean environment
   - Ensure all examples work with `[all]` extra
5. For #3 (Code Organization):
   - Only if contributors request it

---

---

## 🚀 FUTURE: Simple Drop-In LLM Wrapper (ACEClient)

**Priority:** 🟡 DEFERRED (Future work)
**Status:** Design documented, implementation deferred

### Concept: ACE as LLM++

Make ACE a drop-in replacement for LiteLLMClient with automatic learning.

```python
from ace import ACEClient

# Drop-in replacement for any LLM
llm = ACEClient(model="gpt-4o-mini", learning=True)

# Use exactly like a normal LLM
response = llm.complete("What is 2+2?")
print(response.text)  # "4"

# Optionally reflect on results
llm.reflect(feedback="Correct!", ground_truth="4")

# Save/load learned knowledge
llm.save_playbook("smart_llm.json")
llm = ACEClient(model="gpt-4o-mini", playbook="smart_llm.json")
```

### Why This Would Be Valuable

✅ **Zero learning curve** - Just replace your LLM
✅ **Backward compatible** - Implements standard LLMClient interface
✅ **Optional learning** - Can disable for debugging
✅ **Familiar API** - Works with existing LLM-based code

### Target Users

- Developers who want ACE magic with zero conceptual overhead
- Projects that already use LiteLLMClient
- Quick prototyping and demos
- Non-technical users who just want "smarter LLM"

### Implementation Sketch

**File: `ace/ace_client.py` (NEW)**

```python
from typing import Optional
from .llm import LLMClient, LLMResponse
from .playbook import Playbook
from .roles import Generator, Reflector, Curator

class ACEClient(LLMClient):
    """Drop-in LLM replacement with automatic learning."""

    def __init__(
        self,
        model: str,
        learning: bool = True,
        playbook_path: Optional[str] = None,
        **kwargs
    ):
        from .llm_providers import LiteLLMClient
        self.base_llm = LiteLLMClient(model=model, **kwargs)

        self.learning = learning
        if learning:
            self.playbook = (
                Playbook.load_from_file(playbook_path)
                if playbook_path
                else Playbook()
            )
            self.generator = Generator(self.base_llm)
            self.reflector = Reflector(self.base_llm)
            self.curator = Curator(self.base_llm)

        self._interaction_history = []

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Use Generator + playbook if learning enabled."""
        if not self.learning:
            return self.base_llm.complete(prompt, **kwargs)

        output = self.generator.generate(
            question=prompt,
            context="",
            playbook=self.playbook
        )

        self._interaction_history.append({
            'prompt': prompt,
            'output': output
        })

        return LLMResponse(
            text=output.final_answer,
            raw={'reasoning': output.reasoning}
        )

    def reflect(self, feedback: str, ground_truth: Optional[str] = None):
        """Manually trigger reflection and learning."""
        if not self.learning or not self._interaction_history:
            return

        last = self._interaction_history[-1]

        reflection = self.reflector.reflect(
            question=last['prompt'],
            generator_output=last['output'],
            playbook=self.playbook,
            ground_truth=ground_truth,
            feedback=feedback
        )

        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=f"question: {last['prompt']}\nfeedback: {feedback}"
        )

        self.playbook.apply_delta(curator_output.delta)

    def save_playbook(self, path: str):
        if self.learning:
            self.playbook.save_to_file(path)

    def load_playbook(self, path: str):
        if self.learning:
            self.playbook = Playbook.load_from_file(path)
```

### When to Implement

**Implement when:**
- Users request "simpler way to get started"
- You see repeated questions about "basic ACE usage"
- You want to target non-technical users

**Skip if:**
- Current API is working well for target users
- Most users need custom integration (browser-use, etc.)

### Estimated Effort

- **Implementation:** 1-2 days
- **Testing:** 1 day
- **Documentation:** 1 day
- **Total:** 3-4 days

---

## References

- **Full Analysis:** `comparison_analysis/` folder
- **OpenCE Repository:** `other_ace_repos/OpenCE/`
- **Key Documents:**
  - `comparison_analysis/00_executive_summary.md` - High-level findings
  - `comparison_analysis/08_recommendations.md` - Detailed recommendations
  - `comparison_analysis/03_abstraction_layers.md` - Architecture comparison
