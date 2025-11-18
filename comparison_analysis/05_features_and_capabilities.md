# Features & Capabilities Comparison

**Comprehensive feature matrix and unique offerings**

---

## Feature Comparison Matrix

| Feature | Kayba ace-framework | OpenCE | Winner |
|---------|---------------------|---------|--------|
| **Core ACE Implementation** | ✅ | ✅ | Tie |
| **PyPI Package** | ✅ v0.4.0 | ❌ Not published | Kayba |
| **Observability (Opik)** | ✅ Automatic | ❌ None | Kayba |
| **Token/Cost Tracking** | ✅ Automatic | ❌ None | Kayba |
| **Checkpointing** | ✅ During training | ❌ None | Kayba |
| **Prompt Versioning** | ✅ v1, v2, v2.1 | ⚠️ Single version | Kayba |
| **LiteLLM Integration** | ✅ Built-in (100+ providers) | ❌ None | Kayba |
| **Retry Prompts** | ✅ Configurable, multilingual | ⚠️ Hardcoded Chinese | Kayba |
| **JSON Truncation Detection** | ✅ | ❌ Basic | Kayba |
| **LangChain Integration** | ✅ Via client adapter | ✅ Via retriever adapter | Tie |
| **5-Pillar Architecture** | ❌ | ✅ | OpenCE |
| **Pluggable Components** | ❌ | ✅ (Acquirer, Processor, etc.) | OpenCE |
| **Method Registry** | ❌ | ✅ | OpenCE |
| **RAG Integration** | ⚠️ Manual | ✅ Via Acquirer | OpenCE |
| **Processor Chain** | ❌ | ✅ (Compressors, Rerankers) | OpenCE |
| **RWKV Support** | ❌ | ✅ Dedicated client | OpenCE |
| **Browser Automation Demos** | ✅ 4 examples | ❌ None | Kayba |
| **Documentation Quality** | ✅ Extensive | ⚠️ Minimal | Kayba |
| **Test Coverage** | ✅ 10+ integration tests | ✅ Unit + integration | Tie |
| **Dependencies** | Heavy (~50MB LiteLLM) | Light (~5MB core) | OpenCE |
| **Python Version** | 3.11+ | 3.9+ | OpenCE |

---

## Unique Kayba Features

### 1. Opik Observability Integration

**Automatic tracing with zero configuration:**

```python
# Install observability extras
pip install ace-framework[observability]

# Zero config - just use ACE
from ace import LiteLLMClient, Generator

client = LiteLLMClient(model="gpt-4")
generator = Generator(client)

# Automatically tracked in Opik:
# - Every LLM call
# - Token counts & costs
# - Latencies
# - Role attribution (Generator/Reflector/Curator)
result = generator.generate(...)
```

**Opik Dashboard Shows:**
- Total tokens per epoch/sample
- Cost breakdown by role (Generator = 60%, Reflector = 25%, Curator = 15%)
- Latency percentiles (p50, p95, p99)
- Error rates and retry statistics
- Playbook evolution over time

**Production Value:** Essential for cost optimization and SLA monitoring.

### 2. Checkpoint Saving During Training

**Automatic checkpoints:**

```python
from ace import OfflineAdapter

adapter = OfflineAdapter(playbook, generator, reflector, curator)

results = adapter.run(
    samples,
    environment,
    epochs=5,
    checkpoint_interval=10,  # Save every 10 samples
    checkpoint_dir="./checkpoints"
)

# Creates:
# - checkpoints/checkpoint_10.json
# - checkpoints/checkpoint_20.json
# - checkpoints/latest.json
```

**Use Cases:**
- Resume training after interruption
- Early stopping based on validation metrics
- Analyze playbook evolution (which bullets added when)
- A/B test different checkpoint versions

**OpenCE Alternative:** Manual saving in user code.

### 3. Prompt Version Management

**Three prompt versions with benchmarks:**

```python
from ace.prompts import GENERATOR_PROMPT            # v1.0
from ace.prompts_v2 import GENERATOR_PROMPT         # v2.0 (DEPRECATED)
from ace.prompts_v2_1 import PromptManager          # v2.1 (RECOMMENDED)

# Use v2.1 for best performance (+17% vs v1.0)
pm = PromptManager()
generator = Generator(llm, prompt_template=pm.get_generator_prompt())
reflector = Reflector(llm, prompt_template=pm.get_reflector_prompt())
curator = Curator(llm, prompt_template=pm.get_curator_prompt())
```

**Documented benchmarks:**
- v1.0: Baseline (matches paper)
- v2.0: +8% success rate (deprecated due to complexity)
- v2.1: +17% success rate (clearer instructions, better examples)

**OpenCE:** Single prompt version, no benchmarking.

### 4. Configurable Retry Prompts (v0.4.1)

**Multilingual retry prompts:**

```python
# English (default)
generator = Generator(llm)

# Japanese
generator = Generator(
    llm,
    retry_prompt="\n\n[日本語] 有効なJSONオブジェクトのみを返してください。"
)

# German
generator = Generator(
    llm,
    retry_prompt="\n\n[Deutsch] Bitte geben Sie nur ein gültiges JSON-Objekt zurück."
)
```

**Impact:** Reduces JSON parse failures by 7-12% for non-English models.

**OpenCE:** Hardcoded Chinese retry messages.

### 5. Browser Automation Demos

**Production-ready examples:**

```
examples/browser-use/
├── ace_domain_checker.py        # ACE-enhanced automation
├── baseline_domain_checker.py   # Vanilla automation
├── ace_form_filler.py           # Form filling with ACE
└── baseline_form_filler.py      # Baseline comparison
```

**Demonstrated Results:**
- ACE agent: 70% success rate (7/10 domains checked)
- Baseline: 30% success rate (3/10 domains)
- ACE learns strategies like "scroll before clicking hidden elements"

**OpenCE:** No demos, no browser automation support.

### 6. SimpleEnvironment Built-In

**Quick testing environment:**

```python
from ace import SimpleEnvironment, Sample

env = SimpleEnvironment()  # Checks if ground_truth in answer

sample = Sample(question="What is 2+2?", ground_truth="4")
result = env.evaluate(sample, generator_output)
# Returns: "Correct!" or "Incorrect. Expected: 4"
```

**OpenCE:** Users must implement TaskEnvironment for every test.

### 7. Enhanced JSON Parsing

**Kayba strips markdown fences and detects truncation:**

```python
# Handles LLM responses like:
```json
{
  "reasoning": "...",
  "final_answer": "4"
}
```

# Detects truncation:
# Input: {"reasoning": "incomplete... (response cut off by max_tokens)
# Error: "LLM response appears to be truncated JSON.
#         This may indicate the response was cut off mid-generation."
```

**OpenCE:** Basic JSON parsing, no markdown handling.

---

## Unique OpenCE Features

### 1. Five-Pillar Architecture

**Standardized interfaces for any CE technique:**

```python
from opence.interfaces import (
    IAcquirer,      # Fetch knowledge
    IProcessor,     # Transform documents
    IConstructor,   # Build context
    IEvaluator,     # Score outputs
    IEvolver        # Update strategies
)

# ACE is one implementation of these interfaces
# Can plug in ANY context engineering technique
```

**Enables:**
- RAG + ACE hybrids
- Custom compression strategies
- Alternative evaluation metrics (RAGAS, human feedback)
- Prompt optimization evolvers

**Kayba:** No abstraction layer - ACE-specific only.

### 2. Component Composition

**Mix and match implementations:**

```python
from opence.core import ClosedLoopOrchestrator
from opence.components import (
    FileSystemAcquirer,              # Read files
    KeywordBoostReranker,            # Boost priority keywords
    SimpleTruncationProcessor,       # Limit length
    FewShotConstructor,              # Select top-k
    ACEReflectorEvaluator,           # ACE evaluation
    ACECuratorEvolver                # ACE evolution
)

orchestrator = ClosedLoopOrchestrator(
    llm=llm,
    acquirer=FileSystemAcquirer("docs/"),
    processors=[
        KeywordBoostReranker(["safety", "compliance"]),
        SimpleTruncationProcessor(max_chars=2000)
    ],
    constructor=FewShotConstructor(top_k=5),
    evaluator=ACEReflectorEvaluator(reflector, playbook),
    evolver=ACECuratorEvolver(curator, playbook)
)

# You just built a RAG + ACE pipeline
```

**Kayba:** Would require forking Generator to add RAG.

### 3. Method Registry

**Discoverable, configurable methods:**

```python
from opence.methods import ACEClosedLoopMethod, MethodRegistry

# Pre-configured ACE method
method = ACEClosedLoopMethod(
    generator_llm=llm,
    reflector_llm=llm,
    curator_llm=llm
)

artifacts = method.build()
orchestrator = artifacts.orchestrator

# Registry for method discovery
registry = MethodRegistry()
registry.register(method)

# Users can query available methods
print(registry.available())  # ['ace.closed_loop']
```

**Use Case:** CLI tools that let users pick `--method ace.closed_loop` or `--method rag.hybrid`.

**Kayba:** No method registry - users instantiate components directly.

### 4. LangChain Retriever Adapter

**Bridge to LangChain ecosystem:**

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from opence.adapters import LangChainRetrieverAcquirer

# LangChain vector store
vectorstore = FAISS.from_texts(documents, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Use in OpenCE pipeline
acquirer = LangChainRetrieverAcquirer(retriever)

orchestrator = ClosedLoopOrchestrator(
    llm=llm,
    acquirer=acquirer,  # LangChain retriever
    # ... rest of pipeline
)
```

**Kayba:** Has LangChain LLM adapter, but no retriever integration.

### 5. Processor Implementations

**Built-in processors:**

#### KeywordBoostReranker
```python
from opence.components import KeywordBoostReranker

# Boost documents containing priority keywords
reranker = KeywordBoostReranker(["safety", "fire", "emergency"])
documents = reranker.process(documents, request)
# Documents with keywords ranked higher
```

#### SimpleTruncationProcessor
```python
from opence.components import SimpleTruncationProcessor

# Limit total document length
truncator = SimpleTruncationProcessor(max_chars=2000)
documents = truncator.process(documents, request)
# Truncates to fit within limit
```

**Kayba:** No processor layer.

### 6. RWKV Model Support

**Dedicated client for RWKV:**

```python
from opence.models import RWKVModelProvider

provider = RWKVModelProvider(
    model_path="/path/to/rwkv-7b.pth",
    tokenizer_path="/path/to/tokenizer.json",
    strategy="cuda fp16i8 *20 -> cpu fp32"  # Offloading strategy
)

client = provider.client()
generator = Generator(client)
```

**Use Case:** Running ACE on consumer GPUs with RWKV's memory-efficient architecture.

**Kayba:** No RWKV support.

### 7. Provider Pattern with Lazy Loading

**Efficient resource management:**

```python
from opence.models import BaseModelProvider

class CustomModelProvider(BaseModelProvider):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path

    def create_client(self) -> LLMClient:
        # Only loaded when first accessed
        return HeavyLLMClient(self.model_path)

provider = CustomModelProvider("/huge/model")
# Model not loaded yet

client = provider.client()  # Now loaded
client2 = provider.client()  # Returns cached instance
```

**Kayba:** Direct instantiation, no lazy loading.

---

## Feature Gap Analysis

### What Kayba Has (OpenCE Missing)

1. ✅ **Production observability** - Automatic Opik tracing
2. ✅ **Cost tracking** - Token counts and costs per role
3. ✅ **Checkpointing** - Resume training from any epoch
4. ✅ **Prompt versioning** - Benchmarked v1/v2/v2.1
5. ✅ **LiteLLM integration** - 100+ provider support
6. ✅ **Browser automation** - Production demos
7. ✅ **SimpleEnvironment** - Quick testing
8. ✅ **Enhanced JSON parsing** - Markdown stripping, truncation detection
9. ✅ **Configurable retries** - Multilingual retry prompts
10. ✅ **PyPI distribution** - `pip install ace-framework`

### What OpenCE Has (Kayba Missing)

1. ✅ **5-pillar architecture** - Generic CE interfaces
2. ✅ **Component composition** - Mix/match Acquirers, Processors, etc.
3. ✅ **Method registry** - Discoverable CE techniques
4. ✅ **RAG integration** - Built-in Acquirer layer
5. ✅ **Processor chain** - Transform documents before generation
6. ✅ **LangChain retriever adapter** - Use LangChain vector stores
7. ✅ **RWKV support** - Memory-efficient local models
8. ✅ **Provider pattern** - Lazy loading, caching
9. ✅ **Lighter dependencies** - Minimal core
10. ✅ **Broader Python support** - 3.9+ vs 3.11+

---

## Use Case: Which Features Matter?

### Production Deployment → Kayba

**Essential features:**
- Observability (Opik)
- Cost tracking
- PyPI package
- LiteLLM (multi-provider)
- Checkpointing

**Example:** Deploy ACE agent to monitor customer support tickets, track costs per thousand tickets processed.

### Research Experimentation → OpenCE

**Essential features:**
- 5-pillar architecture
- Component composition
- Processor chain
- Method registry

**Example:** Test if compressing documents before ACE generation improves performance on long-context tasks.

### Hybrid RAG + ACE System → OpenCE

**Essential features:**
- Acquirer interface
- LangChain adapter
- Processor chain

**Example:** Retrieve documents from vector store, rerank by relevance, pass to ACE Generator.

### Cost-Sensitive Production → Kayba

**Essential features:**
- Automatic cost tracking
- Multi-provider support (use cheapest provider per role)
- Checkpointing (avoid re-running expensive epochs)

**Example:** Use GPT-3.5 for Generator, GPT-4 for Reflector, optimize per role.

---

## Convergence Opportunities

### Kayba Could Add:

1. **Optional processor interface** for compression/RAG
   ```python
   generator = Generator(llm, processors=[LLMCompressor()])
   ```

2. **Method registry** for discoverable configurations
   ```python
   from ace.methods import ACEMethod
   method = ACEMethod.load("ace_v2_1_with_checkpointing")
   ```

3. **RWKV support** as optional extra
   ```bash
   pip install ace-framework[rwkv]
   ```

### OpenCE Could Add:

1. **Opik integration** as optional dependency
   ```bash
   pip install opence[observability]
   ```

2. **LiteLLM provider** for universal LLM support
   ```python
   from opence.models import LiteLLMProvider
   provider = LiteLLMProvider(model="claude-3-opus")
   ```

3. **Prompt versioning** with benchmarks
   ```python
   from opence.methods.ace import PromptManager
   pm = PromptManager(version="2.1")
   ```

4. **PyPI release** for easier adoption
   ```bash
   pip install opence
   ```

---

## Bottom Line

**Kayba excels at production features:**
- Observability, cost tracking, multi-provider support
- Best for shipping ACE to production quickly

**OpenCE excels at research features:**
- Composable architecture, RAG integration, processor chains
- Best for experimenting with hybrid CE techniques

**Both are excellent implementations** serving different needs. A merger of features would create the ultimate ACE framework: Kayba's production tooling + OpenCE's architectural flexibility.
