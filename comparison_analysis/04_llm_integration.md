# LLM Integration Comparison

**How each implementation handles LLM providers, clients, and observability**

---

## Overview

| Aspect | Kayba | OpenCE |
|--------|-------|--------|
| **Primary LLM Client** | LiteLLM (100+ providers) | OpenAI-compatible API |
| **Provider Support** | ✅ Universal (OpenAI, Anthropic, Google, Cohere, etc.) | ⚠️ OpenAI API format only |
| **Observability** | ✅ Automatic Opik tracing | ❌ None |
| **Cost Tracking** | ✅ Automatic token/cost tracking | ❌ None |
| **Local Models** | ✅ Via transformers | ✅ Via transformers |
| **LangChain Integration** | ✅ Via optional client | ✅ Via adapter layer |
| **RWKV Support** | ❌ | ✅ Dedicated client |

---

## 1. LLM Client Interface

### Kayba: LiteLLM-First

**Base Interface:**
```python
# ace/llm.py
class LLMClient(ABC):
    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Return the model text for a given prompt."""

@dataclass
class LLMResponse:
    text: str
    raw: Optional[Dict[str, Any]] = None  # Full API response
```

**Primary Implementation:**
```python
# ace/llm_providers/litellm_client.py
from litellm import completion

class LiteLLMClient(LLMClient):
    """Universal LLM client supporting 100+ providers via LiteLLM."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **litellm_kwargs: Any
    ) -> None:
        super().__init__(model=model)
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.litellm_kwargs = litellm_kwargs

    @maybe_track(name="llm_complete", tags=["llm", "completion"])
    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        # Automatic Opik tracing via decorator
        messages = [{"role": "user", "content": prompt}]

        response = completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
            **{**self.litellm_kwargs, **kwargs}
        )

        return LLMResponse(
            text=response.choices[0].message.content,
            raw=response.model_dump() if hasattr(response, "model_dump") else None
        )
```

**Supported Providers (via LiteLLM):**
- OpenAI: `gpt-4`, `gpt-3.5-turbo`, `gpt-4-turbo`
- Anthropic: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- Google: `gemini-pro`, `gemini-1.5-pro`
- Cohere: `command-r`, `command-r-plus`
- Azure OpenAI: `azure/<deployment>`
- AWS Bedrock: `bedrock/<model>`
- Ollama: `ollama/<model>`
- Together AI, Replicate, Anyscale, etc.

**Automatic Observability:**
```python
# Usage - observability is automatic
client = LiteLLMClient(model="gpt-4")
response = client.complete("Hello world")
# → Automatically logged to Opik:
#    - Model: gpt-4
#    - Tokens: input=2, output=5, total=7
#    - Cost: $0.00021
#    - Latency: 342ms
#    - Trace ID: abc-123-def
```

### OpenCE: OpenAI API Format

**Base Interface (Identical):**
```python
# opence/models/clients.py
class LLMClient(ABC):
    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Return the model text for a given prompt."""
```

**Primary Implementation:**
```python
# opence/models/clients.py
from openai import OpenAI

class DeepseekLLMClient(LLMClient):
    """OpenAI-compatible API client (named for DeepSeek but works with any OpenAI API)."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(model=model)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.system_prompt = system_prompt

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )

        return LLMResponse(
            text=response.choices[0].message.content or "",
            raw={"usage": response.usage, "model": response.model}
        )
```

**Supported Providers:**
- ✅ OpenAI (native)
- ✅ Any OpenAI-compatible API (DeepSeek, Groq, vLLM, etc.)
- ❌ Anthropic (requires wrapper)
- ❌ Google (requires wrapper)
- ❌ Cohere (requires wrapper)

**No Observability:**
- No automatic tracing
- No token/cost tracking
- Users must manually instrument

---

## 2. Provider Support Comparison

### Kayba: Universal via LiteLLM

**One-line provider switching:**
```python
# OpenAI
client = LiteLLMClient(model="gpt-4")

# Anthropic
client = LiteLLMClient(model="claude-3-opus-20240229")

# Google
client = LiteLLMClient(model="gemini-pro")

# Cohere
client = LiteLLMClient(model="command-r-plus")

# Azure OpenAI
client = LiteLLMClient(
    model="azure/gpt-4-deployment",
    api_key=os.getenv("AZURE_API_KEY"),
    base_url=os.getenv("AZURE_ENDPOINT")
)

# Local Ollama
client = LiteLLMClient(model="ollama/llama2")

# All work identically - no code changes needed
```

**Benefit:** Users can switch providers without changing ACE code.

### OpenCE: Manual Provider Setup

**Switching providers requires:**

**Option 1: Use OpenAI-compatible endpoint**
```python
# Works with OpenAI, Groq, vLLM, etc.
client = DeepseekLLMClient(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1"
)

# DeepSeek
client = DeepseekLLMClient(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
```

**Option 2: Implement custom LLMClient**
```python
# For non-OpenAI APIs (Anthropic, Google), user must write:
import anthropic

class AnthropicLLMClient(LLMClient):
    def __init__(self, model: str, api_key: str):
        super().__init__(model=model)
        self.client = anthropic.Anthropic(api_key=api_key)

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 1024),
            messages=[{"role": "user", "content": prompt}]
        )
        return LLMResponse(text=message.content[0].text)
```

**Tradeoff:**
- ✅ No LiteLLM dependency
- ❌ Users must implement clients for non-OpenAI providers

---

## 3. Observability & Cost Tracking

### Kayba: Automatic Opik Integration

**Zero-config tracing:**
```python
# Just import and use - tracing automatic
from ace import LiteLLMClient, Generator, Reflector, Curator
from ace import OfflineAdapter, Playbook, Sample, SimpleEnvironment

# Opik auto-instruments if installed
client = LiteLLMClient(model="gpt-4")
generator = Generator(client)
reflector = Reflector(client)
curator = Curator(client)

# All LLM calls automatically tracked
adapter = OfflineAdapter(Playbook(), generator, reflector, curator)
results = adapter.run(samples, SimpleEnvironment(), epochs=3)

# View in Opik dashboard:
# - Total tokens: 45,231 input + 12,453 output
# - Total cost: $1.23
# - Average latency: 847ms
# - Success rate: 94%
# - Generator/Reflector/Curator breakdown
```

**What gets tracked:**
```python
# Each LLM call creates an Opik trace with:
{
    "model": "gpt-4",
    "provider": "openai",
    "input_tokens": 1234,
    "output_tokens": 567,
    "total_tokens": 1801,
    "cost_usd": 0.0234,
    "latency_ms": 456,
    "prompt": "...",  # First 1000 chars
    "response": "...",  # Full response
    "metadata": {
        "role": "generator",  # or reflector, curator
        "epoch": 2,
        "sample": 15
    }
}
```

**Tracing Decorator:**
```python
# ace/observability/tracers.py
def maybe_track(name: Optional[str] = None, tags: Optional[List[str]] = None, **kwargs):
    """Decorator that adds Opik tracing if available, else no-op."""
    def decorator(func):
        if has_opik():
            return opik.track(name=name or func.__name__, tags=tags or [], **kwargs)(func)
        return func
    return decorator

# Usage in roles:
@maybe_track(name="generator_generate", tags=["ace", "generator"])
def generate(self, question, context, playbook, **kwargs):
    # Automatically traced if Opik installed
    ...
```

**Graceful degradation:**
- If Opik not installed: No errors, tracing simply disabled
- If Opik installed but not configured: Logs warning, continues without tracing

### OpenCE: No Built-in Observability

**Users must manually instrument:**
```python
# No automatic tracing
client = DeepseekLLMClient(model="gpt-4")
generator = Generator(client)

# To track costs, user must:
import time

def tracked_generate(*args, **kwargs):
    start = time.time()
    result = generator.generate(*args, **kwargs)
    latency = time.time() - start

    # Manually extract tokens from LLM response
    tokens = result.raw.get("usage", {})
    print(f"Tokens: {tokens}, Latency: {latency:.2f}s")

    return result
```

**Limitation:** No centralized dashboard, no cost aggregation across runs.

---

## 4. Local Model Support

### Kayba: Transformers Client

```python
# ace/llm.py (lines 90+)
class TransformersLLMClient(LLMClient):
    """Local model support via Hugging Face transformers."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        device_map: Union[str, Dict[str, int]] = "auto",
        torch_dtype: Union[str, "torch.dtype"] = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().__init__(model=model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            do_sample=True
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove input prompt from output
        text = text[len(prompt):].strip()

        return LLMResponse(text=text)
```

**Usage:**
```python
from ace.llm import TransformersLLMClient

client = TransformersLLMClient(
    model_path="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",  # Multi-GPU
    torch_dtype="float16"
)
```

### OpenCE: Transformers + Provider Pattern

**More structured approach:**
```python
# opence/models/clients.py
class TransformersLLMClient(LLMClient):
    # Similar implementation to Kayba
    ...

# opence/models/providers.py
class TransformersModelProvider(BaseModelProvider):
    """Provider pattern for local models."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        device_map: str | Dict[str, int] = "auto",
        torch_dtype: str | "torch.dtype" = "auto",
    ) -> None:
        super().__init__()
        self.model_path = model_path
        # ... store parameters ...

    def create_client(self) -> LLMClient:
        return TransformersLLMClient(
            self.model_path,
            max_new_tokens=self.max_new_tokens,
            # ...
        )
```

**Usage:**
```python
from opence.models import TransformersModelProvider

provider = TransformersModelProvider(
    model_path="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto"
)
client = provider.client()  # Lazy initialization
```

**Benefit:** Provider pattern enables caching and lazy loading.

### OpenCE Exclusive: RWKV Support

```python
# opence/models/rwkv_client.py
class RWKVLLMClient(LLMClient):
    """Dedicated client for RWKV models."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        strategy: str = "cuda fp16i8 *20 -> cpu fp32",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.5,
    ) -> None:
        from rwkv.model import RWKV
        from rwkv.utils import PIPELINE

        super().__init__(model=model_path)
        self.model = RWKV(model=model_path, strategy=strategy)
        self.pipeline = PIPELINE(self.model, tokenizer_path)
        # ... configure sampling ...

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        # RWKV-specific generation logic
        ...
```

**Usage:**
```python
from opence.models import RWKVModelProvider

provider = RWKVModelProvider(
    model_path="/path/to/rwkv-model.pth",
    tokenizer_path="/path/to/tokenizer.json",
    strategy="cuda fp16"
)
```

**Kayba:** No RWKV support (could be added as optional extra).

---

## 5. LangChain Integration

### Kayba: Optional LangChain Client

```python
# ace/llm_providers/langchain_client.py
from langchain_core.language_models import BaseChatModel

class LangChainLLMClient(LLMClient):
    """Adapter for LangChain chat models."""

    def __init__(self, chat_model: BaseChatModel) -> None:
        super().__init__(model=str(chat_model))
        self.chat_model = chat_model

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content=prompt)]
        response = self.chat_model.invoke(messages, **kwargs)

        return LLMResponse(
            text=response.content,
            raw={"model": str(self.chat_model)}
        )
```

**Usage:**
```python
from langchain_openai import ChatOpenAI
from ace.llm_providers import LangChainLLMClient

lc_model = ChatOpenAI(model="gpt-4", temperature=0.7)
client = LangChainLLMClient(lc_model)

# Use with ACE
generator = Generator(client)
```

**Install:**
```bash
pip install ace-framework[langchain]
```

### OpenCE: Adapter Layer

```python
# opence/adapters/langchain.py
from langchain.retrievers import BaseRetriever
from opence.interfaces import IAcquirer, Document, LLMRequest

class LangChainRetrieverAcquirer(IAcquirer):
    """Bridge LangChain retrievers to OpenCE IAcquirer."""

    def __init__(self, retriever: BaseRetriever) -> None:
        self.retriever = retriever

    def acquire(self, request: LLMRequest) -> List[Document]:
        lc_docs = self.retriever.get_relevant_documents(request.question)
        return [
            Document(
                id=str(i),
                content=doc.page_content,
                metadata=doc.metadata
            )
            for i, doc in enumerate(lc_docs)
        ]
```

**Usage:**
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from opence.adapters import LangChainRetrieverAcquirer

# LangChain vector store
vectorstore = FAISS.from_texts(texts, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Bridge to OpenCE
acquirer = LangChainRetrieverAcquirer(retriever)

# Use in orchestrator
orchestrator = ClosedLoopOrchestrator(
    llm=llm,
    acquirer=acquirer,  # LangChain retriever
    # ...
)
```

**Benefit:** OpenCE can leverage LangChain's ecosystem (retrievers, document loaders, etc.).

---

## 6. Comparison Matrix

| Feature | Kayba | OpenCE |
|---------|-------|--------|
| **Provider Support** | 100+ via LiteLLM | OpenAI-compatible only |
| **One-line provider switch** | ✅ | ❌ (requires custom client) |
| **Automatic observability** | ✅ Opik | ❌ None |
| **Token tracking** | ✅ Automatic | ❌ Manual |
| **Cost tracking** | ✅ Automatic | ❌ Manual |
| **Local models (Transformers)** | ✅ | ✅ |
| **RWKV support** | ❌ | ✅ |
| **LangChain integration** | ✅ Via adapter client | ✅ Via retriever adapter |
| **Provider pattern** | ❌ Direct client | ✅ BaseModelProvider |
| **Lazy loading** | ❌ | ✅ Via provider pattern |
| **Dependency weight** | Heavy (LiteLLM ~50MB) | Light (OpenAI client ~5MB) |

---

## 7. Production Considerations

### Kayba Advantages

1. **Universal compatibility**: Switch providers without code changes
2. **Built-in observability**: Opik tracks every LLM call
3. **Cost optimization**: Real-time cost monitoring and budgeting
4. **Error handling**: Retry logic with exponential backoff (via tenacity)
5. **Rate limiting**: Handled by LiteLLM

**Example production setup:**
```python
import opik

opik.configure(
    workspace="production",
    project="ace-agents"
)

client = LiteLLMClient(
    model="gpt-4",
    max_retries=3,
    timeout=30,
    api_key=os.getenv("OPENAI_API_KEY")
)

# All calls auto-tracked, costs monitored, errors retried
```

### OpenCE Advantages

1. **Lightweight**: No heavy LiteLLM dependency
2. **Provider flexibility**: Users choose their own abstraction (LiteLLM, LangChain, etc.)
3. **RWKV support**: Specialized client for RWKV models
4. **Provider pattern**: Lazy loading and caching

**Example research setup:**
```python
# Minimal dependencies
provider = TransformersModelProvider(
    model_path="/path/to/local/model",
    device_map={"": 0}  # Specific GPU
)

# No observability overhead
client = provider.client()
```

---

## 8. Recommendations

### For Kayba Users

**Strengths to leverage:**
- Use LiteLLM for multi-provider support
- Rely on Opik for production monitoring
- Track costs across different providers

**Gaps to address:**
- Consider adding RWKV support as optional extra
- Document rate limit handling for different providers

### For OpenCE Users

**Strengths to leverage:**
- Provider pattern is clean abstraction
- RWKV support is unique

**Gaps to address:**
- Add LiteLLM as optional provider (best of both worlds)
- Integrate observability layer (optional dependency)
- Document how to implement custom clients for Anthropic/Google/Cohere

---

## 9. Convergence Opportunity

**Both repos could benefit from:**

### Kayba could add:
```python
# Optional provider pattern
from ace.llm_providers import BaseModelProvider, LiteLLMProvider

provider = LiteLLMProvider(model="gpt-4")
client = provider.client()  # Lazy init, caching
```

### OpenCE could add:
```python
# Optional LiteLLM integration
pip install opence[litellm]

from opence.models import LiteLLMProvider

provider = LiteLLMProvider(model="claude-3-opus")  # Works out-of-box
```

**Result:** Best of both worlds - lightweight core with powerful optional integrations.

---

## Bottom Line

**Kayba wins on production convenience:**
- ✅ Universal provider support (100+)
- ✅ Automatic observability and cost tracking
- ✅ Zero-config experience

**OpenCE wins on architectural purity:**
- ✅ Lightweight core (no heavy dependencies)
- ✅ Provider pattern (clean abstraction)
- ✅ RWKV support

**Ideal setup:** Kayba's LiteLLM integration + OpenCE's provider pattern + automatic Opik observability.
