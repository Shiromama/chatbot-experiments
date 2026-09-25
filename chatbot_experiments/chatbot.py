import ollama
import time
import re

MAIN_MODEL = "jaahas/qwen3.5-uncensored:2b"

SYSTEM_PROMPT = "You are a helpful AI assistant. Answer clearly, directly, and naturally."

EXIT_WORDS = {"exit", "quit"}

SIMPLE_PROMPTS = {
    "hi", "hello", "hey", "thanks", "thank you",
    "good morning", "good afternoon", "good evening",
    "yo", "sup", "ok", "okay", "nice", "cool"
}

THINK_KEYWORDS_STRONG = [
    "solve", "calculate", "compute", "derive", "derivation",
    "integrate", "differentiate", "prove", "proof",
    "debug", "fix this code", "analyze this code",
    "step by step", "explain step by step",
    "why does this error happen", "trace this bug",
    "compare in depth", "deep comparison",
    "logic", "reasoning", "multi-step", "analyze"
]

THINK_KEYWORDS_MEDIUM = [
    "equation", "matrix", "determinant", "limit",
    "function", "derivative", "integral", "differential",
    "laplace", "probability", "statistics",
    "algorithm", "complexity", "bug", "error",
    "exception", "stack trace", "refactor",
    "optimize this code", "explain why",
    "how does this work"
]

NO_THINK_PATTERNS = [
    r"^(hi|hello|hey|yo)\b",
    r"^how are you\b",
    r"^what's up\b",
    r"^who are you\b",
    r"^tell me a joke\b",
    r"^give me a caption\b",
    r"^write a short\b",
    r"^summarize this\b",
    r"^translate\b",
]

def looks_like_math(prompt: str) -> bool:
    """
    Detect obvious math structure.
    """
    math_patterns = [
        r"\d+\s*[\+\-\*/=]\s*\d+",   # 2+2, x=3
        r"\b[xy]\s*=\s*[-\d\w\+\*/^()]+",
        r"\bdy/dx\b",
        r"\bdx\b",
        r"\bint\b",
        r"\blim\b",
        r"\bsin\b|\bcos\b|\btan\b|\blog\b|\bln\b",
        r"\^",                       # exponents
        r"[∫∑√π]",
    ]
    return any(re.search(p, prompt, re.IGNORECASE) for p in math_patterns)

def looks_like_code(prompt: str) -> bool:
    """
    Detect obvious code/debug prompts.
    """
    code_patterns = [
        r"```",                      # code block
        r"\bdef\b|\bclass\b|\bimport\b",
        r"\bfor\b.+\bin\b",
        r"\bif\b.+:",
        r"\bprint\s*\(",
        r"\bTraceback\b",
        r"\bSyntaxError\b|\bTypeError\b|\bNameError\b|\bValueError\b",
        r"\.py\b|\bpython\b|\bjavascript\b|\bc\+\+\b|\bjava\b",
    ]
    return any(re.search(p, prompt, re.IGNORECASE) for p in code_patterns)

def should_think_rule_based(prompt: str) -> bool:
    """
    Fast heuristic router:
    - default is NO THINK
    - only enable THINK on strong evidence
    """
    text = prompt.strip()
    lower = text.lower()

    if not text:
        return False

    # Exact trivial messages
    if lower in SIMPLE_PROMPTS:
        return False

    # Very short prompts are usually not worth thinking about
    # unless they look like math/code
    word_count = len(lower.split())
    if word_count <= 3 and not looks_like_math(text) and not looks_like_code(text):
        return False

    # Explicit no-think style prompts
    for pattern in NO_THINK_PATTERNS:
        if re.search(pattern, lower):
            return False

    score = 0

    # Strong keywords
    for kw in THINK_KEYWORDS_STRONG:
        if kw in lower:
            score += 3

    # Medium keywords
    for kw in THINK_KEYWORDS_MEDIUM:
        if kw in lower:
            score += 1

    # Structural hints
    if looks_like_math(text):
        score += 3

    if looks_like_code(text):
        score += 3

    # Questions asking for detailed process
    if "step by step" in lower or "explain in detail" in lower:
        score += 3

    # Longer prompts are more likely to need reasoning
    if word_count > 25:
        score += 1

    # Multiple question marks / dense requests can imply analysis
    if lower.count("?") >= 2:
        score += 1

    return score >= 3


history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

while True:
    prompt = input("You: ").strip()

    if prompt.lower() in EXIT_WORDS:
        print("Goodbye!")
        break

    router_start = time.perf_counter()
    use_thinking = should_think_rule_based(prompt)
    router_end = time.perf_counter()

    history.append({"role": "user", "content": prompt})

    print(f"\n[Router decision: {'THINK' if use_thinking else 'NO THINK'}]")
    print(f"[Router time: {router_end - router_start:.4f} seconds]")

    start_time = time.perf_counter()

    stream = ollama.chat(
        model=MAIN_MODEL,
        messages=history,
        think=use_thinking,
        stream=True
    )

    full_answer = ""
    full_thinking = ""
    in_thinking = False
    answer_started = False

    for chunk in stream:
        msg = chunk["message"]
        thinking_part = msg.get("thinking", "")
        content_part = msg.get("content", "")

        if use_thinking and thinking_part:
            if not in_thinking:
                in_thinking = True
                print("\n[Thinking]")
            print(thinking_part, end="", flush=True)
            full_thinking += thinking_part

        if content_part:
            if not answer_started:
                answer_started = True
                print("\n\nAssistant:")
            print(content_part, end="", flush=True)
            full_answer += content_part

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print(f"\n\n[Done in {elapsed:.2f} seconds]\n")

    history.append({"role": "assistant", "content": full_answer})