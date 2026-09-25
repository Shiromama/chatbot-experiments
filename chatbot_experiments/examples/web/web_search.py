import os
import ollama
import time
import json
import re
import requests
import trafilatura
from datetime import datetime
from urllib.parse import urlparse

try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None


ROUTER_MODEL = "jaahas/qwen3.5-uncensored:2b"
MAIN_MODEL = "jaahas/qwen3.5-uncensored:2b"
VL_MODEL = "qwen3-vl:2b"

SYSTEM_PROMPT = """
You are a precise and reliable AI assistant.

CORE RULES:
1. Follow the user's instruction exactly.
2. Do not add unrelated suggestions or extra content.
3. Do not assume intent beyond what is stated.

CLARITY RULE:
- Ask for clarification only when the user's actual task is genuinely unclear.
- Do NOT ask for clarification when enough information is already available to answer.

IMAGE RULE:
- If the user asks to "describe" or "what do you see":
  → ONLY describe, do NOT analyze or solve.

REASONING RULE:
- If the task inherently requires reasoning (math, logic, debugging, analysis),
  do the reasoning and answer directly.

OUTPUT CONTROL:
- Give the answer directly when the task is clear.
- Present a clean final answer unless the user asks for steps.

PRIORITY:
Accuracy > Correctness > Instruction-following > Brevity
"""

DEBUG_MODE = True
TEMP_IMAGE_DIR = "temp_images"

OLLAMA_WEB_SEARCH_URL = "https://ollama.com/api/web_search"
WEB_SEARCH_MAX_RESULTS = 5

URL_REGEX = r"(https?://[^\s]+)"
MAX_PAGE_CHARS = 12000
FOLLOWUP_MAX_PAGE_CHARS = 9000
REQUEST_TIMEOUT = 20

last_page = None
saved_pages = []


def debug_print(msg: str) -> None:
    if DEBUG_MODE:
        print(msg)


def ensure_temp_dir() -> None:
    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)


def extract_json_object(raw: str) -> dict:
    raw = raw.strip()
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def parse_router_output(raw: str) -> dict:
    data = extract_json_object(raw)
    return {
        "search": bool(data.get("search", False)),
        "think": bool(data.get("think", False))
    }


def route_prompt_with_router(prompt: str) -> dict:
    simple_prompts = {
        "hi", "hello", "hey", "thanks", "thank you",
        "good morning", "good evening", "good afternoon"
    }
    if prompt.lower().strip() in simple_prompts:
        return {"search": False, "think": False}

    router_messages = [
        {
            "role": "system",
            "content": (
                "You are a routing classifier.\n"
                "Your only job is to classify whether the user's prompt needs:\n"
                "1. internet search\n"
                "2. deep reasoning\n\n"
                "Return ONLY valid JSON in exactly this format:\n"
                '{"search": true, "think": false}\n\n'
                "Use search=true if the prompt:\n"
                "- asks for latest, current, recent, news, updates\n"
                "- needs internet lookup or web information\n"
                "- asks to search, look up, find online, check online\n"
                "- asks about documentation, APIs, releases, or current tools\n"
                "- asks about current events, websites, products, prices, recent info, or online facts\n\n"
                "Use search=false if the prompt is answerable without web search.\n\n"
                "Use think=true if the prompt involves:\n"
                "For math:\n"
                "- think=false for routine calculations\n"
                "- think=false for simple algebra\n"
                "- think=false for basic derivatives/integrals\n"
                "- think=false for straightforward equation solving\n"
                "- think=true only for hard proofs, difficult word problems, tricky symbolic reasoning, or clearly complex multi-step math\n\n"

            "Do not enable think just because the prompt contains math.\n"
            "If unsure, set think=false.\n"
                "Use think=false for simple/direct prompts.\n\n"
                "Output JSON only. No explanation."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = ollama.chat(
        model=ROUTER_MODEL,
        messages=router_messages,
        think=False,
        stream=False,
        options={"temperature": 0}
    )

    raw = response["message"]["content"]
    return parse_router_output(raw)


def extract_paste_instruction(prompt: str) -> str:
    return prompt[len("/paste"):].strip()


def grab_clipboard_image():
    if ImageGrab is None:
        return None

    try:
        return ImageGrab.grabclipboard()
    except Exception:
        return None


def save_clipboard_image(img) -> str:
    ensure_temp_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(TEMP_IMAGE_DIR, f"clipboard_{timestamp}.png")
    img.save(img_path)
    return img_path


def analyze_image_with_vl(image_path: str, instruction: str) -> str:
    debug_print("\n[IMAGE] Clipboard image captured")
    debug_print(f"[SAVE] Temp image saved: {image_path}")
    debug_print(f"[VL] Sending image to vision-language model: {VL_MODEL}")

    _ = instruction.strip() if instruction.strip() else "Describe the image clearly and accurately."

    vl_messages = [
        {
            "role": "system",
            "content": (
                "You are a strict image transcription model.\n"
                "\n"
                "YOUR ONLY JOB:\n"
                "- Extract and describe what is VISIBLE in the image\n"
                "- Transcribe text, equations, symbols, and structure exactly\n"
                "\n"
                "STRICT RULES:\n"
                "- DO NOT solve problems\n"
                "- DO NOT analyze or interpret meaning\n"
                "- DO NOT explain anything\n"
                "- DO NOT answer the user's question\n"
                "- DO NOT infer beyond what is visible\n"
                "\n"
                "IF the image contains a math problem:\n"
                "- Output ONLY the problem exactly as written\n"
                "- Preserve equations and formatting\n"
                "\n"
                "If something is unclear, say it is unclear.\n"
            )
        },
        {
            "role": "user",
            "content": (
                "Transcribe and describe ONLY what is visible in the image.\n"
                "Do NOT solve or explain anything."
            ),
            "images": [image_path]
        }
    ]

    response = ollama.chat(
        model=VL_MODEL,
        messages=vl_messages,
        stream=False
    )

    vl_output = response["message"]["content"].strip()

    debug_print("[VL] Vision model response received")
    if vl_output:
        preview = vl_output[:1000]
        debug_print(f"[VL OUTPUT]\n{preview}\n")
    else:
        debug_print("[VL OUTPUT] [empty]\n")

    return vl_output


def looks_like_math(text: str) -> bool:
    text_l = text.lower()
    math_patterns = [
        "f(x)", "solve", "determine", "find",
        "integral", "derivative", "limit",
        "dx", "dy", "d/dx",
        "=", "^", "∫", "√",
        "x^", "y^",
        "interval", "[", "]"
    ]
    return any(p in text_l for p in math_patterns)


def is_description_request(instruction: str) -> bool:
    if not instruction.strip():
        return False

    text = instruction.lower().strip()
    description_phrases = [
        "describe",
        "what do you see",
        "what is in the image",
        "read this image",
        "transcribe this",
        "what's in this screenshot"
    ]
    return any(p in text for p in description_phrases)


def build_image_prompt_from_vl(instruction: str, vl_output: str) -> str:
    extracted = vl_output.strip() if vl_output.strip() else "[Could not read the image clearly.]"
    user_task = instruction.strip() if instruction.strip() else "Describe the image."

    return (
        f"User request:\n{user_task}\n\n"
        f"Extracted text/content from the image:\n{extracted}\n\n"
        "Use the extracted image content above as the source material.\n"
        "Answer the user's request directly.\n"
        "Do not discuss tools, OCR, pipeline stages, or alternative methods.\n"
        "Only ask for clarification if the extracted content is actually unreadable or incomplete."
    )


def web_search_ollama(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> list:
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        debug_print("[WEB SEARCH] Missing OLLAMA_API_KEY")
        return []

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "query": query,
        "max_results": max_results
    }

    try:
        response = requests.post(
            OLLAMA_WEB_SEARCH_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        debug_print(f"[WEB SEARCH ERROR] {e}")
        return []


def format_web_results(results: list) -> str:
    if not results:
        return "[No web search results found.]"

    blocks = []
    for i, item in enumerate(results, 1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        content = item.get("content", "")

        blocks.append(
            f"[Result {i}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Snippet: {content}"
        )

    return "\n\n".join(blocks)


def build_search_augmented_prompt(user_prompt: str, web_context: str) -> str:
    return (
        f"The user asked:\n{user_prompt}\n\n"
        f"Web search results:\n{web_context}\n\n"
        f"Use the web search results above when helpful. "
        f"Prefer grounded answers based on the search results when they are relevant. "
        f"If the search results are insufficient or uncertain, say so clearly."
    )


def extract_first_url(text: str):
    match = re.search(URL_REGEX, text)
    if not match:
        return None
    return match.group(1).rstrip(".,);]}>\"'")


def remove_url_from_text(text: str, url: str) -> str:
    cleaned = text.replace(url, " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def fetch_url_content(url: str) -> tuple[str, str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    html = response.text
    downloaded = trafilatura.fetch_url(url)
    extracted = trafilatura.extract(
        downloaded if downloaded else html,
        include_comments=False,
        include_tables=True,
        include_links=True,
        favor_precision=True
    )

    if not extracted or not extracted.strip():
        raise ValueError("Trafilatura could not extract useful content from this page.")

    metadata = trafilatura.extract_metadata(downloaded if downloaded else html)
    title = metadata.title.strip() if metadata and metadata.title else ""

    if not title:
        parsed = urlparse(url)
        title = parsed.netloc

    return title, extracted.strip()


def trim_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[Content truncated for context size.]"


def build_link_analysis_prompt(user_instruction: str, url: str, title: str, content: str) -> str:
    task = user_instruction.strip() if user_instruction.strip() else "Analyze and summarize this page."

    return (
        "The user provided a webpage link.\n\n"
        f"User request:\n{task}\n\n"
        f"Page URL:\n{url}\n\n"
        f"Page title:\n{title}\n\n"
        "Extracted page content:\n"
        f"{trim_text(content, MAX_PAGE_CHARS)}\n\n"
        "Use only the extracted page content above when answering. "
        "If the content appears incomplete or unclear, say so."
    )


def build_link_followup_prompt(user_question: str, page: dict) -> str:
    return (
        "The user is asking a follow-up question about a webpage that was analyzed earlier.\n\n"
        f"Follow-up question:\n{user_question}\n\n"
        f"Page URL:\n{page['url']}\n\n"
        f"Page title:\n{page['title']}\n\n"
        "Previously extracted page content:\n"
        f"{trim_text(page['content'], FOLLOWUP_MAX_PAGE_CHARS)}\n\n"
        "Answer the follow-up using the page content above. "
        "If the answer is not in the extracted content, say so clearly."
    )


def looks_like_followup_question(prompt: str) -> bool:
    p = prompt.strip().lower()

    if not p:
        return False

    followup_starters = (
        "what", "why", "how", "when", "where", "who", "which",
        "does", "did", "is", "are", "can", "could", "would",
        "tell me", "summarize", "explain", "compare", "list"
    )

    followup_keywords = (
        "the page", "that page", "this page", "the link", "that link",
        "this link", "the article", "that article", "this article",
        "the website", "that website", "this website", "the doc",
        "that doc", "this doc", "the documentation", "readme"
    )

    if p.endswith("?"):
        return True

    if p.startswith(followup_starters):
        return True

    if any(k in p for k in followup_keywords):
        return True

    return False


def upsert_saved_page(page: dict) -> None:
    for i, existing in enumerate(saved_pages):
        if existing["url"] == page["url"]:
            saved_pages[i] = page
            return
    saved_pages.append(page)


def print_help() -> None:
    print("""
Available Commands:

/help
  Show this help menu

/debug on
  Enable debug logs

/debug off
  Disable debug logs

/paste [instruction]
  Analyze clipboard image using the vision model
  Example: /paste describe this screenshot

/links
  List all stored analyzed links

/clearlink
  Clear all stored link contexts

exit
quit
  Exit the chatbot

Link Features:
- Paste a URL in your prompt to analyze it
- Follow-up questions will use the last analyzed page automatically
""")


def print_saved_links() -> None:
    if not saved_pages:
        print("[No saved links]\n")
        return

    print("\n[Saved Links]")
    for i, page in enumerate(saved_pages, 1):
        print(f"{i}. {page['title']}")
        print(f"   URL: {page['url']}")
        print(f"   Saved at: {page['saved_at']}\n")


history = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]


while True:
    prompt = input("You: ").strip()

    if prompt.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if prompt.lower() == "/help":
        print_help()
        print()
        continue

    if prompt.lower() == "/debug on":
        DEBUG_MODE = True
        print("[Debug mode enabled]\n")
        continue

    if prompt.lower() == "/debug off":
        DEBUG_MODE = False
        print("[Debug mode disabled]\n")
        continue

    if prompt.lower() == "/links":
        print_saved_links()
        continue

    if prompt.lower() == "/clearlink":
        last_page = None
        saved_pages.clear()
        print("[All stored link contexts cleared]\n")
        continue

    final_user_prompt = prompt
    router_input = prompt
    skip_router_search = False
    found_url = None
    used_link_followup = False
    

    # IMAGE PIPELINE
    if prompt.startswith("/paste"):
        instruction = extract_paste_instruction(prompt)
        skip_router_search = True  # image tasks should not use web search by default

        if ImageGrab is None:
            print("\n[Error] Pillow is not installed. Run: pip install pillow\n")
            continue

        img = grab_clipboard_image()
        if img is None:
            print("\n[Error] No image found in clipboard.\n")
            continue

        try:
            img_path = save_clipboard_image(img)
        except Exception as e:
            print(f"\n[Error] Failed to save clipboard image: {e}\n")
            continue

        try:
            vl_output = analyze_image_with_vl(img_path, instruction)
        except Exception as e:
            print(f"\n[Error] Vision model analysis failed: {e}\n")
            continue

        final_user_prompt = build_image_prompt_from_vl(
            instruction=instruction,
            vl_output=vl_output
        )
        debug_print("[FINAL PROMPT] Built prompt from VL pipeline")

        router_input = f"{instruction}\n\nExtracted image text:\n{vl_output}"

        

    else:
        # LINK PIPELINE
        found_url = extract_first_url(prompt)

        if found_url:
            user_instruction = remove_url_from_text(prompt, found_url)
            debug_print(f"[LINK] URL detected: {found_url}")

            try:
                link_start = time.perf_counter()
                page_title, page_content = fetch_url_content(found_url)
                link_end = time.perf_counter()

                last_page = {
                    "url": found_url,
                    "title": page_title,
                    "content": page_content,
                    "saved_at": datetime.now().isoformat(timespec="seconds")
                }
                upsert_saved_page(last_page)

                final_user_prompt = build_link_analysis_prompt(
                    user_instruction=user_instruction,
                    url=found_url,
                    title=page_title,
                    content=page_content
                )

                router_input = user_instruction if user_instruction else "analyze webpage content"
                skip_router_search = True

                debug_print(f"[LINK] Title: {page_title}")
                debug_print(f"[LINK] Extracted chars: {len(page_content)}")
                debug_print(f"[LINK TIME] {link_end - link_start:.2f} seconds")

            except Exception as e:
                print(f"\n[Error] Failed to analyze link: {e}\n")
                continue

        # FOLLOW-UP ON LAST LINK
        elif last_page and looks_like_followup_question(prompt):
            debug_print(f"[LINK FOLLOW-UP] Reusing stored page: {last_page['url']}")
            final_user_prompt = build_link_followup_prompt(prompt, last_page)
            router_input = prompt
            skip_router_search = True
            used_link_followup = True

    router_start = time.perf_counter()
    route = route_prompt_with_router(router_input)
    router_end = time.perf_counter()

    use_search = False if skip_router_search else route["search"]
    use_thinking = route["think"]

    if use_search:
        search_start = time.perf_counter()
        results = web_search_ollama(router_input, max_results=WEB_SEARCH_MAX_RESULTS)
        search_end = time.perf_counter()

        web_context = format_web_results(results)
        final_user_prompt = build_search_augmented_prompt(final_user_prompt, web_context)

        debug_print(f"[WEB SEARCH] Retrieved {len(results)} result(s)")
        debug_print(f"[WEB SEARCH TIME] {search_end - search_start:.2f} seconds")

    history.append({"role": "user", "content": final_user_prompt})

    route_label_parts = []

    if found_url:
        route_label_parts.append("LINK")
    elif used_link_followup:
        route_label_parts.append("LINK FOLLOW-UP")
    elif prompt.startswith("/paste"):
        route_label_parts.append("IMAGE")

    if use_search:
        route_label_parts.append("SEARCH")

    route_label_parts.append("THINK" if use_thinking else "NO THINK")
    route_label = " + ".join(route_label_parts)

    print(f"\n[Router decision: {route_label}]")
    print(f"[Router time: {router_end - router_start:.2f} seconds]")

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