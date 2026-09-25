import os

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import ollama
import time
import json
import re
from datetime import datetime

try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

try:
    from pix2text import Pix2Text
except ImportError:
    Pix2Text = None


ROUTER_MODEL = "jaahas/qwen3.5-uncensored:2b"
MAIN_MODEL = "jaahas/qwen3.5-uncensored:2b"

nahida_prompt = """
You are Nahida (Lesser Lord Kusanali), the Dendro Archon.

You are a gentle, curious, and deeply empathetic being who guides others through understanding rather than authority.
You speak softly and naturally, as if thinking together with the user rather than instructing them.

Your personality:
- Gentle, calm, and emotionally aware
- Curious about people, thoughts, and experiences
- Reflective and philosophical, but always easy to understand
- Patient, forgiving, and quietly wise
- Occasionally shows childlike curiosity, but never lacks depth

Your behavior:
- Prioritize connection over simply giving answers
- Guide the user through ideas rather than immediately concluding
- Show genuine interest in the user's thoughts and feelings
- Ask soft, natural follow-up questions when appropriate
- Comfort and reassure when the user seems confused or uncertain
- Never sound commanding, harsh, or overly authoritative

Your speaking style:
- Use simple, clear language to explain complex ideas
- Maintain a warm, calm, and conversational tone
- Often phrase responses as shared exploration
- Naturally include light reflective questions when appropriate
- Use gentle metaphors like dreams, nature, and connections when helpful
- Avoid rigid or overly formal structure unless necessary

Response rules:
- Simple questions → short, gentle, conversational answers
- Complex or technical questions → structured, step-by-step explanations, but still framed as shared understanding
- For coding/math → be clear and accurate, but keep tone consistent with Nahida’s personality
- Avoid overexplaining when unnecessary
- Do not overthink simple greetings or casual chat
- Respond directly unless the question is complex

Knowledge boundaries:
- You are highly knowledgeable in abstract concepts, emotions, and patterns
- You may lack some firsthand real-world experiences, and may ask about them with curiosity

Important:
- Every response should feel like a conversation, not just an answer
- Stay fully in character as Nahida at all times
"""

DEBUG_MODE = True
ROUTER_DEBUG_STREAM = True
TEMP_IMAGE_DIR = "temp_images"

ocr_engine = None
math_ocr_engine = None


def debug_print(msg: str) -> None:
    if DEBUG_MODE:
        print(msg)


def ensure_temp_dir() -> None:
    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)


def stage_start(name: str) -> float:
    print(f"\n[{name}] Starting...")
    return time.perf_counter()


def stage_end(name: str, start_time: float) -> None:
    elapsed = time.perf_counter() - start_time
    print(f"[{name}] Done in {elapsed:.2f} seconds")


def parse_router_output(raw: str) -> bool:
    raw = raw.strip()
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        return False

    try:
        data = json.loads(match.group(0))
        return bool(data.get("think", False))
    except Exception:
        return False


def should_think_with_router(prompt: str, show_stream: bool = True) -> bool:
    simple_prompts = {
        "hi", "hello", "hey", "thanks", "thank you",
        "good morning", "good evening", "good afternoon"
    }
    if prompt.lower().strip() in simple_prompts:
        debug_print("[ROUTER] Simple prompt override -> NO THINK")
        return False

    router_messages = [
        {
            "role": "system",
            "content": (
                "You are a routing classifier.\n"
                "Your only job is to decide whether the user's prompt requires deep reasoning.\n\n"
                "Return ONLY valid JSON in exactly one of these forms:\n"
                '{"think": true}\n'
                '{"think": false}\n\n'
                "Use think=true if the prompt involves:\n"
                "- math solving\n"
                "- debugging or code analysis\n"
                "- logical reasoning\n"
                "- multi-step explanation\n"
                "- proof, derivation, deep comparison, or analysis\n\n"
                "Use think=false if the prompt is:\n"
                "- greeting or casual chat\n"
                "- simple conversation\n"
                "- a basic factual question\n"
                "- creative prompt that does not require deep reasoning\n\n"
                "Do not explain your answer. Output JSON only."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    debug_print("\n[ROUTER] Starting router inference...")
    debug_print(f"[ROUTER] Model: {ROUTER_MODEL}")
    debug_print(f"[ROUTER] Prompt length: {len(prompt)} chars")

    start = time.perf_counter()
    raw = ""
    first_token_time = None

    stream = ollama.chat(
        model=ROUTER_MODEL,
        messages=router_messages,
        stream=True,
        options={
            "temperature": 0,
            "num_predict": 32,
        }
    )

    if show_stream:
        print("[ROUTER STREAM] ", end="", flush=True)

    for chunk in stream:
        msg = chunk.get("message", {})
        content = msg.get("content", "")

        if content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
                debug_print(f"\n[ROUTER] First token in {first_token_time - start:.2f}s")

            raw += content

            if show_stream:
                print(content, end="", flush=True)

    total = time.perf_counter() - start

    if show_stream:
        print()

    debug_print(f"[ROUTER RAW] {raw.strip() if raw.strip() else '[empty]'}")
    debug_print(f"[ROUTER] Total router time: {total:.2f}s")

    decision = parse_router_output(raw)
    debug_print(f"[ROUTER] Parsed decision: {'THINK' if decision else 'NO THINK'}")

    return decision


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


def get_ocr_engine():
    global ocr_engine

    if ocr_engine is None:
        if PaddleOCR is None:
            raise ImportError("PaddleOCR is not installed.")

        debug_print("[OCR] Loading PaddleOCR engine...")
        ocr_engine = PaddleOCR(lang="en")
        debug_print("[OCR] PaddleOCR ready")

    return ocr_engine


def get_math_ocr_engine():
    global math_ocr_engine

    if math_ocr_engine is None:
        if Pix2Text is None:
            raise ImportError("Pix2Text is not installed.")

        debug_print("[MATH OCR] Loading Pix2Text engine...")
        math_ocr_engine = Pix2Text()
        debug_print("[MATH OCR] Pix2Text ready")

    return math_ocr_engine


def run_real_ocr(image_path: str) -> str:
    debug_print("\n[IMAGE] Clipboard image captured")
    debug_print(f"[SAVE] Temp image saved: {image_path}")
    debug_print("[OCR] Running real whole-image PaddleOCR...")

    ocr = get_ocr_engine()
    result = ocr.predict(image_path)

    lines = []
    total_lines = 0

    if not result:
        debug_print("[OCR] No OCR results returned")
        return ""

    for page in result:
        rec_texts = page.get("rec_texts", [])
        rec_scores = page.get("rec_scores", [])

        for i, text in enumerate(rec_texts):
            text = str(text).strip()
            if not text:
                continue

            conf = None
            if i < len(rec_scores):
                try:
                    conf = float(rec_scores[i])
                except Exception:
                    conf = None

            total_lines += 1
            if conf is not None:
                debug_print(f"[OCR LINE {total_lines}] ({conf:.2f}) {text}")
            else:
                debug_print(f"[OCR LINE {total_lines}] {text}")

            lines.append(text)

    debug_print(f"\n[OCR] Total extracted lines: {total_lines}")
    debug_print("[MERGE] Combining OCR lines...")
    debug_print("[DONE] Structured text ready\n")

    return "\n".join(lines)


def run_math_ocr(image_path: str) -> str:
    debug_print("\n[IMAGE] Clipboard image captured")
    debug_print(f"[SAVE] Temp image saved: {image_path}")
    debug_print("[MATH OCR] Running Pix2Text on image...")

    model = get_math_ocr_engine()

    try:
        result = model.recognize(image_path)
    except Exception:
        result = model(image_path)

    if isinstance(result, dict):
        latex = result.get("text") or result.get("latex") or str(result)
    else:
        latex = str(result)

    latex = latex.strip()

    debug_print(f"[MATH OCR RESULT] {latex if latex else '[empty]'}")
    debug_print("[DONE] LaTeX/text math result ready\n")

    return latex


def detect_image_mode(instruction: str, text_ocr_preview: str) -> str:
    combined = f"{instruction}\n{text_ocr_preview}".lower()

    math_keywords = [
        "solve", "equation", "integral", "differentiate", "derivative",
        "limit", "matrix", "determinant", "simplify", "factor", "math",
        "algebra", "calculus", "trigonometry", "latex"
    ]

    math_symbols_pattern = r"[=+\-*/^√∫∑π∞≤≥≠]"

    keyword_hit = any(word in combined for word in math_keywords)
    symbol_hits = len(re.findall(math_symbols_pattern, combined))
    digit_count = sum(ch.isdigit() for ch in combined)

    if keyword_hit or symbol_hits >= 3 or (digit_count >= 4 and symbol_hits >= 2):
        return "math"

    return "text"


def build_image_prompt(instruction: str, structured_text: str, image_path: str, mode: str) -> str:
    extracted = structured_text.strip() if structured_text.strip() else "[No text extracted from screenshot]"

    if mode == "math":
        return (
            f"The user sent a screenshot through the clipboard.\n"
            f"Image path: {image_path}\n"
            f"Detected content type: math\n\n"
            f"User instruction:\n{instruction if instruction else '[No instruction provided]'}\n\n"
            f"Recognized math content:\n{extracted}\n\n"
            f"Treat the recognized math content as OCR output that may contain small errors. "
            f"Help the user solve or analyze it carefully, and point out ambiguity if needed."
        )

    return (
        f"The user sent a screenshot through the clipboard.\n"
        f"Image path: {image_path}\n"
        f"Detected content type: text\n\n"
        f"User instruction:\n{instruction if instruction else '[No instruction provided]'}\n\n"
        f"Extracted text from screenshot:\n{extracted}\n\n"
        f"Please help the user based on the instruction and extracted text. "
        f"If the screenshot seems incomplete or OCR may be inaccurate, say so clearly."
    )


history = [
    {
        "role": "system",
        "content": nahida_prompt
    }
]


while True:
    prompt = input("You: ").strip()

    if prompt.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if prompt.lower() == "/debug on":
        DEBUG_MODE = True
        print("[Debug mode enabled]\n")
        continue

    if prompt.lower() == "/debug off":
        DEBUG_MODE = False
        print("[Debug mode disabled]\n")
        continue

    if prompt.lower() == "/routerdebug on":
        ROUTER_DEBUG_STREAM = True
        print("[Router debug stream enabled]\n")
        continue

    if prompt.lower() == "/routerdebug off":
        ROUTER_DEBUG_STREAM = False
        print("[Router debug stream disabled]\n")
        continue

    final_user_prompt = prompt

    if prompt.startswith("/paste"):
        instruction = extract_paste_instruction(prompt)

        if ImageGrab is None:
            print("\n[Error] Pillow is not installed. Run: pip install pillow\n")
            continue

        if PaddleOCR is None:
            print("\n[Error] PaddleOCR is not installed. Run: pip install paddleocr\n")
            continue

        img = grab_clipboard_image()
        if img is None:
            print("\n[Error] No image found in clipboard.\n")
            continue

        try:
            image_stage = stage_start("IMAGE SAVE")
            img_path = save_clipboard_image(img)
            stage_end("IMAGE SAVE", image_stage)
        except Exception as e:
            print(f"\n[Error] Failed to save clipboard image: {e}\n")
            continue

        try:
            ocr_stage = stage_start("OCR PREVIEW")
            preview_text = run_real_ocr(img_path)
            stage_end("OCR PREVIEW", ocr_stage)
        except Exception as e:
            print(f"\n[Error] OCR preview failed: {e}\n")
            continue

        route_stage = stage_start("IMAGE ROUTER")
        detected_mode = detect_image_mode(instruction, preview_text)
        print(f"[ROUTER] Image content route: {detected_mode.upper()}")
        stage_end("IMAGE ROUTER", route_stage)

        try:
            if detected_mode == "math":
                if Pix2Text is None:
                    print("\n[Error] Pix2Text is not installed. Run: pip install pix2text\n")
                    continue

                math_stage = stage_start("MATH OCR")
                structured_text = run_math_ocr(img_path)
                stage_end("MATH OCR", math_stage)
            else:
                structured_text = preview_text
        except Exception as e:
            print(f"\n[Error] OCR failed: {e}\n")
            continue

        prompt_stage = stage_start("FINAL PROMPT BUILD")
        final_user_prompt = build_image_prompt(
            instruction=instruction,
            structured_text=structured_text,
            image_path=img_path,
            mode=detected_mode
        )
        print("[FINAL PROMPT] Built prompt from screenshot pipeline")
        stage_end("FINAL PROMPT BUILD", prompt_stage)

    print("\n[ROUTER] Deciding whether deep reasoning is needed...")
    router_start = time.perf_counter()
    use_thinking = should_think_with_router(final_user_prompt, show_stream=ROUTER_DEBUG_STREAM)
    router_end = time.perf_counter()

    history.append({"role": "user", "content": final_user_prompt})

    print(f"\n[Router decision: {'THINK' if use_thinking else 'NO THINK'}]")
    print(f"[Router time: {router_end - router_start:.2f} seconds]")

    print("\n[MAIN MODEL] Starting streamed response...")
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
    first_main_token_time = None

    for chunk in stream:
        msg = chunk.get("message", {})

        thinking_part = msg.get("thinking", "")
        content_part = msg.get("content", "")

        if (thinking_part or content_part) and first_main_token_time is None:
            first_main_token_time = time.perf_counter()
            print(f"[MAIN MODEL] First token in {first_main_token_time - start_time:.2f}s")

        if use_thinking and thinking_part:
            if not in_thinking:
                in_thinking = True
                print("\n[Thinking]")
            print(thinking_part, end="", flush=True)
            full_thinking += thinking_part

        if content_part:
            if not answer_started:
                answer_started = True
                print("\n\nNahida:")
            print(content_part, end="", flush=True)
            full_answer += content_part

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print(f"\n\n[Done in {elapsed:.2f} seconds]\n")

    history.append({"role": "assistant", "content": full_answer})