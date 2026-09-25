"""

-Improved router
-Better OCR 

"""




import os
import ollama
import time
import json
import re
from datetime import datetime

try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None


ROUTER_MODEL = "jaahas/qwen3.5-uncensored:2b"
MAIN_MODEL = "jaahas/qwen3.5-uncensored:2b"
VL_MODEL = "qwen3-vl:2b"

SYSTEM_PROMPT = "You are a helpful AI assistant. Answer clearly, directly, and naturally."

DEBUG_MODE = True
TEMP_IMAGE_DIR = "temp_images"


def debug_print(msg: str) -> None:
    if DEBUG_MODE:
        print(msg)


def ensure_temp_dir() -> None:
    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)


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


def should_think_with_router(prompt: str) -> bool:
    simple_prompts = {
        "hi", "hello", "hey", "thanks", "thank you",
        "good morning", "good evening", "good afternoon"
    }
    if prompt.lower().strip() in simple_prompts:
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

    user_instruction = instruction.strip() if instruction.strip() else "Describe the image clearly and accurately."

    vl_messages = [
        {
            "role": "system",
            "content": (
                "You are an image analysis model.\n"
                "Read the image directly.\n"
                "Do NOT mention OCR.\n"
                "Do NOT guess missing details.\n"
                "If something is unclear, say it is unclear.\n\n"
                "When the image contains math, equations, code, terminal text, tables, UI, or screenshots, "
                "transcribe and describe them as accurately as possible.\n"
                "Preserve important symbols, numbers, operators, variable names, filenames, and structure.\n"
                "If the user asks to solve or analyze, first identify the visible content faithfully."
            )
        },
        {
            "role": "user",
            "content": (
                f"User instruction:\n{user_instruction}\n\n"
                "Please inspect the attached image and provide the most accurate usable reading/analysis of its contents."
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


def build_image_prompt_from_vl(instruction: str, vl_output: str, image_path: str) -> str:
    analyzed = vl_output.strip() if vl_output.strip() else "[The vision model could not extract useful content from the image.]"

    return (
        f"The user sent a screenshot through the clipboard.\n"
        f"Image path: {image_path}\n\n"
        f"User instruction:\n{instruction if instruction else '[No instruction provided]'}\n\n"
        f"Vision model analysis of the image:\n{analyzed}\n\n"
        f"Use the vision model analysis above as the image reading result. "
        f"Help the user based on their instruction. "
        f"If parts of the image may be unclear or uncertain, say so clearly."
    )


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

    if prompt.lower() == "/debug on":
        DEBUG_MODE = True
        print("[Debug mode enabled]\n")
        continue

    if prompt.lower() == "/debug off":
        DEBUG_MODE = False
        print("[Debug mode disabled]\n")
        continue

    final_user_prompt = prompt
    router_input = prompt

    if prompt.startswith("/paste"):
        instruction = extract_paste_instruction(prompt)

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
            vl_output=vl_output,
            image_path=img_path
        )
        debug_print("[FINAL PROMPT] Built prompt from VL pipeline")

        router_input = instruction if instruction else "describe the image"

    router_start = time.perf_counter()
    use_thinking = should_think_with_router(router_input)
    router_end = time.perf_counter()

    history.append({"role": "user", "content": final_user_prompt})

    print(f"\n[Router decision: {'THINK' if use_thinking else 'NO THINK'}]")
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