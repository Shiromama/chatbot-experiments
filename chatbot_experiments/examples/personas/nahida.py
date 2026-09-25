import ollama
import time

def should_think(prompt: str) -> bool:
    prompt = prompt.lower()

    keywords = [
        "solve", "calculate", "compute", "derive", "integrate", "differentiate",
        "limit", "matrix", "determinant", "equation", "proof", "analyze",
        "explain deeply", "step by step", "debug", "error", "why"
    ]

    return any(keyword in prompt for keyword in keywords)

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

history = [
    {
        "role": "system",
        "content": nahida_prompt
    }
]

model = "jaahas/qwen3.5-uncensored:2b"

while True:
    prompt = input("You: ").strip()

    if prompt.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    use_thinking = should_think(prompt)

    history.append({"role": "user", "content": prompt})

    print(f"\n[Thinking mode: {'ON' if use_thinking else 'OFF'}]")

    start_time = time.perf_counter()

    stream = ollama.chat(
        model=model,
        messages=history,
        think=use_thinking,
        stream=True
    )

    full_thinking = ""
    full_answer = ""

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
                print("\n\nNahida:")
            print(content_part, end="", flush=True)
            full_answer += content_part

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print(f"\n\n[Done in {elapsed:.2f} seconds]\n")

    history.append({"role": "assistant", "content": full_answer})