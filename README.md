# Ollama Chatbot Experiments

A collection of standalone Python command-line chatbots exploring thinking-mode routing, character personas, clipboard images, OCR, web lookup and symbolic/numerical math. These are experimental scripts, with feature differences between variants.

## Start with basic chat

Use Python 3.10 or newer (the collection contains `X | None` annotations). From this repository folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install and start [Ollama](https://github.com/ollama/ollama-python#readme), then pull the model configured in the source:

```powershell
ollama pull jaahas/qwen3.5-uncensored:2b
.\.venv\Scripts\python.exe chatbot.py
```

Model names are copied from the original scripts; their availability and runtime compatibility were not tested. If a model is unavailable, edit the model constant in the chosen script to a model you have installed that supports the requested thinking/vision behavior. Type `exit` or `quit` to stop.

## Choose a variant

Run scripts directly from the repository root, e.g.:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/vision.txt
ollama pull qwen3-vl:2b
.\.venv\Scripts\python.exe examples/vision/clipboard_vision.py
```

| Script | Purpose | Dependency file |
| --- | --- | --- |
| `chatbot.py` | Basic chat with rule-based thinking | `requirements.txt` |
| `examples/personas/nahida.py` | Nahida persona with keyword thinking | `requirements.txt` |
| `examples/personas/kei.py` | Kei persona, OCR and router diagnostics | `requirements/ocr.txt` |
| `examples/personas/nahida_dynamic.py` | Nahida persona with model router and OCR | `requirements/ocr.txt` |
| `examples/vision/clipboard_ocr.py` | Clipboard OCR with model router | `requirements/ocr.txt` |
| `examples/vision/clipboard_vision.py` | Clipboard analysis using vision model | `requirements/vision.txt` |
| `examples/vision/clipboard_no_router.py` | Clipboard OCR without model router | `requirements/ocr.txt` |
| `examples/debug/router_debug.py` | Router streaming and timing diagnostics with OCR | `requirements/ocr.txt` |
| `examples/web/web_search.py` | Web search, URL analysis and clipboard vision | `requirements/web.txt` |
| `examples/math/symbolic_math.py` | SymPy math plus web and clipboard vision | `requirements/symbolic.txt` |
| `examples/math/math_pipeline.py` | Math normalization, SymPy, SciPy and OCR | `requirements/math-pipeline.txt` |

For each variant, install its dependency file with `python -m pip install -r <file>` using the virtual environment Python. Most variants use `jaahas/qwen3.5-uncensored:2b` for both the main model and router. `math_pipeline.py` additionally uses `qwen3.5:0.8b` as its router. Vision, web and symbolic-math variants use `qwen3-vl:2b` for images. Pull the extra models before using those features.

Dependency files are import-based starting points, not verified lockfiles. OCR needs a platform-appropriate PaddlePaddle installation in addition to the listed packages; follow the [official backend installation instructions](https://www.paddleocr.ai/main/en/version3.x/paddlepaddle_installation.html). OCR/model initialization may download additional weights. No model weights are included here.

## Commands and configuration

- All scripts: `exit`, `quit`.
- Clipboard variants: copy an image, then `/paste <instruction>`; clipboard support depends on the operating system. The original implementation targets desktop use.
- Advanced variants: `/debug on`, `/debug off`.
- Router debug and dynamic persona variants: `/routerdebug on`, `/routerdebug off`.
- Web and symbolic-math variants: `/help`, `/links`, `/clearlink`; paste a URL into a prompt to analyze it.

Web search reads `OLLAMA_API_KEY` from the process environment. Set that variable locally before starting a web-enabled script; never put the value in committed source. These scripts do not load `.env` files automatically. Search queries go to `https://ollama.com/api/web_search`, and URL analysis contacts the supplied website.

Model names, persona prompts and debug defaults are configured near the top of each script. There is no shared configuration layer. Conversation state is in memory. Clipboard images are written under `temp_images/` in the current working directory and are not automatically deleted; this directory is ignored by Git.

## Current limitations

All scripts start their input loop at import time, so run them directly rather than importing them. Live model, clipboard, OCR, network and solver behavior has not been validated in this cleanup. Math variants accept expressions through SymPy parsing functions; use trusted local input only, including OCR/model-derived expressions. These parsers are not a sandbox ([SymPy guidance](https://docs.sympy.org/latest/modules/core.html)). See [the cleanup and analysis report](CLEANUP_REPORT.md) for code references and follow-up work.

## Check the source

```powershell
python tools/check_syntax.py
```

This compiles source in memory without importing scripts, contacting models, installing packages or writing bytecode. It checks syntax only.


