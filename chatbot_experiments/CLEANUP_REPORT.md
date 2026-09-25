# Cleanup and analysis report

Prepared: 2026-09-25

## Result and scope

The original folder contained 12 files: 11 Python scripts and one feature note (6,071 Python source lines). There was no Git repository, README, dependency specification, test suite or license. All 12 original files are preserved unchanged in the parent folder, and all 12 are included in this organized copy. No duplicate files were found by SHA-256. No originals or variants were deleted. No Python application logic, prompts, model names or source formatting were changed.

The cleanup organizes variants into purpose-based directories, standardizes copied filenames, and adds the missing repository documentation and packaging metadata. `chatbot.py` remains the basic entry point. This is a repository-preparation pass and static code review, not a full security audit or runtime certification.

## Copied and renamed files

| Original file | Path in prepared repository | Content change |
| --- | --- | --- |
| `chatbot.py` | `chatbot.py` | Byte-for-byte unchanged |
| `Nahi chatbot.py` | `examples/personas/nahida.py` | Byte-for-byte unchanged |
| `Kei_chatbot.py` | `examples/personas/kei.py` | Byte-for-byte unchanged |
| `Nahi chatbot_dynamic thinking mode.py` | `examples/personas/nahida_dynamic.py` | Byte-for-byte unchanged |
| `chatbot_ssPasteFeature.py` | `examples/vision/clipboard_ocr.py` | Byte-for-byte unchanged |
| `chatbot_ssPasteFeatureUsingVLModel.py` | `examples/vision/clipboard_vision.py` | Byte-for-byte unchanged |
| `chatbot_withPasteFeature_noRouter.py` | `examples/vision/clipboard_no_router.py` | Byte-for-byte unchanged |
| `chatbot_router debug.py` | `examples/debug/router_debug.py` | Byte-for-byte unchanged |
| `chatbot_withSearchOnlineFeature.py` | `examples/web/web_search.py` | Byte-for-byte unchanged |
| `better math chatbot.py` | `examples/math/symbolic_math.py` | Byte-for-byte unchanged |
| `chatbot_faster math.py` | `examples/math/math_pipeline.py` | Byte-for-byte unchanged |
| `chatbot features.txt` | `docs/original_feature_notes.txt` | Byte-for-byte unchanged |

Relative image paths still resolve from the working directory; run variants from the repository root as documented. The scripts do not import each other, so the new filenames do not change internal import paths.

## Added files

- `README.md`: project overview, variant/dependency table, PowerShell setup, configured models, commands, runtime limitations and GitHub upload instructions.
- `.gitignore`: excludes environments, caches, generated screenshots, logs, common credential files and editor/OS files from future Git commits.
- `.gitattributes`: text normalization rules for future Git operations. It does not rewrite the copied files on disk.
- `requirements.txt`: minimal Ollama dependency.
- `requirements/vision.txt`, `requirements/web.txt`, `requirements/symbolic.txt`, `requirements/ocr.txt`, `requirements/math-pipeline.txt`: feature-specific dependencies inferred from imports. No exact versions were invented; the OCR major range reflects use of the `predict` API. These are not tested locks.
- `tools/check_syntax.py`: standalone syntax checker that avoids running application code.
- `docs/source_manifest.json`: every original-to-output mapping, file size and SHA-256 checksum for comparison.
- `docs/validation.json`: recorded static validation and preservation results.
- `CLEANUP_REPORT.md`: this report.

Removed files: **none**. Excluded original files: **none**. Source-code edits: **none**. There were no environments, caches, screenshots or build artifacts in the supplied folder to remove.

## Analysis findings

### Reproducibility and structure

1. **All 11 scripts run at import time.** Each has a module-level `while True` input loop and no entry-point guard. Importing them for tests or reuse blocks on input and can load dependencies. A later refactor should introduce `main()` and an `if __name__ == "__main__"` guard.
2. **Repeated implementations can drift.** Router, clipboard and streaming helpers recur across files. They are distinct experiments, not byte-identical duplicates, so none were discarded. A future consolidation could share helpers while preserving persona and feature differences.
3. **No reproducible environment was supplied.** Dependencies are now documented but were not installed or resolved. Test one desired variant in a clean virtual environment before creating a lockfile. OCR additionally requires a suitable PaddlePaddle backend.
4. **Feature notes are broader than an individual variant.** The original note mentions attachments; inspected scripts implement clipboard image handling, not a general attachment upload interface. The README describes actual per-script features instead of promising every feature everywhere.

### Input, privacy and robustness

5. **Math parsing is not a security boundary.** `examples/math/symbolic_math.py:538` calls `parse_expr` despite the helper name `safe_parse_expr`; bounds also use `sympify`. `examples/math/math_pipeline.py:687` and following call `parse_expr`, and line 840 uses `sympify` before numerical solving. SymPy documents evaluation-based parsing as unsafe for unsanitized input. This includes expressions extracted from images or generated by a model. Do not expose these paths to untrusted users as a service without redesigning expression validation/isolation. This cleanup leaves that implementation unchanged. [SymPy reference](https://docs.sympy.org/latest/modules/core.html).
6. **Screenshots persist locally.** Clipboard variants save PNGs under `temp_images/`, with no automatic cleanup. The new ignore rule prevents normal Git staging, but does not delete images or protect manual browser uploads. Debug mode defaults to enabled in advanced variants and can print extracted text.
7. **Web variants send data outside the machine.** Search uses the environment-provided `OLLAMA_API_KEY`; URL retrieval contacts the given site. The source uses environment lookup rather than a literal API key. No obvious hardcoded credentials, private keys, access tokens or user-specific absolute filesystem paths were found in the original text review; this is not a guarantee that prompts contain no personal information.
8. **URL analysis downloads each page twice.** `examples/web/web_search.py:357` uses `requests.get`, followed by `trafilatura.fetch_url` at line 361. `examples/math/symbolic_math.py:364` repeats this pattern. Extracting from the already-downloaded HTML could reduce latency and inconsistent results. URL fetching also lacks a host allowlist; a future hosted version would need explicit network access restrictions.
9. **Interactive robustness needs work.** For example, `chatbot.py:148` reads input and line 169 calls the model without a surrounding connection/EOF handler. Missing servers, unsupported models and interrupted input can terminate the program. Histories grow without an explicit cap.
10. **Licensing remains an owner decision.** No license was present and none was assigned. Character-specific roleplay prompts are retained verbatim; the repository does not include model weights.

## Validation and limits

All 11 original scripts parsed successfully under the available Python 3.10 interpreter. The copied scripts and added syntax checker compile without executing code. SHA-256 comparisons verify every copied original is byte-for-byte unchanged and every original still matches the inventory. `docs/validation.json` records the final checks, including dependency coverage and local documentation links.

No package installation, dependency vulnerability scan, model download, live inference, network search, clipboard access, OCR execution, solver correctness test or performance benchmark was run. No GitHub upload or Git initialization was performed. Model availability and a fully compatible package combination remain unverified.

## Suggested next work

Start by running `chatbot.py` in a clean virtual environment. Then select the advanced variants you want to support and verify their model and dependency combinations. Prioritize replacing unsafe math-input evaluation before accepting untrusted expressions, then add import-safe entry points, connection handling and focused behavior tests. Preserve the experimental variants until their differences are intentionally consolidated.
