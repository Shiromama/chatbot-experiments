import os

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import ollama
import time
import json
import re
from datetime import datetime
from typing import Dict, Any, Tuple

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


ROUTER_MODEL = "qwen3.5:0.8b"
MAIN_MODEL = "jaahas/qwen3.5-uncensored:2b"

SYSTEM_PROMPT = "You are a helpful AI assistant. Answer clearly, directly, and naturally."

DEBUG_MODE = True
TEMP_IMAGE_DIR = "temp_images"

ocr_engine = None
math_ocr_engine = None


# ============================================================
# General helpers
# ============================================================

def debug_print(msg: str) -> None:
    if DEBUG_MODE:
        print(msg)


def ensure_temp_dir() -> None:
    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)


# ============================================================
# Rule-based normalizer
# ============================================================

def strip_problem_leadin(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = re.sub(r'\$([^$]+)\$', r'\1', cleaned)
    cleaned = cleaned.strip('"\'`')

    leadin_patterns = [
        r'^what\s+is\s+the\s+solution\s+to\s+(this\s+)?differential\s+equation\??\s*',
        r'^solve\s+(the\s+)?(following\s+)?differential\s+equation\s*:?\s*',
        r'^solve\s+(for\s+\w+\s*:?\s*)?',
        r'^find\s+(the\s+)?solution\s*(to|of)?\s*',
        r'^determine\s+(the\s+)?solution\s*(to|of)?\s*',
        r'^evaluate\s*',
        r'^simplify\s*',
    ]

    for pattern in leadin_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    if '?' in cleaned:
        parts = [part.strip() for part in cleaned.split('?') if part.strip()]
        if parts:
            cleaned = max(parts, key=lambda s: sum(ch.isalnum() for ch in s))

    return cleaned.strip(' ,:;')


def normalize_latexish_math(text: str) -> str:
    if not text:
        return ""

    cleaned = text
    cleaned = cleaned.replace(r'\left', '').replace(r'\right', '')
    cleaned = cleaned.replace(r'\,', ' ')
    cleaned = re.sub(r'\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'(\1)/(\2)', cleaned)
    cleaned = re.sub(r'e\s*\^\s*\{([^{}]+)\}', r'exp(\1)', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'e\s*\^\s*\(([^()]+)\)', r'exp(\1)', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\^\s*\{([^{}]+)\}', r'^(\1)', cleaned)
    cleaned = cleaned.replace('{', '(').replace('}', ')')
    cleaned = re.sub(r'\\cdot|\\times', '*', cleaned)
    return cleaned


def insert_implicit_multiplication(text: str) -> str:
    if not text:
        return ""

    cleaned = text
    cleaned = re.sub(r'(\d)([A-Za-z(])', r'\1*\2', cleaned)
    cleaned = re.sub(r'([)])(\d)', r'\1*\2', cleaned)
    cleaned = re.sub(r'\)(\()', r')*(', cleaned)
    cleaned = re.sub(r'([xyz])\(', r'\1*(', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\)([A-Za-z])', r')*\1', cleaned)
    cleaned = re.sub(r'(?<![A-Za-z0-9_*])dx', '1*dx', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(?<![A-Za-z0-9_*])dy', '1*dy', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(exp\([^\)]*\))(dx|dy)', r'\1*\2', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'([A-Za-z0-9_\)]+)(dx|dy)', r'\1*\2', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r'exp\(([^()]*)\)',
        lambda m: 'exp(' + re.sub(r'(?<=\d)\s+(?=[A-Za-z])', '*', m.group(1).strip()) + ')',
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


def contains_mixed_english_math(text: str) -> bool:
    if not text:
        return False

    english_words = re.findall(r'[A-Za-z]{4,}', text)
    math_tokens = re.findall(r'(?:dy/dx|d/dx|dx|dy|=|\+|\-|\*|/|\^|\(|\)|exp\(|\d)', text, flags=re.IGNORECASE)
    return len(english_words) >= 3 and len(math_tokens) >= 3


def looks_like_differential_form(text: str) -> bool:
    if not text:
        return False

    t = text.lower()
    return (
        ('dx' in t and 'dy' in t and '=' in t)
        or bool(re.search(r'[^=]*dx[+\-].*dy=0', t))
        or bool(re.search(r'[^=]*dy[+\-].*dx=0', t))
    )


def clean_math_ocr_text(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()
    replacements = {
        "−": "-",
        "—": "-",
        "–": "-",
        "÷": "/",
        "×": "*",
        "∗": "*",
        "·": "*",
        "^ ": "^",
        " =": "=",
        "= ": "=",
        "′": "'",
    }

    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)

    cleaned = strip_problem_leadin(cleaned)
    cleaned = normalize_latexish_math(cleaned)
    cleaned = re.sub(r'd\s*2\s*y\s*/\s*d\s*x\s*2', 'd2y/dx2', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'd\s*y\s*/\s*d\s*x', 'dy/dx', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'd\s*x\s*/\s*d\s*y', 'dx/dy', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'dy\s*dx', 'dy/dx', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'd\s*/\s*d\s*x', 'd/dx', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'd\s*x\b', 'dx', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'd\s*y\b', 'dy', cleaned, flags=re.IGNORECASE)
    cleaned = insert_implicit_multiplication(cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\s*([=+\-*/^(),])\s*', r'\1', cleaned)
    return cleaned.strip()


def detect_task(text: str) -> Tuple[str, float]:
    t = text.lower()
    task_keywords = {
        "solve": ["solve", "find", "compute", "determine"],
        "differentiate": ["differentiate", "derivative"],
        "integrate": ["integrate", "integral", "antiderivative"],
        "simplify": ["simplify"],
        "evaluate": ["evaluate"],
    }

    for task, keywords in task_keywords.items():
        for keyword in keywords:
            if keyword in t:
                return task, 0.9

    if any(sym in text for sym in ["dy/dx", "d/dx", "∫", "="]):
        return "solve", 0.6

    return "unknown", 0.35


def detect_equation_type(text: str) -> Tuple[str, float]:
    t = text.lower()

    if "dy/dx" in t or "d2y/dx2" in t or looks_like_differential_form(t):
        return "ode", 0.95

    if "differential equation" in t and ("dx" in t or "dy" in t):
        return "ode", 0.9

    if "d/dx" in t and "=" not in t:
        return "derivative", 0.9

    if "∫" in text or " integral " in f" {t} ":
        return "integral", 0.9

    if "matrix" in t or ("[" in text and "]" in text):
        return "matrix", 0.75

    if "=" in text:
        return "algebra", 0.7

    return "unknown", 0.3


def detect_symbolic_vs_numeric(text: str) -> Tuple[str, float]:
    has_letters = bool(re.search(r'[a-zA-Z]', text))
    has_numbers = bool(re.search(r'\d', text))

    numerical_keywords = [
        "approx", "approximate", "decimal", "numerical",
        "use euler", "rk4", "runge-kutta", "to 3 decimal", "to 2 decimal"
    ]
    tl = text.lower()

    if any(k in tl for k in numerical_keywords):
        return "numeric", 0.9

    if has_letters:
        return "symbolic", 0.8

    if has_numbers and re.fullmatch(r'[\d\.\s\+\-\*/=(),]+', text):
        return "numeric", 0.85

    return "unknown", 0.45


def extract_initial_condition(text: str) -> Tuple[bool, str, float]:
    patterns = [
        r'y\s*\(\s*[^\)]+\s*\)\s*=\s*[^\s,;]+',
        r'y\s*\(\s*[^\)]+\s*\)\s*=\s*.+?(?=$|,|;)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return True, match.group(0).strip(), 0.95

    return False, "", 0.0


def extract_equation(text: str, initial_condition_raw: str = "") -> str:
    candidate = text.strip()

    if initial_condition_raw:
        candidate = candidate.replace(initial_condition_raw, "").strip(" ,;")

    if re.search(r'=', candidate):
        match = re.search(r'([^\n]*=.*)', candidate)
        if match:
            return match.group(1).strip(" ,;")

    return candidate


def normalize_math_problem(text: str) -> Dict[str, Any]:
    cleaned = clean_math_ocr_text(text)

    task, task_conf = detect_task(cleaned)
    eq_type, type_conf = detect_equation_type(cleaned)
    nature, nature_conf = detect_symbolic_vs_numeric(cleaned)
    has_ic, ic_raw, ic_conf = extract_initial_condition(cleaned)
    equation = extract_equation(cleaned, ic_raw)

    if looks_like_differential_form(cleaned):
        eq_type = "ode"
        type_conf = max(type_conf, 0.95)

    equation_conf = 0.9 if equation else 0.25
    ic_component = ic_conf if has_ic else 0.5

    confidence = (
        task_conf * 0.2 +
        type_conf * 0.3 +
        nature_conf * 0.2 +
        ic_component * 0.1 +
        equation_conf * 0.2
    )

    fallback_reasons = []

    if contains_mixed_english_math(text):
        confidence -= 0.18
        fallback_reasons.append("mixed_english_and_math")

    if '$' in text:
        confidence -= 0.08
        fallback_reasons.append("latex_wrapper_detected")

    if re.match(r'^(what|solve|find|determine)', text.strip(), flags=re.IGNORECASE):
        confidence -= 0.08
        fallback_reasons.append("question_leadin_detected")

    if eq_type == "algebra" and re.search(r'dx|dy', cleaned, flags=re.IGNORECASE):
        confidence -= 0.20
        fallback_reasons.append("algebra_but_has_dx_dy")

    confidence = max(0.0, min(1.0, confidence))
    needs_llm_fallback = confidence < 0.70 or eq_type == "unknown" or bool(fallback_reasons)

    return {
        "task": task,
        "type": eq_type,
        "equation_raw": equation,
        "has_initial_condition": has_ic,
        "initial_condition_raw": ic_raw,
        "nature": nature,
        "confidence": round(confidence, 3),
        "needs_llm_fallback": needs_llm_fallback,
        "cleaned_text": cleaned,
        "fallback_reasons": fallback_reasons,
    }


# ============================================================
# LLM fallback normalizer
# ============================================================

def extract_first_json_object(raw: str) -> Dict[str, Any] | None:
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    except Exception:
        return None

    return None


def coerce_llm_normalization(data: Dict[str, Any], original_text: str) -> Dict[str, Any]:
    cleaned_text = clean_math_ocr_text(str(data.get("cleaned_text") or original_text or ""))
    equation_raw = str(data.get("equation_raw") or "").strip()
    initial_condition_raw = str(data.get("initial_condition_raw") or "").strip()

    if not equation_raw:
        equation_raw = extract_equation(cleaned_text, initial_condition_raw)

    has_ic = bool(data.get("has_initial_condition", False))
    if not initial_condition_raw and has_ic:
        has_ic2, ic_raw2, _ = extract_initial_condition(cleaned_text)
        if has_ic2:
            initial_condition_raw = ic_raw2
    elif initial_condition_raw and not has_ic:
        has_ic = True

    task = str(data.get("task") or "unknown").strip().lower()
    eq_type = str(data.get("type") or "unknown").strip().lower()
    nature = str(data.get("nature") or "unknown").strip().lower()

    allowed_tasks = {"solve", "differentiate", "integrate", "simplify", "evaluate", "unknown"}
    allowed_types = {"ode", "derivative", "integral", "matrix", "algebra", "unknown"}
    allowed_natures = {"symbolic", "numeric", "unknown"}

    if task not in allowed_tasks:
        task = "unknown"
    if eq_type not in allowed_types:
        eq_type = "unknown"
    if nature not in allowed_natures:
        nature = "unknown"

    confidence = data.get("confidence", 0.55)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.55

    confidence = max(0.0, min(1.0, confidence))

    return {
        "task": task,
        "type": eq_type,
        "equation_raw": equation_raw,
        "has_initial_condition": has_ic,
        "initial_condition_raw": initial_condition_raw,
        "nature": nature,
        "confidence": round(confidence, 3),
        "needs_llm_fallback": False,
        "cleaned_text": cleaned_text,
        "normalizer_source": "llm",
    }


def normalize_math_problem_with_llm(text: str) -> Dict[str, Any]:
    cleaned = clean_math_ocr_text(text)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a math OCR normalization engine.\n"
                "Your task is to reconstruct messy OCR math into structured JSON.\n\n"
                "Return ONLY valid JSON. No markdown. No explanation.\n\n"
                "Use exactly this schema:\n"
                "{\n"
                '  "task": "solve|differentiate|integrate|simplify|evaluate|unknown",\n'
                '  "type": "ode|derivative|integral|matrix|algebra|unknown",\n'
                '  "equation_raw": "string",\n'
                '  "has_initial_condition": true,\n'
                '  "initial_condition_raw": "string",\n'
                '  "nature": "symbolic|numeric|unknown",\n'
                '  "confidence": 0.0,\n'
                '  "cleaned_text": "string"\n'
                "}\n\n"
                "Rules:\n"
                "- Reconstruct likely math symbols when OCR spacing is broken.\n"
                "- Preserve ambiguity conservatively.\n"
                "- If unsure, use unknown instead of guessing too hard.\n"
                "- confidence must be between 0 and 1.\n"
                "- Do not include extra keys.\n"
            )
        },
        {
            "role": "user",
            "content": (
                "Normalize this OCR math text into JSON:\n\n"
                f"{cleaned}"
            )
        }
    ]

    response = ollama.chat(
        model=ROUTER_MODEL,
        messages=messages,
        stream=False
    )

    raw = response["message"]["content"]
    parsed = extract_first_json_object(raw)

    if not parsed:
        raise ValueError("LLM fallback normalizer did not return valid JSON")

    return coerce_llm_normalization(parsed, cleaned)


def run_full_math_normalization(text: str) -> Dict[str, Any]:
    rule_result = normalize_math_problem(text)
    rule_result["normalizer_source"] = "rule"

    broken_text = not rule_result.get("equation_raw") or len(rule_result.get("cleaned_text", "")) < 3

    should_fallback = (
        rule_result.get("needs_llm_fallback", False)
        or rule_result.get("type") == "unknown"
        or broken_text
    )

    if not should_fallback:
        return rule_result

    debug_print("[NORMALIZER] Rule-based confidence low or structure unclear -> using LLM fallback")

    try:
        llm_result = normalize_math_problem_with_llm(text)
        llm_result["fallback_used"] = True
        llm_result["rule_confidence"] = rule_result.get("confidence")
        return llm_result
    except Exception as e:
        debug_print(f"[NORMALIZER] LLM fallback failed: {e}")
        rule_result["fallback_used"] = False
        rule_result["fallback_error"] = str(e)
        return rule_result


# ============================================================
# Math router
# ============================================================

def route_math_problem(normalized: Dict[str, Any]) -> Dict[str, Any]:
    eq_type = str(normalized.get("type", "unknown")).lower()
    task = str(normalized.get("task", "unknown")).lower()
    nature = str(normalized.get("nature", "unknown")).lower()
    has_ic = bool(normalized.get("has_initial_condition", False))
    equation_raw = str(normalized.get("equation_raw", "") or "")
    confidence = float(normalized.get("confidence", 0.0) or 0.0)
    cleaned = str(normalized.get("cleaned_text", "") or "")

    numerical_keywords = [
        "approx", "approximate", "numerical", "decimal",
        "euler", "rk4", "runge-kutta", "iteration", "interpolate",
        "optimization", "optimize", "minimize", "maximize"
    ]
    requests_numeric = any(k in cleaned.lower() for k in numerical_keywords)

    route = "llm"
    reason = "Unknown or ambiguous math structure"
    solver_mode = "explain_only"

    if confidence < 0.45:
        route = "llm"
        reason = "Normalization confidence too low"
        solver_mode = "explain_only"
    elif eq_type == "ode":
        if has_ic or nature == "numeric" or requests_numeric:
            route = "scipy"
            reason = "ODE with initial condition or numerical clue"
            solver_mode = "numeric"
        else:
            route = "sympy"
            reason = "Symbolic ODE without numerical requirement"
            solver_mode = "symbolic"
    elif eq_type in {"derivative", "integral", "algebra", "matrix"}:
        if nature == "numeric" and requests_numeric and eq_type not in {"matrix"}:
            route = "scipy"
            reason = "Numerical approximation requested"
            solver_mode = "numeric"
        else:
            route = "sympy"
            reason = f"{eq_type.capitalize()} is best handled symbolically"
            solver_mode = "symbolic"
    elif task in {"differentiate", "integrate", "simplify"}:
        route = "sympy"
        reason = f"Task '{task}' is symbolic by default"
        solver_mode = "symbolic"
    elif equation_raw and "=" in equation_raw:
        route = "sympy"
        reason = "Equation detected and symbolic solve is the safest default"
        solver_mode = "symbolic"

    return {
        "route": route,
        "solver_mode": solver_mode,
        "reason": reason,
        "ready_for_solver": route in {"sympy", "scipy"},
    }


# ============================================================
# SymPy solver
# ============================================================

def extract_expression_for_symbolic_work(expr: str) -> str:
    if not expr:
        return ""

    cleaned = expr.strip()
    cleaned = re.sub(
        r'^(solve|find|compute|determine|differentiate|derivative|integrate|integral|antiderivative|simplify|evaluate)\s+',
        '',
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r'^(of|the)\s+', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip(' :')


def convert_basic_math_to_sympy(expr: str) -> str:
    if not expr:
        return ""

    converted = expr.strip()
    converted = converted.replace('^', '**')
    converted = re.sub(r'\be\s*\*\*', 'E**', converted)
    converted = re.sub(r'\bln\s*\(', 'log(', converted)
    converted = re.sub(r'\bpi\b', 'pi', converted, flags=re.IGNORECASE)
    return converted


def differential_form_to_ode_expr(expr: str) -> str | None:
    if not expr:
        return None

    candidate = expr.strip()
    candidate = re.sub(r'\s+', '', candidate)
    candidate = re.sub(r'(?<![A-Za-z0-9_*])dx', '1*dx', candidate, flags=re.IGNORECASE)
    candidate = re.sub(r'(?<![A-Za-z0-9_*])dy', '1*dy', candidate, flags=re.IGNORECASE)

    if '=' not in candidate or 'dx' not in candidate.lower() or 'dy' not in candidate.lower():
        return None

    left, right = candidate.split('=', 1)
    if right not in {'0', '0.0'}:
        return None

    left = left.replace('-', '+-')
    parts = [part for part in left.split('+') if part]

    m_terms = []
    n_terms = []
    for part in parts:
        low = part.lower()
        if low.endswith('dx'):
            m_terms.append(part[:-2] or '1')
        elif low.endswith('dy'):
            n_terms.append(part[:-2] or '1')
        else:
            return None

    if not m_terms or not n_terms:
        return None

    m_expr = '(' + '+'.join(m_terms) + ')'
    n_expr = '(' + '+'.join(n_terms) + ')'
    return f'dy/dx=-({m_expr})/({n_expr})'


def solve_with_sympy(normalized: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr

        expr_raw = str(normalized.get("equation_raw", "") or "")
        eq_type = str(normalized.get("type", "") or "").lower()
        task = str(normalized.get("task", "") or "").lower()

        expr = extract_expression_for_symbolic_work(expr_raw)
        expr = convert_basic_math_to_sympy(expr)

        x = sp.symbols('x')
        y = sp.Function('y')
        local_dict = {
            'x': x,
            'y': y,
            'e': sp.E,
            'E': sp.E,
            'pi': sp.pi,
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'log': sp.log,
            'ln': sp.log,
            'sqrt': sp.sqrt,
            'exp': sp.exp,
        }

        if eq_type == "ode":
            ode_expr = expr

            # REMOVE $ symbols (CRITICAL FIX)
            ode_expr = ode_expr.replace("$", "").strip()

            # Detect differential form: M dx + N dy = 0
            if "dx" in ode_expr and "dy" in ode_expr:
                # Extract M and N
                # Example: dx + exp(3*x)dy = 0

                ode_expr = ode_expr.replace(" ", "")

                match = re.match(r'(.*)dx\+(.*)dy=0', ode_expr)
                if match:
                    M = match.group(1) if match.group(1) else "1"
                    N = match.group(2) if match.group(2) else "1"

                    # dy/dx = -M/N
                    rhs_expr = f"-({M})/({N})"

                    ode_expr = f"Derivative(y(x), x) = {rhs_expr}"

            # Convert dy/dx normally
            ode_expr = re.sub(r'd2y/dx2', 'Derivative(y(x), (x, 2))', ode_expr, flags=re.IGNORECASE)
            ode_expr = re.sub(r'dy/dx', 'Derivative(y(x), x)', ode_expr, flags=re.IGNORECASE)

            # Ensure y → y(x)
            ode_expr = re.sub(r'(?<![A-Za-z0-9_])y(?!\s*\()', 'y(x)', ode_expr)

            if '=' in ode_expr:
                left, right = ode_expr.split('=', 1)
                left_expr = parse_expr(left.strip(), local_dict=local_dict, evaluate=False)
                right_expr = parse_expr(right.strip(), local_dict=local_dict, evaluate=False)
                eq = sp.Eq(left_expr, right_expr)
            else:
                eq = sp.Eq(parse_expr(ode_expr, local_dict=local_dict, evaluate=False), 0)

            sol = sp.dsolve(eq)

            return {
                "status": "success",
                "engine": "sympy",
                "result_raw": str(sol),
                "result_text": str(sol),
                "result_latex": sp.latex(sol),
            }

        if eq_type == "derivative" or task == "differentiate":
            target = re.sub(r'^d/dx\s*', '', expr, flags=re.IGNORECASE).strip()
            f = parse_expr(target, local_dict=local_dict, evaluate=False)
            sol = sp.diff(f, x)
            return {
                "status": "success",
                "engine": "sympy",
                "result_raw": str(sol),
                "result_text": str(sol),
                "result_latex": sp.latex(sol),
            }

        if eq_type == "integral" or task == "integrate":
            target = expr.replace('∫', '').strip()
            target = re.sub(r'\bdx\b$', '', target, flags=re.IGNORECASE).strip()
            f = parse_expr(target, local_dict=local_dict, evaluate=False)
            sol = sp.integrate(f, x)
            return {
                "status": "success",
                "engine": "sympy",
                "result_raw": str(sol),
                "result_text": str(sol),
                "result_latex": sp.latex(sol),
            }

        if eq_type == "algebra":
            if '=' in expr:
                left, right = expr.split('=', 1)
                left_expr = parse_expr(left.strip(), local_dict=local_dict, evaluate=False)
                right_expr = parse_expr(right.strip(), local_dict=local_dict, evaluate=False)
                eq = sp.Eq(left_expr, right_expr)
                sol = sp.solve(eq)
            else:
                algebra_expr = parse_expr(expr.strip(), local_dict=local_dict, evaluate=False)
                sol = sp.solve(algebra_expr)

            return {
                "status": "success",
                "engine": "sympy",
                "result_raw": str(sol),
                "result_text": str(sol),
                "result_latex": sp.latex(sol),
            }

        return {
            "status": "fail",
            "engine": "sympy",
            "error": "Unsupported problem type for current SymPy solver",
        }

    except Exception as e:
        return {
            "status": "error",
            "engine": "sympy",
            "error": str(e),
        }


# ============================================================
# SciPy solver
# ============================================================

def parse_initial_condition_value(initial_condition_raw: str) -> Tuple[float | None, float | None]:
    if not initial_condition_raw:
        return None, None

    match = re.search(r'y\s*\(\s*([^\)]+)\s*\)\s*=\s*([^\s,;]+)', initial_condition_raw, flags=re.IGNORECASE)
    if not match:
        return None, None

    try:
        x0 = float(match.group(1).strip())
        y0 = float(match.group(2).strip())
        return x0, y0
    except Exception:
        return None, None


def parse_numeric_target_x(cleaned_text: str, x0: float) -> float | None:
    if not cleaned_text:
        return None

    patterns = [
        r'find\s+y\s*\(\s*([-+]?\d*\.?\d+)\s*\)',
        r'at\s+x\s*=\s*([-+]?\d*\.?\d+)',
        r'when\s+x\s*=\s*([-+]?\d*\.?\d+)',
        r'for\s+x\s*=\s*([-+]?\d*\.?\d+)',
        r'to\s+x\s*=\s*([-+]?\d*\.?\d+)',
        r'on\s*\[\s*[-+]?\d*\.?\d+\s*,\s*([-+]?\d*\.?\d+)\s*\]',
        r'interval\s*[-+]?\d*\.?\d+\s*to\s*([-+]?\d*\.?\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned_text, flags=re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                if value != x0:
                    return value
            except Exception:
                pass

    all_y_points = re.findall(r'y\s*\(\s*([-+]?\d*\.?\d+)\s*\)', cleaned_text, flags=re.IGNORECASE)
    for raw in all_y_points:
        try:
            value = float(raw)
            if value != x0:
                return value
        except Exception:
            pass

    return None


def convert_rhs_to_callable(rhs_expr: str):
    import sympy as sp

    x_sym, y_sym = sp.symbols('x y')
    rhs_expr = convert_basic_math_to_sympy(rhs_expr)
    rhs_expr = re.sub(r'(?<![A-Za-z0-9_])y\(x\)', 'y', rhs_expr)
    rhs_expr = re.sub(r'(?<![A-Za-z0-9_])y(?![A-Za-z0-9_])', 'y', rhs_expr)

    local_dict = {
        'x': x_sym,
        'y': y_sym,
        'e': sp.E,
        'E': sp.E,
        'pi': sp.pi,
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'log': sp.log,
        'ln': sp.log,
        'sqrt': sp.sqrt,
        'exp': sp.exp,
    }

    rhs = sp.sympify(rhs_expr, locals=local_dict)
    return sp.lambdify((x_sym, y_sym), rhs, 'numpy'), rhs


def solve_with_scipy(normalized: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import numpy as np
        from scipy.integrate import solve_ivp

        eq_type = str(normalized.get('type', '') or '').lower()
        expr_raw = str(normalized.get('equation_raw', '') or '')
        cleaned_text = str(normalized.get('cleaned_text', '') or '')
        initial_condition_raw = str(normalized.get('initial_condition_raw', '') or '')

        if eq_type != 'ode':
            return {
                'status': 'fail',
                'engine': 'scipy',
                'error': 'Current SciPy solver only supports ODE initial value problems',
            }

        x0, y0 = parse_initial_condition_value(initial_condition_raw)
        if x0 is None or y0 is None:
            return {
                'status': 'fail',
                'engine': 'scipy',
                'error': 'Could not parse numeric initial condition for SciPy solver',
            }

        ode_expr = extract_expression_for_symbolic_work(expr_raw)
        ode_expr = ode_expr.strip()
        ode_expr = re.sub(r'dy/dx', 'yprime', ode_expr, flags=re.IGNORECASE)

        if '=' in ode_expr:
            left, right = ode_expr.split('=', 1)
            left = left.strip()
            right = right.strip()

            if 'yprime' in left and 'yprime' not in right:
                rhs_expr = f'({right}) - ({left.replace("yprime", "0")})' if left.replace('yprime', '').strip() else right
                # better handling for yprime + g = h => rhs = h - g
                residual = left.replace('yprime', '').strip()
                if residual:
                    rhs_expr = f'({right}) - ({residual})'
                else:
                    rhs_expr = right
            elif 'yprime' in right and 'yprime' not in left:
                residual = right.replace('yprime', '').strip()
                if residual:
                    rhs_expr = f'({left}) - ({residual})'
                else:
                    rhs_expr = left
            else:
                return {
                    'status': 'fail',
                    'engine': 'scipy',
                    'error': 'SciPy solver could not isolate dy/dx from the ODE',
                }
        else:
            return {
                'status': 'fail',
                'engine': 'scipy',
                'error': 'SciPy solver requires an equation containing =',
            }

        f, rhs_sympy = convert_rhs_to_callable(rhs_expr)

        x_target = parse_numeric_target_x(cleaned_text, x0)
        if x_target is None:
            x_target = x0 + 1.0

        def ode_func(x, y):
            return np.asarray([f(x, y[0])], dtype=float)

        num_eval_points = max(25, int(abs(x_target - x0) * 50) + 1)
        t_eval = np.linspace(x0, x_target, num_eval_points)

        sol = solve_ivp(
            ode_func,
            (x0, x_target),
            [y0],
            t_eval=t_eval,
            dense_output=True,
        )

        if not sol.success:
            return {
                'status': 'fail',
                'engine': 'scipy',
                'error': f'SciPy solve_ivp failed: {sol.message}',
            }

        y_target = float(sol.y[0][-1])
        sample_points = [
            {'x': float(tx), 'y': float(ty)}
            for tx, ty in zip(sol.t[:10], sol.y[0][:10])
        ]

        result_text = f'y({x_target}) ≈ {y_target}'

        return {
            'status': 'success',
            'engine': 'scipy',
            'result_raw': result_text,
            'result_text': result_text,
            'result_latex': '',
            'x0': x0,
            'y0': y0,
            'x_target': x_target,
            'y_target': y_target,
            'rhs_expr': str(rhs_sympy),
            'sample_points': sample_points,
        }

    except Exception as e:
        return {
            'status': 'error',
            'engine': 'scipy',
            'error': str(e),
        }


# ============================================================
# Router / thinking mode
# ============================================================

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
        stream=False
    )

    raw = response["message"]["content"]
    return parse_router_output(raw)


# ============================================================
# Clipboard / OCR helpers
# ============================================================

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


def build_image_prompt(
    instruction: str,
    structured_text: str,
    image_path: str,
    mode: str,
    normalization: Dict[str, Any] | None = None,
    math_route: Dict[str, Any] | None = None,
    solver_result: Dict[str, Any] | None = None,
) -> str:
    extracted = structured_text.strip() if structured_text.strip() else "[No text extracted from screenshot]"

    if mode == "math":
        normalization_block = ""
        if normalization is not None:
            normalization_block = (
                "Normalization result (JSON):\n"
                f"{json.dumps(normalization, indent=2)}\n\n"
            )

        route_block = ""
        if math_route is not None:
            route_block = (
                "Math router result (JSON):\n"
                f"{json.dumps(math_route, indent=2)}\n\n"
            )

        solver_block = ""
        if solver_result is not None:
            solver_block = (
                "Solver result (JSON):\n"
                f"{json.dumps(solver_result, indent=2)}\n\n"
            )

        return (
            f"The user sent a screenshot through the clipboard.\n"
            f"Image path: {image_path}\n"
            f"Detected content type: math\n\n"
            f"User instruction:\n{instruction if instruction else '[No instruction provided]'}\n\n"
            f"Recognized math content:\n{extracted}\n\n"
            f"{normalization_block}"
            f"{route_block}"
            f"{solver_block}"
            f"Treat the recognized math content as OCR output that may contain small errors. "
            f"Use the normalization, routing, and solver results as helpful hints, not absolute truth. "
            f"If a solver result exists and is successful, explain that result instead of solving fully from scratch. "
            f"If ambiguity remains, point it out clearly."
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



def build_router_input(
    original_prompt: str,
    final_user_prompt: str,
    is_paste: bool = False,
    detected_mode: str = "",
    instruction: str = "",
    normalization: Dict[str, Any] | None = None,
    solver_result: Dict[str, Any] | None = None,
) -> str:
    if not is_paste:
        return final_user_prompt

    parts = ["/paste", f"mode={detected_mode or 'unknown'}"]

    if instruction:
        parts.append(f"instruction={instruction}")

    if normalization:
        eq = str(normalization.get("equation_raw", "") or "").strip()
        eq_type = str(normalization.get("type", "") or "").strip()
        task = str(normalization.get("task", "") or "").strip()
        if task:
            parts.append(f"task={task}")
        if eq_type:
            parts.append(f"type={eq_type}")
        if eq:
            parts.append(f"equation={eq}")

    if solver_result:
        status = str(solver_result.get("status", "") or "").strip()
        engine = str(solver_result.get("engine", "") or "").strip()
        if status:
            parts.append(f"solver_status={status}")
        if engine:
            parts.append(f"solver_engine={engine}")

    return "; ".join(parts)


def should_skip_think_router_for_math(
    is_paste: bool,
    detected_mode: str,
    solver_result: Dict[str, Any] | None,
) -> bool:
    return (
        is_paste
        and detected_mode == "math"
        and solver_result is not None
        and str(solver_result.get("status", "")).lower() == "success"
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
    detected_mode = ""
    instruction = ""
    normalization_result = None
    solver_result = None

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
            img_path = save_clipboard_image(img)
        except Exception as e:
            print(f"\n[Error] Failed to save clipboard image: {e}\n")
            continue

        try:
            preview_text = run_real_ocr(img_path)
        except Exception as e:
            print(f"\n[Error] OCR preview failed: {e}\n")
            continue

        detected_mode = detect_image_mode(instruction, preview_text)
        debug_print(f"[ROUTER] Image content route: {detected_mode.upper()}")

        normalization_result = None
        math_route_result = None
        solver_result = None

        try:
            if detected_mode == "math":
                if Pix2Text is None:
                    print("\n[Error] Pix2Text is not installed. Run: pip install pix2text\n")
                    continue
                structured_text = run_math_ocr(img_path)
                normalization_result = run_full_math_normalization(structured_text)
                math_route_result = route_math_problem(normalization_result)

                if math_route_result.get("route") == "sympy":
                    solver_result = solve_with_sympy(normalization_result)
                elif math_route_result.get("route") == "scipy":
                    solver_result = solve_with_scipy(normalization_result)

                debug_print(f"[NORMALIZER] Normalization complete via: {normalization_result.get('normalizer_source', 'rule').upper()}")
                debug_print(json.dumps(normalization_result, indent=2))
                debug_print(f"[MATH ROUTER] Route selected: {math_route_result.get('route', 'llm').upper()}")
                debug_print(json.dumps(math_route_result, indent=2))
                if solver_result is not None:
                    debug_print(f"[{str(solver_result.get('engine', 'solver')).upper()} SOLVER RESULT]")
                    debug_print(json.dumps(solver_result, indent=2))
            else:
                structured_text = preview_text
        except Exception as e:
            print(f"\n[Error] OCR failed: {e}\n")
            continue

        final_user_prompt = build_image_prompt(
            instruction=instruction,
            structured_text=structured_text,
            image_path=img_path,
            mode=detected_mode,
            normalization=normalization_result,
            math_route=math_route_result,
            solver_result=solver_result,
        )
        debug_print("[FINAL PROMPT] Built prompt from screenshot pipeline")

    is_paste_prompt = prompt.startswith("/paste")
    router_start = time.perf_counter()

    router_input = build_router_input(
        original_prompt=prompt,
        final_user_prompt=final_user_prompt,
        is_paste=is_paste_prompt,
        detected_mode=detected_mode if is_paste_prompt else "",
        instruction=instruction if is_paste_prompt else "",
        normalization=normalization_result if is_paste_prompt else None,
        solver_result=solver_result if is_paste_prompt else None,
    )

    if should_skip_think_router_for_math(
        is_paste=is_paste_prompt,
        detected_mode=detected_mode if is_paste_prompt else "",
        solver_result=solver_result if is_paste_prompt else None,
    ):
        use_thinking = False
        debug_print("[ROUTER] Skipping think-router for solved math screenshot")
    else:
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
