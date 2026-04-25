"""
ai_evaluator.py — Production-Grade Answer Evaluator
=====================================================
PRIMARY:  Google Gemini 1.5 Flash (semantic evaluation + feedback)
FALLBACK: 7-technique local NLP pipeline
Features:
  - Handles questions WITH or WITHOUT expected answers
  - Proper normalisation (case, spaces, punctuation)
  - Meaningful, explainable feedback (not generic)
  - In-memory cache (thread-safe)
  - Timeout-guarded Gemini calls
  - Auto-generate expected answers via Gemini
  - Score range 0–10, realistically distributed
"""

import os
import re
import json
import hashlib
import time
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-1.5-flash"
MAX_RETRIES    = 2
RETRY_DELAY    = 1
GEMINI_TIMEOUT = 10   # seconds before falling back to local

# ── Thread-safe in-memory cache ─────────────────────────────────────────────────
_eval_cache: dict = {}
_cache_lock = threading.Lock()

def _cache_key(expected: str, student: str) -> str:
    combined = f"{expected.strip().lower()}|||{student.strip().lower()}"
    return hashlib.sha256(combined.encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# TEXT NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","to","of","in","for","on","with","at","by","from","as",
    "into","through","and","or","but","not","this","that","it","its","i",
    "we","you","he","she","they","which","who","what","when","where","how",
    "also","so","if","then","than","about","up","out","use","our","their",
    "there","here","just","very","much","more","some","any","all","each",
    "both","few","those","these","such","only","same","than","too","very",
}

def _tokens(text: str) -> set:
    """Extract meaningful tokens, remove stopwords."""
    return {w for w in re.findall(r"[a-z]+", text.lower())
            if w not in STOPWORDS and len(w) > 2}


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI — AUTO GENERATE EXPECTED ANSWER
# ══════════════════════════════════════════════════════════════════════════════

def generate_expected_answer(question_text: str) -> str:
    """
    Call Gemini to generate a correct expected answer for a question.
    Returns empty string on failure.
    """
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set — cannot auto-generate answer.")
        return ""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = (
            "You are an academic subject matter expert. "
            "Provide a comprehensive, accurate expected answer for this question. "
            "Cover all key concepts in 2–4 clear sentences. "
            "Do NOT include preamble, just the answer itself.\n\n"
            f"Question: {question_text}"
        )
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 350}
        )
        answer = response.text.strip()
        logger.info("Gemini generated expected answer for: %s", question_text[:60])
        return answer
    except Exception as e:
        logger.error("Gemini answer generation failed: %s", e)
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 1 — NORMALISED EXACT MATCH
# ══════════════════════════════════════════════════════════════════════════════

def _exact_match(expected: str, student: str) -> float:
    return 1.0 if normalise(expected) == normalise(student) else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 2 — KEYWORD MATCH WITH LENGTH PENALTY
# ══════════════════════════════════════════════════════════════════════════════

def _keyword_match(expected: str, student: str) -> float:
    exp_words = _tokens(expected)
    stu_words = _tokens(student)
    if not exp_words:
        return 0.5   # no reference — neutral
    overlap = len(exp_words & stu_words) / len(exp_words)
    exp_len = len(expected.split())
    stu_len = len(student.split())
    length_ratio = min(1.0, stu_len / max(1, exp_len))
    return overlap * 0.80 + length_ratio * 0.20


# ══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 3 — TF-IDF COSINE SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

def _tfidf_cosine(expected: str, student: str) -> float:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        mat = vec.fit_transform([expected, student])
        return float(cosine_similarity(mat[0], mat[1])[0][0])
    except Exception as e:
        logger.warning("TF-IDF failed: %s", e)
        return _keyword_match(expected, student) * 0.85


# ══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 4 — SENTENCE TRANSFORMER SEMANTIC SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

_st_model = None
_st_lock  = threading.Lock()

def _get_st_model():
    global _st_model
    if _st_model is None:
        with _st_lock:
            if _st_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    _st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
                    logger.info("SentenceTransformer loaded.")
                except Exception as e:
                    logger.warning("SentenceTransformer load failed: %s", e)
                    _st_model = "FAILED"
    return None if _st_model == "FAILED" else _st_model

def _semantic_similarity(expected: str, student: str) -> float:
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        model = _get_st_model()
        if model is None:
            return _tfidf_cosine(expected, student)
        emb = model.encode([expected, student])
        score = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
        return max(0.0, score)
    except Exception as e:
        logger.warning("Semantic similarity failed: %s", e)
        return _tfidf_cosine(expected, student)


# ══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 5 — GEMINI SEMANTIC EVALUATION
# Runs in a thread with strict timeout so it never hangs the request
# ══════════════════════════════════════════════════════════════════════════════

# Prompt when expected answer IS provided
GEMINI_EVAL_PROMPT_WITH_EXPECTED = """You are a strict but fair academic evaluator scoring a student's answer.

Return ONLY valid JSON with no markdown fences:
{{
  "semantic_score": <float 0.0–1.0>,
  "feedback": "<2–3 specific sentences: what is correct, what is missing or wrong, how to improve>",
  "key_points_covered": [<list of concept strings the student correctly addressed>],
  "key_points_missing": [<list of concept strings from expected answer the student missed>]
}}

SCORING GUIDE:
- 0.90–1.00: All key concepts covered correctly (wording may differ)
- 0.75–0.89: Very good, only minor details missing
- 0.55–0.74: Good, main idea present but important details absent
- 0.35–0.54: Partial, some relevant content but significant gaps
- 0.15–0.34: Weak, barely touches on the topic
- 0.00–0.14: Wrong or completely irrelevant

RULES:
- Synonyms and paraphrasing = full credit for that concept
- Spelling/grammar errors do not reduce score unless meaning is lost
- Ignore case and extra spaces (they are normalised)
- Never return exactly 0.0 unless the answer is empty/gibberish
- Never return exactly 1.0 unless the answer is flawlessly complete
- Feedback MUST name specific concepts, not generic phrases like "good answer"

EXPECTED ANSWER:
\"\"\"{expected}\"\"\"

STUDENT ANSWER:
\"\"\"{student}\"\"\""""

# Prompt when NO expected answer is provided — Gemini evaluates on its own knowledge
GEMINI_EVAL_PROMPT_NO_EXPECTED = """You are a strict but fair academic evaluator.
The teacher did NOT provide an expected answer, so evaluate the student's answer based on
your own subject-matter knowledge.

Question: \"\"\"{question}\"\"\"
Student Answer: \"\"\"{student}\"\"\"

Return ONLY valid JSON with no markdown fences:
{{
  "semantic_score": <float 0.0–1.0, how correct and complete the answer is>,
  "feedback": "<2–3 specific sentences: what is correct, what is missing or wrong, what the full answer should include>",
  "key_points_covered": [<concepts the student correctly mentioned>],
  "key_points_missing": [<important concepts the student missed>]
}}

RULES:
- Evaluate solely on factual correctness and completeness
- Never return exactly 0.0 unless the answer is empty/gibberish
- Feedback MUST reference specific subject-matter concepts"""


def _gemini_evaluate(expected: str, student: str, question: str = "") -> Optional[dict]:
    """Call Gemini with a thread timeout. Returns None on failure."""
    if not GEMINI_API_KEY:
        return None

    result_holder = [None]
    error_holder  = [None]

    def _call():
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)

            if expected.strip():
                prompt = GEMINI_EVAL_PROMPT_WITH_EXPECTED.format(
                    expected=expected, student=student
                )
            else:
                prompt = GEMINI_EVAL_PROMPT_NO_EXPECTED.format(
                    question=question or "Unknown question", student=student
                )

            for attempt in range(MAX_RETRIES + 1):
                try:
                    resp = model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": 0,
                            "max_output_tokens": 500,
                            "response_mime_type": "application/json",
                        }
                    )
                    text = resp.text.strip()
                    # Strip markdown fences if model ignores mime type
                    text = re.sub(r"^```json\s*", "", text)
                    text = re.sub(r"```$", "", text).strip()
                    data = json.loads(text)
                    score = float(data.get("semantic_score", 0.5))
                    score = max(0.01, min(0.99, score))

                    # Validate and clean key points lists
                    covered = data.get("key_points_covered", [])
                    missing = data.get("key_points_missing", [])
                    if not isinstance(covered, list): covered = []
                    if not isinstance(missing, list): missing = []
                    covered = [str(x).strip() for x in covered if str(x).strip()]
                    missing = [str(x).strip() for x in missing if str(x).strip()]

                    result_holder[0] = {
                        "score":              score,
                        "feedback":           str(data.get("feedback", "")).strip(),
                        "key_points_covered": covered,
                        "key_points_missing": missing,
                    }
                    return
                except (json.JSONDecodeError, Exception) as e:
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                    else:
                        error_holder[0] = e
        except Exception as e:
            error_holder[0] = e

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=GEMINI_TIMEOUT)

    if t.is_alive():
        logger.warning("Gemini timed out after %ss — using local fallback.", GEMINI_TIMEOUT)
        return None
    if error_holder[0]:
        logger.warning("Gemini error: %s", error_holder[0])
        return None
    return result_holder[0]


# ══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 6 — SYNONYM MATCHING (built-in map + optional WordNet)
# ══════════════════════════════════════════════════════════════════════════════

SYNONYM_GROUPS = [
    {"big","large","huge","enormous","great","vast","massive","gigantic"},
    {"small","little","tiny","minute","petite","mini","miniature"},
    {"fast","quick","rapid","swift","speedy","hasty"},
    {"slow","sluggish","leisurely","gradual","unhurried"},
    {"smart","intelligent","clever","bright","wise","brilliant"},
    {"happy","joyful","glad","pleased","content","cheerful","delighted"},
    {"sad","unhappy","sorrowful","miserable","depressed","gloomy"},
    {"angry","furious","mad","irate","enraged"},
    {"beautiful","pretty","attractive","gorgeous","lovely","stunning"},
    {"good","great","excellent","fine","superb","outstanding","wonderful"},
    {"bad","poor","terrible","awful","dreadful","horrible"},
    {"start","begin","commence","initiate","launch","open"},
    {"end","finish","conclude","terminate","complete","close","stop"},
    {"make","create","build","construct","produce","form","generate"},
    {"show","display","present","exhibit","demonstrate","reveal"},
    {"think","believe","consider","suppose","assume"},
    {"say","tell","speak","utter","state","mention","declare"},
    {"help","assist","aid","support","facilitate"},
    {"use","utilize","employ","apply","operate"},
    {"increase","grow","rise","expand","escalate","improve","enhance"},
    {"decrease","reduce","fall","decline","diminish","lessen"},
    {"important","significant","crucial","vital","essential","key","critical"},
    {"difficult","hard","challenging","tough","complex","complicated"},
    {"easy","simple","straightforward","effortless","uncomplicated"},
    {"car","automobile","vehicle","auto","motorcar"},
    {"house","home","residence","dwelling","abode"},
    {"doctor","physician","medic","clinician"},
    {"teacher","educator","instructor","professor","tutor"},
    {"student","pupil","learner","scholar"},
    {"water","liquid","fluid","aqua"},
    {"food","nourishment","sustenance","nutrition","meal"},
    {"money","currency","cash","funds","capital","finance"},
    {"work","job","employment","occupation","career","profession"},
    {"problem","issue","trouble","difficulty","challenge"},
    {"answer","solution","response","reply","result"},
    {"cause","reason","factor","source","origin"},
    {"effect","result","outcome","consequence","impact"},
    {"method","approach","technique","process","procedure","way"},
    {"change","modify","alter","transform","adjust","update"},
    {"control","manage","regulate","govern","direct","oversee"},
    {"system","network","structure","framework","organization"},
    {"energy","power","force","strength"},
    {"data","information","facts","details","evidence"},
    {"law","rule","principle","regulation","guideline"},
    {"theory","concept","idea","hypothesis","model"},
    {"cell","unit","element","component","particle"},
    {"part","component","section","portion","piece"},
    {"type","kind","sort","category","class","variety"},
    {"area","region","zone","territory","domain","field"},
    {"send","transmit","transfer","deliver","dispatch"},
    {"receive","get","obtain","acquire","gain"},
    {"produce","generate","yield","output","emit"},
    {"absorb","take in","consume","ingest","incorporate"},
    {"contain","hold","include","encompass","comprise"},
    {"allow","permit","enable","let","authorize"},
    {"prevent","stop","block","inhibit","restrict","hinder"},
    {"provide","supply","give","offer","deliver","furnish"},
    {"support","back","endorse","uphold","sustain"},
    {"plant","vegetation","flora","organism","herb","shrub"},
    {"animal","creature","organism","beast","fauna"},
    {"connect","link","join","attach","unite","combine"},
    {"break","damage","destroy","ruin","harm","impair"},
    {"protect","guard","defend","shield","preserve","secure"},
    {"measure","calculate","compute","quantify","assess","evaluate"},
    {"store","save","keep","retain","preserve","hold"},
    {"remove","delete","eliminate","erase","clear"},
    {"move","travel","go","proceed","advance","migrate"},
    {"release","emit","discharge","expel"},
]

def _build_synonym_map() -> dict:
    m = {}
    for i, grp in enumerate(SYNONYM_GROUPS):
        for w in grp:
            m[w] = i
    return m

_SYNONYM_MAP = _build_synonym_map()

def _synonym_match(expected: str, student: str) -> float:
    # Try NLTK WordNet first
    try:
        import nltk
        from nltk.corpus import wordnet
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)

        exp_tokens = _tokens(expected)
        stu_tokens = _tokens(student)
        if not exp_tokens:
            return 0.5  # neutral when no reference

        def get_synonyms(word):
            syns = {word}
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    syns.add(lemma.name().lower().replace("_", " "))
            return syns

        matched = 0.0
        for ew in exp_tokens:
            if ew in stu_tokens:
                matched += 1.0
                continue
            ew_syns = get_synonyms(ew)
            if ew_syns & stu_tokens:
                matched += 0.9
                continue
            grp = _SYNONYM_MAP.get(ew)
            if grp is not None and any(_SYNONYM_MAP.get(sw) == grp for sw in stu_tokens):
                matched += 0.85

        return min(1.0, matched / len(exp_tokens))

    except Exception:
        # Pure built-in synonym map fallback
        exp_tokens = _tokens(expected)
        stu_tokens = _tokens(student)
        if not exp_tokens:
            return 0.5
        matched = 0.0
        for ew in exp_tokens:
            if ew in stu_tokens:
                matched += 1.0
                continue
            grp = _SYNONYM_MAP.get(ew)
            if grp is not None and any(_SYNONYM_MAP.get(sw) == grp for sw in stu_tokens):
                matched += 0.85
        return min(1.0, matched / len(exp_tokens))


# ══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 7 — CONCEPT MATCH WITH STEM PREFIX
# ══════════════════════════════════════════════════════════════════════════════

def _concept_match(expected: str, student: str) -> float:
    exp_tokens = _tokens(expected)
    stu_tokens = _tokens(student)
    if not exp_tokens:
        return 0.5
    matched = 0.0
    for ew in exp_tokens:
        if ew in stu_tokens:
            matched += 1.0
            continue
        grp = _SYNONYM_MAP.get(ew)
        if grp is not None and any(_SYNONYM_MAP.get(sw) == grp for sw in stu_tokens):
            matched += 0.85
            continue
        if len(ew) > 5:
            stem = ew[:max(3, len(ew) - 3)]
            if any(sw.startswith(stem) for sw in stu_tokens):
                matched += 0.70
    return min(1.0, matched / len(exp_tokens))


# ══════════════════════════════════════════════════════════════════════════════
# SCORING WEIGHTS (must sum to 1.0)
# ══════════════════════════════════════════════════════════════════════════════

WEIGHTS_WITH_GEMINI = {
    "exact_match":   0.03,
    "keyword_match": 0.10,
    "tfidf_cosine":  0.10,
    "semantic_sim":  0.20,
    "gemini":        0.35,   # dominant when available
    "synonym_match": 0.12,
    "concept_match": 0.10,
}

WEIGHTS_LOCAL_ONLY = {
    "exact_match":   0.05,
    "keyword_match": 0.18,
    "tfidf_cosine":  0.18,
    "semantic_sim":  0.32,
    "synonym_match": 0.15,
    "concept_match": 0.12,
}


# ══════════════════════════════════════════════════════════════════════════════
# BUILD RESULT — apply guardrails, produce structured output
# ══════════════════════════════════════════════════════════════════════════════

def _build_result(raw: float, scores: dict, feedback: str,
                  covered: list, missing: list, used_gemini: bool,
                  no_expected: bool = False) -> dict:
    """Apply tiered floors and produce final result dict."""

    gem = scores.get("gemini", 0.5)
    sem = scores.get("semantic_sim", 0)
    kw  = scores.get("keyword_match", 0)
    con = scores.get("concept_match", 0)
    syn = scores.get("synonym_match", 0)

    content_signal = gem * 0.4 + sem * 0.3 + kw * 0.15 + syn * 0.08 + con * 0.07

    # Tiered floors — ensure realistic spread
    if content_signal >= 0.88:
        raw = max(raw, 0.87)
    elif content_signal >= 0.72:
        raw = max(raw, 0.68)
    elif content_signal >= 0.55:
        raw = max(raw, 0.47)
    elif content_signal >= 0.38:
        raw = max(raw, 0.30)
    elif content_signal >= 0.20:
        raw = max(raw, 0.14)
    elif content_signal >= 0.08:
        raw = max(raw, 0.07)

    # Map to final integer score
    if raw >= 0.96 and content_signal >= 0.93:
        final_score = 10
    else:
        scaled = raw * 10
        final_score = max(1, min(9, round(scaled)))
        if content_signal < 0.05:
            final_score = 0

    final_score = int(max(0, min(10, final_score)))

    # Build feedback if Gemini did not supply one
    if not feedback:
        feedback = _build_local_feedback(
            final_score, scores, covered, missing, no_expected
        )

    logger.info(
        "Score→ exact:%.2f kw:%.2f tfidf:%.2f sem:%.2f gem:%.2f "
        "syn:%.2f con:%.2f | raw:%.3f → %d (gemini=%s, no_expected=%s)",
        scores.get("exact_match", 0), kw,
        scores.get("tfidf_cosine", 0), sem,
        gem, syn, con,
        raw, final_score, used_gemini, no_expected
    )

    return {
        "score":              final_score,
        "feedback":           feedback,
        "key_points_covered": covered,
        "key_points_missing": missing,
        "breakdown":          {k: round(v * 10, 2) for k, v in scores.items()},
        "used_gemini":        used_gemini,
        "no_expected_answer": no_expected,
    }


def _build_local_feedback(score: int, scores: dict, covered: list,
                           missing: list, no_expected: bool) -> str:
    """
    Produce meaningful, specific feedback without Gemini.
    References actual metric values — not generic phrases.
    """
    kw  = scores.get("keyword_match", 0)
    sem = scores.get("semantic_sim", 0)
    con = scores.get("concept_match", 0)
    syn = scores.get("synonym_match", 0)

    covered_str = ", ".join(covered[:3]) if covered else None
    missing_str = ", ".join(missing[:3]) if missing else None

    if score == 10:
        return (
            "Excellent — your answer covers all key concepts with precision. "
            "The semantic structure closely matches the expected answer."
        )
    if score >= 8:
        msg = f"Very strong answer (keyword overlap: {int(kw*100)}%, concept match: {int(con*100)}%). "
        if missing_str:
            msg += f"Minor points that could be added: {missing_str}."
        else:
            msg += "Only minor elaboration could improve this further."
        return msg
    if score >= 6:
        msg = f"Good answer — main idea is present (semantic similarity: {int(sem*100)}%). "
        if missing_str:
            msg += f"Key concepts that are missing: {missing_str}. "
        if covered_str:
            msg += f"Correctly covered: {covered_str}."
        return msg
    if score >= 4:
        msg = f"Partial answer (concept match: {int(con*100)}%). "
        if covered_str:
            msg += f"You correctly mentioned: {covered_str}. "
        if missing_str:
            msg += f"Important missing concepts: {missing_str}."
        else:
            msg += "Significant portions of the expected answer are absent."
        return msg
    if score >= 2:
        msg = f"Weak answer — limited relevant content (keyword match: {int(kw*100)}%). "
        if missing_str:
            msg += f"Missing core concepts: {missing_str}. "
        msg += "Review the material and focus on including specific terminology."
        return msg
    if score == 1:
        if no_expected:
            return (
                "The answer contains minimal relevant content based on AI knowledge evaluation. "
                "Review the topic and aim to include specific definitions and examples."
            )
        return (
            "The answer barely touches the topic. "
            "Most key concepts from the expected answer are absent. "
            "Please review the material thoroughly."
        )
    # score == 0
    if no_expected:
        return "No answer was submitted, or the response was entirely off-topic."
    return "No answer submitted, or the response does not address the question."


# ══════════════════════════════════════════════════════════════════════════════
# CORE EVALUATION PIPELINES
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_with_expected(expected: str, student: str) -> dict:
    """Full 7-technique pipeline when expected answer is available."""
    scores = {
        "exact_match":   _exact_match(expected, student),
        "keyword_match": _keyword_match(expected, student),
        "tfidf_cosine":  _tfidf_cosine(expected, student),
        "semantic_sim":  _semantic_similarity(expected, student),
        "gemini":        0.5,   # placeholder
        "synonym_match": _synonym_match(expected, student),
        "concept_match": _concept_match(expected, student),
    }

    feedback = ""
    covered = []
    missing = []
    used_gemini = False

    gemini_result = _gemini_evaluate(expected, student)
    if gemini_result:
        scores["gemini"] = gemini_result["score"]
        feedback = gemini_result.get("feedback", "")
        covered  = gemini_result.get("key_points_covered", [])
        missing  = gemini_result.get("key_points_missing", [])
        used_gemini = True
        raw = sum(scores[k] * WEIGHTS_WITH_GEMINI[k] for k in WEIGHTS_WITH_GEMINI)
    else:
        # Redistribute Gemini's weight among local techniques
        raw = sum(scores[k] * WEIGHTS_LOCAL_ONLY[k] for k in WEIGHTS_LOCAL_ONLY
                  if k in scores)

    return _build_result(raw, scores, feedback, covered, missing, used_gemini, no_expected=False)


def _evaluate_without_expected(student: str, question: str = "") -> dict:
    """
    Evaluate when NO expected answer is stored.
    Gemini evaluates based on its own subject-matter knowledge.
    Local fallback uses heuristics (length, coherence, vocabulary richness).
    """
    feedback = ""
    covered = []
    missing = []
    used_gemini = False

    # Try Gemini first — it can evaluate factual correctness without a reference
    gemini_result = _gemini_evaluate(expected="", student=student, question=question)

    if gemini_result:
        gem_score = gemini_result["score"]
        feedback  = gemini_result.get("feedback", "")
        covered   = gemini_result.get("key_points_covered", [])
        missing   = gemini_result.get("key_points_missing", [])
        used_gemini = True

        # Use Gemini score directly (no local reference to compare against)
        final_score = max(1, min(9, round(gem_score * 10)))
        if gem_score < 0.05:
            final_score = 0

        if not feedback:
            feedback = _build_local_feedback(
                final_score,
                {"keyword_match": gem_score, "semantic_sim": gem_score,
                 "concept_match": gem_score, "synonym_match": gem_score, "gemini": gem_score},
                covered, missing, no_expected=True
            )

        return {
            "score":              final_score,
            "feedback":           feedback,
            "key_points_covered": covered,
            "key_points_missing": missing,
            "breakdown":          {"gemini": round(gem_score * 10, 2)},
            "used_gemini":        True,
            "no_expected_answer": True,
        }

    else:
        # Pure heuristic: estimate quality from length + vocabulary richness
        words = student.split()
        word_count = len(words)
        unique_ratio = len(set(w.lower() for w in words)) / max(1, word_count)
        meaningful = len(_tokens(student))

        if word_count == 0:
            heuristic = 0.0
        elif word_count < 5:
            heuristic = 0.10
        elif word_count < 15:
            heuristic = 0.25 + unique_ratio * 0.15
        elif word_count < 40:
            heuristic = 0.35 + unique_ratio * 0.20 + min(0.10, meaningful / 50)
        else:
            heuristic = 0.45 + unique_ratio * 0.20 + min(0.10, meaningful / 60)

        heuristic = min(0.65, heuristic)   # cap at 6.5/10 without a reference
        final_score = max(1, min(6, round(heuristic * 10)))
        if word_count == 0:
            final_score = 0

        feedback = (
            f"No expected answer is on file for this question. "
            f"Your answer has been evaluated on length ({word_count} words) and "
            f"vocabulary richness ({int(unique_ratio*100)}% unique words). "
            f"For a more accurate score, ask your teacher to add an expected answer."
        )

        return {
            "score":              final_score,
            "feedback":           feedback,
            "key_points_covered": [],
            "key_points_missing": [],
            "breakdown":          {"heuristic": round(heuristic * 10, 2)},
            "used_gemini":        False,
            "no_expected_answer": True,
        }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def ai_evaluate(expected_answer: str, student_answer: str,
                question_text: str = "") -> dict:
    """
    Evaluate student answer.

    Parameters
    ----------
    expected_answer : str
        The teacher's expected answer. May be empty — Gemini will evaluate
        based on its own knowledge if the API key is set.
    student_answer  : str
        The student's answer.
    question_text   : str
        The question text — used when expected_answer is absent so Gemini
        has context to evaluate against.

    Returns
    -------
    dict with keys: score, feedback, key_points_covered, key_points_missing,
                    breakdown, used_gemini, no_expected_answer
    """
    expected = (expected_answer or "").strip()
    student  = (student_answer  or "").strip()

    # Empty student answer — always 0
    if not student:
        return {
            "score":              0,
            "feedback":           "No answer was submitted.",
            "key_points_covered": [],
            "key_points_missing": [],
            "breakdown":          {},
            "used_gemini":        False,
            "no_expected_answer": not bool(expected),
        }

    # Normalised exact match fast-path (only when expected exists)
    if expected and normalise(expected) == normalise(student):
        return {
            "score":              10,
            "feedback":           "Perfect match — your answer covers all required concepts.",
            "key_points_covered": [],
            "key_points_missing": [],
            "breakdown":          {},
            "used_gemini":        False,
            "no_expected_answer": False,
        }

    # Cache check (include question_text so no-expected evaluations differ per question)
    cache_key_str = f"{expected}|||{student}|||{question_text}"
    key = hashlib.sha256(cache_key_str.lower().strip().encode()).hexdigest()
    with _cache_lock:
        if key in _eval_cache:
            return _eval_cache[key]

    if expected:
        result = _evaluate_with_expected(expected, student)
    else:
        result = _evaluate_without_expected(student, question=question_text)

    with _cache_lock:
        _eval_cache[key] = result
    return result


def ai_evaluate_safe(expected_answer: str, student_answer: str,
                     question_text: str = "",
                     fallback_score: Optional[int] = None) -> dict:
    """
    Never raises. Returns a safe fallback result on any error.
    Always call this from routes — never call ai_evaluate() directly.
    """
    try:
        return ai_evaluate(expected_answer, student_answer, question_text)
    except Exception as e:
        logger.error("ai_evaluate_safe error: %s", e)
        score = fallback_score if fallback_score is not None else 0
        return {
            "score":              score,
            "feedback":           "Evaluation service temporarily unavailable. Score assigned automatically.",
            "key_points_covered": [],
            "key_points_missing": [],
            "breakdown":          {},
            "used_gemini":        False,
            "no_expected_answer": not bool((expected_answer or "").strip()),
            "error":              str(e),
        }
