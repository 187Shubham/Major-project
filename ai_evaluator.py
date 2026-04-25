"""
ai_evaluator.py — Industry-Grade Hybrid Answer Evaluation Pipeline
==================================================================
ARCHITECTURE:
  User Answer
      │
      ▼
  1. Text Normalizer      ← fixes spaces, case, punctuation, abbreviations
      │
      ▼
  2. Exact Match          ← short-circuit: identical after normalisation → 10/10
      │
      ▼
  3. Keyword Overlap      ← stop-word filtered, lemmatized, length-penalised
      │
      ▼
  4. Semantic Similarity  ← MiniLM embeddings (precomputed + cached per question)
      │
      ▼
  5. Gemini AI (optional) ← dominant signal when API key is set; strict timeout
      │
      ▼
  6. Weighted Fusion      ← semantic dominates local; Gemini dominates all
      │
      ▼
  7. Threshold → Score    ← tiered floors, graceful fallback, 0–10 output

FIXES over previous version:
  - Abbreviation expansion ("SQL" → "Structured Query Language") before any scoring
  - Space normalisation ("data base" → "database") using compound-word collapse
  - TF-IDF weight reduced from 0.18 → 0.10 (was causing over-scoring on n-gram matches)
  - Semantic similarity now dominates local pipeline (weight: 0.42)
  - Precomputed embedding cache with LRU eviction (thread-safe, per question_id)
  - Every technique wrapped in its own try/except — one failure can't kill the score
  - Fallback chain: Gemini → local semantic → TF-IDF → keyword → heuristic
  - Score thresholds recalibrated: 0.85+ → 10, 0.70+ → 8-9, etc.
"""

import os
import re
import json
import hashlib
import time
import logging
import threading
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-1.5-flash"
MAX_RETRIES    = 2
RETRY_DELAY    = 1
GEMINI_TIMEOUT = 10          # seconds before falling back to local pipeline

# Embedding cache: question_id → numpy vector (avoids re-encoding expected answers)
_embedding_cache: dict = {}
_embedding_lock  = threading.Lock()
EMBEDDING_CACHE_MAX = 500    # evict oldest when limit reached

# Result cache: sha256(expected+student+question) → final result dict
_eval_cache: dict = {}
_cache_lock = threading.Lock()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — TEXT NORMALISATION
# Handles the root causes of unfair scoring:
#   • "data base" vs "database"   → collapse then compare
#   • "SQL" vs "Structured Query Language" → expand abbreviations first
#   • trailing punctuation, extra spaces, mixed case
# ═════════════════════════════════════════════════════════════════════════════

# Common technical/academic abbreviations → expand before any matching
ABBREVIATIONS = {
    "sql":    "structured query language",
    "dbms":   "database management system",
    "rdbms":  "relational database management system",
    "nosql":  "not only sql",
    "os":     "operating system",
    "cpu":    "central processing unit",
    "gpu":    "graphics processing unit",
    "ram":    "random access memory",
    "rom":    "read only memory",
    "hdd":    "hard disk drive",
    "ssd":    "solid state drive",
    "oop":    "object oriented programming",
    "api":    "application programming interface",
    "url":    "uniform resource locator",
    "http":   "hypertext transfer protocol",
    "https":  "hypertext transfer protocol secure",
    "html":   "hypertext markup language",
    "css":    "cascading style sheets",
    "xml":    "extensible markup language",
    "json":   "javascript object notation",
    "ai":     "artificial intelligence",
    "ml":     "machine learning",
    "dl":     "deep learning",
    "nlp":    "natural language processing",
    "nn":     "neural network",
    "cnn":    "convolutional neural network",
    "rnn":    "recurrent neural network",
    "io":     "input output",
    "ui":     "user interface",
    "ux":     "user experience",
    "ide":    "integrated development environment",
    "sdk":    "software development kit",
    "mvc":    "model view controller",
    "tcp":    "transmission control protocol",
    "ip":     "internet protocol",
    "dns":    "domain name system",
    "ftp":    "file transfer protocol",
    "lan":    "local area network",
    "wan":    "wide area network",
    "dna":    "deoxyribonucleic acid",
    "rna":    "ribonucleic acid",
    "atp":    "adenosine triphosphate",
    "co2":    "carbon dioxide",
    "h2o":    "water",
    "er":     "entity relationship",
    "erd":    "entity relationship diagram",
    "dfd":    "data flow diagram",
    "uml":    "unified modeling language",
    "crud":   "create read update delete",
    "acid":   "atomicity consistency isolation durability",
    "bst":    "binary search tree",
    "dll":    "doubly linked list",
    "lifo":   "last in first out",
    "fifo":   "first in first out",
    "adt":    "abstract data type",
    "avg":    "average",
    "max":    "maximum",
    "min":    "minimum",
}


def _expand_abbreviations(text: str) -> str:
    """
    Replace known abbreviations with full forms.
    Works on word boundaries so 'OS' in 'BOSS' is not expanded.
    """
    words = text.split()
    expanded = []
    for word in words:
        clean = word.lower().strip(".,;:!?()")
        if clean in ABBREVIATIONS:
            expanded.append(ABBREVIATIONS[clean])
        else:
            expanded.append(word)
    return " ".join(expanded)


def _collapse_compound_spaces(text: str) -> str:
    """
    'data base' → 'database', 'hard ware' → 'hardware'.
    Strategy: try collapsing adjacent word pairs and check against a known
    compound-word vocabulary. Falls back gracefully if pair isn't known.
    """
    COMPOUND_WORDS = {
        "database", "databases", "hardware", "software", "firmware",
        "keyboard", "touchscreen", "smartphone", "broadband", "bluetooth",
        "username", "password", "firewall", "malware", "ransomware",
        "frontend", "backend", "fullstack", "codebase", "runtime",
        "middleware", "namespace", "callback", "overload", "override",
        "underflow", "overflow", "deadlock", "blockchain", "timestamp",
        "checksum", "bitmap", "bytecode", "sourcecode", "microprocessor",
        "multiprocessing", "multithreading", "subprocess", "subprocess",
        "hyperlink", "hypertext", "localhost", "bandwidth", "throughput",
        "photosynthesis", "carbohydrate", "mitochondria", "chromosome",
        "electromagnetic", "thermodynamics", "semiconductors",
    }
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            pair = words[i].lower() + words[i + 1].lower()
            if pair in COMPOUND_WORDS:
                result.append(pair)
                i += 2
                continue
        result.append(words[i])
        i += 1
    return " ".join(result)


def normalise(text: str) -> str:
    """
    Full normalisation pipeline:
      1. Lowercase + strip
      2. Expand abbreviations (SQL → structured query language)
      3. Collapse compound spaces (data base → database)
      4. Remove punctuation
      5. Collapse whitespace
    This is the single entry point — all techniques call this first.
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = _expand_abbreviations(text)
    text = _collapse_compound_spaces(text)
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Stop-words used by keyword and concept matching
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "and",
    "or", "but", "not", "this", "that", "it", "its", "i", "we", "you",
    "he", "she", "they", "which", "who", "what", "when", "where", "how",
    "also", "so", "if", "then", "than", "about", "up", "out", "use",
    "our", "their", "there", "here", "just", "very", "much", "more",
    "some", "any", "all", "each", "both", "few", "those", "these", "such",
    "only", "same", "too", "used", "using", "called", "known", "defined",
}


def _meaningful_tokens(text: str) -> set:
    """
    Extract meaningful tokens after normalisation.
    Removes stopwords and very short words (≤2 chars).
    """
    norm = normalise(text)
    return {w for w in norm.split() if w not in STOPWORDS and len(w) > 2}


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — EXACT MATCH (fast-path)
# ═════════════════════════════════════════════════════════════════════════════

def _exact_match(expected: str, student: str) -> float:
    """Returns 1.0 if both answers normalise to identical strings."""
    return 1.0 if normalise(expected) == normalise(student) else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — KEYWORD OVERLAP (with length fairness)
# FIX: old version over-rewarded short answers that happened to share 1 token.
# New version: overlap is penalised by a length-fairness factor.
# ═════════════════════════════════════════════════════════════════════════════

def _keyword_match(expected: str, student: str) -> float:
    """
    Token overlap with two fairness corrections:
      a) Length penalty — a 2-word answer covering 2/5 tokens scores lower
         than a 10-word answer covering the same 2 tokens
      b) Over-answer bonus — answering more than required is never penalised
    """
    try:
        exp_tokens = _meaningful_tokens(expected)
        stu_tokens = _meaningful_tokens(student)

        if not exp_tokens:
            return 0.5   # no reference → neutral

        overlap = len(exp_tokens & stu_tokens)
        recall  = overlap / len(exp_tokens)    # how much of expected is covered

        # Length fairness: student should write roughly as much as expected
        exp_word_count = max(1, len(expected.split()))
        stu_word_count = max(1, len(student.split()))
        length_ratio   = min(1.0, stu_word_count / exp_word_count)

        # Weighted: recall is primary (80%), length ratio is penalty guard (20%)
        return round(recall * 0.80 + length_ratio * 0.20, 4)

    except Exception as e:
        logger.warning("keyword_match failed: %s", e)
        return 0.3   # safe fallback


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3b — TF-IDF COSINE (supporting signal, NOT dominant)
# FIX: weight reduced. TF-IDF is good for n-gram overlap but rewards
# surface similarity; semantic is far more reliable for subjective answers.
# ═════════════════════════════════════════════════════════════════════════════

def _tfidf_cosine(expected: str, student: str) -> float:
    """
    TF-IDF cosine similarity. Normalised text fed in to benefit from
    abbreviation expansion and compound-word collapse done upstream.
    Uses bigrams (1,2) to catch phrase-level similarity.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        exp_norm = normalise(expected)
        stu_norm = normalise(student)

        if not exp_norm or not stu_norm:
            return 0.0

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                              stop_words="english", sublinear_tf=True)
        mat = vec.fit_transform([exp_norm, stu_norm])
        score = float(cosine_similarity(mat[0], mat[1])[0][0])
        return max(0.0, min(1.0, score))

    except Exception as e:
        logger.warning("tfidf_cosine failed: %s — using keyword fallback", e)
        return _keyword_match(expected, student) * 0.80


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — SEMANTIC SIMILARITY (dominant local signal)
# Uses sentence-transformers MiniLM. Embeddings for expected answers are
# PRECOMPUTED and cached by question_id to avoid re-encoding on each submission.
# ═════════════════════════════════════════════════════════════════════════════

# Singleton model — loaded once, reused across all requests
_st_model       = None
_st_model_lock  = threading.Lock()


def _get_st_model():
    """Thread-safe lazy loader for SentenceTransformer."""
    global _st_model
    if _st_model is None:
        with _st_model_lock:
            if _st_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    _st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
                    logger.info("SentenceTransformer (MiniLM) loaded successfully.")
                except Exception as e:
                    logger.warning("SentenceTransformer load failed: %s", e)
                    _st_model = "FAILED"
    return None if _st_model == "FAILED" else _st_model


def get_expected_embedding(question_id: int, expected_text: str):
    """
    Returns the embedding for an expected answer.
    CACHES by question_id — avoids re-encoding the same expected answer
    for every student submission. This is the key performance optimization.
    """
    with _embedding_lock:
        if question_id in _embedding_cache:
            return _embedding_cache[question_id]

    model = _get_st_model()
    if model is None:
        return None

    try:
        embedding = model.encode([normalise(expected_text)])[0]

        with _embedding_lock:
            # LRU eviction: if cache is too large, drop oldest entry
            if len(_embedding_cache) >= EMBEDDING_CACHE_MAX:
                oldest_key = next(iter(_embedding_cache))
                del _embedding_cache[oldest_key]
            _embedding_cache[question_id] = embedding

        return embedding

    except Exception as e:
        logger.warning("Embedding failed for q%s: %s", question_id, e)
        return None


def _semantic_similarity(expected: str, student: str,
                          question_id: Optional[int] = None) -> float:
    """
    Semantic similarity via MiniLM cosine similarity.
    If question_id is provided, uses cached expected embedding (faster).
    Falls back to TF-IDF if model is unavailable.

    FIX: threshold changed. Old code was thresholding at 0.5, which caused
    correct paraphrased answers to score < 50%. Now raw cosine is returned
    and thresholding is done only at the final scoring stage.
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        import numpy as np

        model = _get_st_model()
        if model is None:
            return _tfidf_cosine(expected, student)

        # Use cached expected embedding if question_id is provided
        if question_id is not None:
            exp_emb = get_expected_embedding(question_id, expected)
        else:
            exp_emb = model.encode([normalise(expected)])[0]

        if exp_emb is None:
            return _tfidf_cosine(expected, student)

        stu_emb = model.encode([normalise(student)])[0]

        score = float(cos_sim([exp_emb], [stu_emb])[0][0])
        return max(0.0, min(1.0, score))

    except Exception as e:
        logger.warning("semantic_similarity failed: %s — using TF-IDF", e)
        return _tfidf_cosine(expected, student)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — GEMINI AI (optional dominant signal)
# Runs in a thread with strict timeout. Evaluates semantic quality,
# key point coverage, and provides human-readable feedback.
# ═════════════════════════════════════════════════════════════════════════════

GEMINI_PROMPT_WITH_EXPECTED = """\
You are a strict but fair academic evaluator scoring a student's answer.

Return ONLY valid JSON with no markdown fences:
{{
  "semantic_score": <float 0.0–1.0>,
  "feedback": "<2–3 specific sentences: what is correct, what is missing, how to improve>",
  "key_points_covered": ["<concept 1>", "<concept 2>"],
  "key_points_missing": ["<concept A>", "<concept B>"]
}}

SCORING GUIDE:
- 0.90–1.00: All key concepts covered (wording may differ — synonyms count)
- 0.75–0.89: Very good, only minor details missing
- 0.55–0.74: Good, main idea present but important details absent
- 0.35–0.54: Partial — some relevant content but significant gaps
- 0.15–0.34: Weak — barely touches the topic
- 0.00–0.14: Wrong or completely irrelevant

RULES:
- Synonyms, paraphrasing, and abbreviation expansions = full credit for that concept
- "SQL" and "Structured Query Language" are the same — never penalise abbreviations
- "data base" and "database" are the same — never penalise spacing variants
- Spelling/grammar errors do NOT reduce score unless meaning is lost
- Feedback MUST name specific concepts, not generic phrases like "good answer"
- Never return exactly 0.0 unless the answer is empty or gibberish
- Never return exactly 1.0 unless the answer is flawlessly complete

EXPECTED ANSWER:
\"\"\"{expected}\"\"\"

STUDENT ANSWER:
\"\"\"{student}\"\"\"\
"""

GEMINI_PROMPT_NO_EXPECTED = """\
You are a strict but fair academic evaluator.
The teacher did NOT provide an expected answer. Evaluate the student's answer
based solely on your subject-matter knowledge.

Question: \"\"\"{question}\"\"\"
Student Answer: \"\"\"{student}\"\"\"

Return ONLY valid JSON with no markdown fences:
{{
  "semantic_score": <float 0.0–1.0>,
  "feedback": "<2–3 specific sentences about correctness, completeness, and how to improve>",
  "key_points_covered": ["<concept correctly mentioned>"],
  "key_points_missing": ["<important concept not mentioned>"]
}}

RULES:
- Evaluate on factual correctness and completeness only
- Synonyms and paraphrasing count as full credit
- Never return exactly 0.0 unless the answer is empty or gibberish\
"""


def _gemini_evaluate(expected: str, student: str,
                     question: str = "") -> Optional[dict]:
    """
    Call Gemini with a daemon thread + strict timeout.
    Returns None if API key is missing, request times out, or JSON is invalid.
    The calling code MUST handle None (falls back to local pipeline).
    """
    if not GEMINI_API_KEY:
        return None

    result_holder = [None]
    error_holder  = [None]

    def _call():
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)

            prompt = (
                GEMINI_PROMPT_WITH_EXPECTED.format(expected=expected, student=student)
                if expected.strip()
                else GEMINI_PROMPT_NO_EXPECTED.format(
                    question=question or "Unknown question", student=student
                )
            )

            for attempt in range(MAX_RETRIES + 1):
                try:
                    resp = model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": 0,
                            "max_output_tokens": 500,
                            "response_mime_type": "application/json",
                        },
                    )
                    text = resp.text.strip()
                    # Strip markdown fences if model ignores mime type
                    text = re.sub(r"^```json\s*", "", text)
                    text = re.sub(r"```$", "", text).strip()

                    data  = json.loads(text)
                    score = float(data.get("semantic_score", 0.5))
                    score = max(0.01, min(0.99, score))

                    covered = data.get("key_points_covered", [])
                    missing = data.get("key_points_missing", [])
                    covered = [str(x).strip() for x in covered if str(x).strip()]
                    missing = [str(x).strip() for x in missing if str(x).strip()]

                    result_holder[0] = {
                        "score":              score,
                        "feedback":           str(data.get("feedback", "")).strip(),
                        "key_points_covered": covered,
                        "key_points_missing": missing,
                    }
                    return

                except (json.JSONDecodeError, Exception):
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                    else:
                        error_holder[0] = "Max retries exceeded"

        except Exception as e:
            error_holder[0] = e

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=GEMINI_TIMEOUT)

    if t.is_alive():
        logger.warning("Gemini timed out after %ss — falling back to local.", GEMINI_TIMEOUT)
        return None
    if error_holder[0]:
        logger.warning("Gemini error: %s — falling back to local.", error_holder[0])
        return None
    return result_holder[0]


# ═════════════════════════════════════════════════════════════════════════════
# SYNONYM MATCHING — domain-aware vocabulary map
# ═════════════════════════════════════════════════════════════════════════════

SYNONYM_GROUPS = [
    # Computer Science / DB
    {"database", "db", "datastore", "repository", "data store"},
    {"query", "request", "search", "retrieve", "fetch", "lookup"},
    {"table", "relation", "entity", "record"},
    {"primary key", "unique key", "identifier", "id"},
    {"index", "indexing", "indices"},
    {"transaction", "operation", "commit"},
    {"join", "merge", "combine", "link"},
    {"schema", "structure", "layout", "design"},
    {"normalization", "normalisation", "standardization"},
    {"concurrency", "parallel", "simultaneous"},
    # Programming
    {"function", "method", "procedure", "routine", "subroutine"},
    {"class", "blueprint", "template", "prototype"},
    {"object", "instance", "entity"},
    {"variable", "identifier", "name", "symbol"},
    {"loop", "iteration", "cycle", "repetition"},
    {"condition", "conditional", "branch", "decision"},
    {"array", "list", "collection", "sequence"},
    {"pointer", "reference", "address"},
    {"recursion", "recursive", "self-referential"},
    {"inheritance", "derive", "extend", "subclass"},
    {"encapsulation", "wrapping", "hiding", "bundling"},
    {"polymorphism", "overloading", "overriding"},
    {"abstraction", "interface", "contract"},
    # General academic
    {"increase", "grow", "rise", "expand", "enhance", "improve", "escalate"},
    {"decrease", "reduce", "fall", "decline", "diminish", "lessen"},
    {"important", "significant", "crucial", "vital", "essential", "key", "critical"},
    {"process", "procedure", "method", "approach", "technique", "mechanism"},
    {"store", "save", "retain", "persist", "hold", "keep"},
    {"transfer", "transmit", "send", "move", "relay"},
    {"convert", "transform", "change", "alter", "modify"},
    {"compare", "contrast", "differentiate", "distinguish"},
    {"define", "describe", "explain", "state", "specify"},
    {"use", "utilize", "employ", "apply", "leverage"},
    {"create", "generate", "produce", "make", "build", "construct"},
    {"delete", "remove", "erase", "drop", "eliminate"},
    {"access", "retrieve", "fetch", "get", "read"},
    {"update", "modify", "edit", "change", "alter"},
    {"manage", "control", "handle", "govern", "administer"},
    {"protect", "secure", "guard", "defend", "shield"},
    {"allow", "permit", "enable", "authorize", "grant"},
    {"prevent", "restrict", "block", "prohibit", "disallow"},
    {"fast", "quick", "rapid", "efficient", "speedy"},
    {"slow", "sluggish", "inefficient", "delayed"},
    {"simple", "easy", "straightforward", "uncomplicated"},
    {"complex", "complicated", "difficult", "hard", "challenging"},
    {"small", "little", "tiny", "minimal", "compact"},
    {"large", "big", "huge", "massive", "extensive", "vast"},
    # Science
    {"energy", "power", "force", "strength"},
    {"cell", "unit", "component", "element"},
    {"organism", "living thing", "creature", "being"},
    {"photosynthesis", "light reaction", "carbon fixation"},
    {"respiration", "breathing", "gas exchange"},
    {"atom", "particle", "molecule"},
    {"reaction", "process", "interaction"},
]


def _build_synonym_map() -> dict:
    """Map each word to a group index for O(1) synonym lookup."""
    m = {}
    for i, grp in enumerate(SYNONYM_GROUPS):
        for w in grp:
            for token in w.split():   # multi-word phrases: map each token
                if token not in m:
                    m[token] = set()
                m[token].add(i)
    return m


_SYNONYM_MAP = _build_synonym_map()


def _synonym_match(expected: str, student: str) -> float:
    """
    Matches tokens that are synonyms (same group index).
    Falls back to NLTK WordNet if available, otherwise uses built-in map.
    """
    exp_tokens = _meaningful_tokens(expected)
    stu_tokens = _meaningful_tokens(student)
    if not exp_tokens:
        return 0.5

    def _groups(word: str) -> set:
        """Return all synonym group indices a word belongs to."""
        return _SYNONYM_MAP.get(word, set())

    try:
        import nltk
        from nltk.corpus import wordnet
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)

        def _wn_synonyms(word: str) -> set:
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
            # WordNet synonym check
            ew_syns = _wn_synonyms(ew)
            if ew_syns & stu_tokens:
                matched += 0.92
                continue
            # Built-in group check
            ew_groups = _groups(ew)
            if ew_groups and any(_groups(sw) & ew_groups for sw in stu_tokens):
                matched += 0.85
                continue
            # Stem prefix check (last resort)
            if len(ew) > 5:
                stem = ew[:max(4, len(ew) - 3)]
                if any(sw.startswith(stem) for sw in stu_tokens):
                    matched += 0.65

        return min(1.0, matched / len(exp_tokens))

    except Exception:
        # Pure built-in map fallback (no NLTK)
        matched = 0.0
        for ew in exp_tokens:
            if ew in stu_tokens:
                matched += 1.0
                continue
            ew_groups = _groups(ew)
            if ew_groups and any(_groups(sw) & ew_groups for sw in stu_tokens):
                matched += 0.85
                continue
            if len(ew) > 5:
                stem = ew[:max(4, len(ew) - 3)]
                if any(sw.startswith(stem) for sw in stu_tokens):
                    matched += 0.65
        return min(1.0, matched / len(exp_tokens))


# ═════════════════════════════════════════════════════════════════════════════
# SCORING WEIGHTS
# Design principle:
#   - Semantic (MiniLM) is the most reliable local signal → highest weight
#   - Keyword + synonym support semantic, catch what embeddings miss
#   - TF-IDF is a weak supporting signal only — was over-weighted before
#   - Gemini dominates all when available (it reasons about meaning)
#
# Weights MUST sum to 1.0 within each set.
# ═════════════════════════════════════════════════════════════════════════════

WEIGHTS_WITH_GEMINI = {
    "exact_match":   0.02,   # fast-path already handled before this point
    "keyword_match": 0.08,
    "tfidf_cosine":  0.08,   # FIX: reduced from 0.18 — was over-weighted
    "semantic_sim":  0.18,   # MiniLM supports Gemini
    "gemini":        0.42,   # dominant — reasons about meaning not surface
    "synonym_match": 0.12,
    "concept_match": 0.10,
}

WEIGHTS_LOCAL_ONLY = {
    "exact_match":   0.04,
    "keyword_match": 0.15,
    "tfidf_cosine":  0.10,   # FIX: reduced from 0.18
    "semantic_sim":  0.42,   # FIX: dominant when Gemini unavailable
    "synonym_match": 0.16,
    "concept_match": 0.13,
}

assert abs(sum(WEIGHTS_WITH_GEMINI.values()) - 1.0) < 1e-9, "Gemini weights must sum to 1.0"
assert abs(sum(WEIGHTS_LOCAL_ONLY.values())  - 1.0) < 1e-9, "Local weights must sum to 1.0"


# ═════════════════════════════════════════════════════════════════════════════
# CONCEPT MATCH (stem-based, synonym-aware)
# Separate from keyword_match: tries harder via stems + synonym groups
# ═════════════════════════════════════════════════════════════════════════════

def _concept_match(expected: str, student: str) -> float:
    """
    Like keyword match but more aggressive:
    exact → synonym group → 5-char stem prefix
    """
    try:
        exp_tokens = _meaningful_tokens(expected)
        stu_tokens = _meaningful_tokens(student)
        if not exp_tokens:
            return 0.5

        matched = 0.0
        for ew in exp_tokens:
            if ew in stu_tokens:
                matched += 1.0
                continue
            ew_groups = _SYNONYM_MAP.get(ew, set())
            if ew_groups and any(_SYNONYM_MAP.get(sw, set()) & ew_groups
                                 for sw in stu_tokens):
                matched += 0.85
                continue
            if len(ew) > 5:
                stem = ew[:max(4, len(ew) - 3)]
                if any(sw.startswith(stem) for sw in stu_tokens):
                    matched += 0.65

        return min(1.0, matched / len(exp_tokens))

    except Exception as e:
        logger.warning("concept_match failed: %s", e)
        return 0.3


# ═════════════════════════════════════════════════════════════════════════════
# SCORE FUSION + THRESHOLD LOGIC
# FIX: tiered floors recalibrated using content_signal composite.
# content_signal blends semantic, keyword, and synonym so neither alone
# can unfairly push the score up or down.
# ═════════════════════════════════════════════════════════════════════════════

def _compute_content_signal(scores: dict, used_gemini: bool) -> float:
    """
    Composite content quality signal used for floor/threshold decisions.
    Blended to prevent any single metric from dominating.
    """
    gem = scores.get("gemini", 0.5) if used_gemini else 0.0
    sem = scores.get("semantic_sim", 0.0)
    kw  = scores.get("keyword_match", 0.0)
    syn = scores.get("synonym_match", 0.0)
    con = scores.get("concept_match", 0.0)

    if used_gemini:
        # Gemini leads
        return gem * 0.50 + sem * 0.25 + kw * 0.10 + syn * 0.08 + con * 0.07
    else:
        # Semantic leads locally
        return sem * 0.50 + kw * 0.20 + syn * 0.15 + con * 0.15


def _apply_thresholds(raw_score: float, content_signal: float) -> int:
    """
    Convert normalised score (0–1) to integer grade (0–10).
    Thresholds designed so:
      - Correct paraphrased answers reach 8–10
      - Partial answers get 4–7
      - Off-topic / empty get 0–3

    FIX: Old thresholds were too conservative, causing correct answers
    to be capped at 7. Recalibrated for fairness.
    """
    # Perfect or near-perfect
    if content_signal >= 0.93 and raw_score >= 0.95:
        return 10
    if content_signal >= 0.88:
        return 9
    if content_signal >= 0.80:
        return 8
    if content_signal >= 0.70:
        return 7
    if content_signal >= 0.58:
        return 6
    if content_signal >= 0.45:
        return 5
    if content_signal >= 0.33:
        return 4
    if content_signal >= 0.20:
        return 3
    if content_signal >= 0.10:
        return 2
    if content_signal >= 0.04:
        return 1
    return 0


def _build_local_feedback(score: int, scores: dict, covered: list,
                           missing: list, no_expected: bool) -> str:
    """Generate meaningful, metric-aware feedback without Gemini."""
    sem = scores.get("semantic_sim", 0)
    kw  = scores.get("keyword_match", 0)
    con = scores.get("concept_match", 0)

    covered_str = ", ".join(covered[:3]) if covered else None
    missing_str = ", ".join(missing[:3]) if missing else None

    if score == 10:
        return ("Perfect — your answer covers all key concepts with accuracy. "
                "The meaning aligns completely with the expected answer.")
    if score >= 8:
        msg = (f"Strong answer (semantic similarity: {int(sem * 100)}%, "
               f"keyword coverage: {int(kw * 100)}%). ")
        msg += (f"Minor gaps: {missing_str}." if missing_str
                else "Only minor elaboration could improve this further.")
        return msg
    if score >= 6:
        msg = f"Good — main idea is present (semantic: {int(sem * 100)}%). "
        if missing_str:
            msg += f"Key missing concepts: {missing_str}. "
        if covered_str:
            msg += f"Correctly addressed: {covered_str}."
        return msg
    if score >= 4:
        msg = f"Partial answer (concept match: {int(con * 100)}%). "
        if covered_str:
            msg += f"You correctly mentioned: {covered_str}. "
        if missing_str:
            msg += f"Important missing concepts: {missing_str}."
        else:
            msg += "Significant portions of the expected answer are absent."
        return msg
    if score >= 2:
        msg = f"Weak — limited relevant content (keyword match: {int(kw * 100)}%). "
        if missing_str:
            msg += f"Missing core concepts: {missing_str}. "
        msg += "Review the material and include specific terminology."
        return msg
    if score == 1:
        return ("The answer barely touches the topic. "
                "Most key concepts from the expected answer are absent. "
                "Please review the material thoroughly.")
    return ("No answer submitted, or the response does not address the question." if not no_expected
            else "No answer was submitted, or the response was entirely off-topic.")


def _build_result(raw: float, scores: dict, feedback: str,
                  covered: list, missing: list, used_gemini: bool,
                  no_expected: bool = False) -> dict:
    """Apply threshold logic and build the final result dictionary."""
    content_signal = _compute_content_signal(scores, used_gemini)
    final_score    = _apply_thresholds(raw, content_signal)

    if not feedback:
        feedback = _build_local_feedback(
            final_score, scores, covered, missing, no_expected
        )

    logger.info(
        "Eval → sem:%.2f kw:%.2f tfidf:%.2f gem:%.2f syn:%.2f con:%.2f "
        "| signal:%.3f raw:%.3f → %d (gemini=%s no_exp=%s)",
        scores.get("semantic_sim", 0), scores.get("keyword_match", 0),
        scores.get("tfidf_cosine", 0), scores.get("gemini", 0),
        scores.get("synonym_match", 0), scores.get("concept_match", 0),
        content_signal, raw, final_score, used_gemini, no_expected,
    )

    return {
        "score":              int(max(0, min(10, final_score))),
        "feedback":           feedback,
        "key_points_covered": covered,
        "key_points_missing": missing,
        "breakdown":          {k: round(v * 10, 2) for k, v in scores.items()},
        "used_gemini":        used_gemini,
        "no_expected_answer": no_expected,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CORE PIPELINES
# ═════════════════════════════════════════════════════════════════════════════

def _evaluate_with_expected(expected: str, student: str,
                             question_id: Optional[int] = None) -> dict:
    """
    Full hybrid pipeline when an expected answer exists.
    All local techniques run first (fast), then Gemini (async, with timeout).
    """
    # Run all local techniques — each wrapped separately so one failure
    # doesn't cascade and kill the entire score
    scores = {
        "exact_match":   _exact_match(expected, student),
        "keyword_match": _keyword_match(expected, student),
        "tfidf_cosine":  _tfidf_cosine(expected, student),
        "semantic_sim":  _semantic_similarity(expected, student, question_id),
        "gemini":        0.5,   # placeholder; overwritten below if Gemini succeeds
        "synonym_match": _synonym_match(expected, student),
        "concept_match": _concept_match(expected, student),
    }

    feedback    = ""
    covered     = []
    missing     = []
    used_gemini = False

    gemini_result = _gemini_evaluate(expected, student)
    if gemini_result:
        scores["gemini"] = gemini_result["score"]
        feedback    = gemini_result.get("feedback", "")
        covered     = gemini_result.get("key_points_covered", [])
        missing     = gemini_result.get("key_points_missing", [])
        used_gemini = True
        weights     = WEIGHTS_WITH_GEMINI
    else:
        # Gemini failed — redistribute weight across local techniques
        weights = WEIGHTS_LOCAL_ONLY

    raw = sum(scores.get(k, 0) * w for k, w in weights.items())
    return _build_result(raw, scores, feedback, covered, missing, used_gemini, no_expected=False)


def _evaluate_without_expected(student: str, question: str = "") -> dict:
    """
    Evaluation when NO expected answer is stored.
    Gemini evaluates on its own subject knowledge.
    Local heuristic (length + vocabulary richness) is the fallback.
    """
    gemini_result = _gemini_evaluate(expected="", student=student, question=question)

    if gemini_result:
        gem_score = gemini_result["score"]
        feedback  = gemini_result.get("feedback", "")
        covered   = gemini_result.get("key_points_covered", [])
        missing   = gemini_result.get("key_points_missing", [])

        final_score = max(1, min(9, round(gem_score * 10)))
        if gem_score < 0.05:
            final_score = 0

        if not feedback:
            feedback = _build_local_feedback(
                final_score,
                {"semantic_sim": gem_score, "keyword_match": gem_score,
                 "concept_match": gem_score, "synonym_match": gem_score},
                covered, missing, no_expected=True,
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

    # No Gemini — heuristic based on length and vocabulary richness
    words        = student.split()
    word_count   = len(words)
    unique_ratio = len({w.lower() for w in words}) / max(1, word_count)
    meaningful   = len(_meaningful_tokens(student))

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

    heuristic   = min(0.65, heuristic)   # cap at 6.5 — we can't verify correctness
    final_score = max(1, min(6, round(heuristic * 10))) if word_count > 0 else 0

    return {
        "score":              final_score,
        "feedback":           (
            f"No expected answer is on file. Your answer was evaluated on "
            f"length ({word_count} words) and vocabulary richness "
            f"({int(unique_ratio * 100)}% unique words). "
            f"Ask your teacher to add an expected answer for a more accurate score."
        ),
        "key_points_covered": [],
        "key_points_missing": [],
        "breakdown":          {"heuristic": round(heuristic * 10, 2)},
        "used_gemini":        False,
        "no_expected_answer": True,
    }


# ═════════════════════════════════════════════════════════════════════════════
# AUTO-GENERATE EXPECTED ANSWER
# ═════════════════════════════════════════════════════════════════════════════

def generate_expected_answer(question_text: str) -> str:
    """
    Use Gemini to generate a reference answer for a question.
    Called by the teacher interface when no expected answer is provided.
    Returns empty string on any failure.
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
            generation_config={"temperature": 0.3, "max_output_tokens": 350},
        )
        return response.text.strip()
    except Exception as e:
        logger.error("generate_expected_answer failed: %s", e)
        return ""


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API — call these from admin.py, never internals directly
# ═════════════════════════════════════════════════════════════════════════════

def ai_evaluate(expected_answer: str, student_answer: str,
                question_text: str = "",
                question_id: Optional[int] = None) -> dict:
    """
    Evaluate a student answer against an expected answer.

    Parameters
    ----------
    expected_answer : str
        Teacher's reference answer. May be empty — Gemini evaluates on its
        own knowledge if the API key is set.
    student_answer  : str
        The student's submitted answer.
    question_text   : str
        The question text. Required when expected_answer is empty so Gemini
        has context. Also helps the local pipeline's synonym/concept matching.
    question_id     : int | None
        If provided, expected_answer embedding is cached by this ID,
        avoiding re-encoding for every student who answers the same question.

    Returns
    -------
    dict with keys:
        score              : int (0–10)
        feedback           : str
        key_points_covered : list[str]
        key_points_missing : list[str]
        breakdown          : dict (per-technique scores ×10)
        used_gemini        : bool
        no_expected_answer : bool
    """
    expected = (expected_answer or "").strip()
    student  = (student_answer  or "").strip()

    # ── Fast-paths ────────────────────────────────────────────────────────────
    if not student:
        return {
            "score": 0, "feedback": "No answer was submitted.",
            "key_points_covered": [], "key_points_missing": [],
            "breakdown": {}, "used_gemini": False,
            "no_expected_answer": not bool(expected),
        }

    if expected and normalise(expected) == normalise(student):
        return {
            "score": 10,
            "feedback": "Perfect match — your answer covers all required concepts.",
            "key_points_covered": [], "key_points_missing": [],
            "breakdown": {}, "used_gemini": False, "no_expected_answer": False,
        }

    # ── Cache check ───────────────────────────────────────────────────────────
    cache_raw = f"{expected}|||{student}|||{question_text}"
    cache_key = hashlib.sha256(cache_raw.lower().strip().encode()).hexdigest()
    with _cache_lock:
        if cache_key in _eval_cache:
            return _eval_cache[cache_key]

    # ── Route to correct pipeline ─────────────────────────────────────────────
    if expected:
        result = _evaluate_with_expected(expected, student, question_id)
    else:
        result = _evaluate_without_expected(student, question=question_text)

    with _cache_lock:
        _eval_cache[cache_key] = result

    return result


def ai_evaluate_safe(expected_answer: str, student_answer: str,
                     question_text: str = "",
                     question_id: Optional[int] = None,
                     fallback_score: Optional[int] = None) -> dict:
    """
    Wrapper that NEVER raises. Always returns a valid result dict.
    This is what admin.py should call — never call ai_evaluate() directly
    from routes.
    """
    try:
        return ai_evaluate(expected_answer, student_answer,
                           question_text, question_id)
    except Exception as e:
        logger.error("ai_evaluate_safe unhandled error: %s", e)
        score = fallback_score if fallback_score is not None else 0
        return {
            "score":              score,
            "feedback":           "Evaluation service temporarily unavailable.",
            "key_points_covered": [],
            "key_points_missing": [],
            "breakdown":          {},
            "used_gemini":        False,
            "no_expected_answer": not bool((expected_answer or "").strip()),
            "error":              str(e),
        }
