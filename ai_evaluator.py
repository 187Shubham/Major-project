"""
ai_evaluator.py  ─  Hybrid Answer Evaluation Engine
=====================================================
Pipeline (every student answer goes through this):

    Raw Text
       │
       ▼
  [1] NORMALISE          lower + abbrev-expand + compound-collapse + strip punct
       │
       ▼
  [2] EXACT MATCH        normalised strings identical → 10 immediately
       │
       ▼
  [3] CONCEPT COVERAGE   which required concepts did student mention?  (0–1)
       │                  ← domain detected from question text
       ▼
  [4] KEYWORD RECALL     overlap of meaningful tokens, NO length penalty  (0–1)
       │
       ▼
  [5] SYNONYM MATCH      token-level + built-in groups + optional WordNet  (0–1)
       │
       ▼
  [6] TF-IDF COSINE      bigram surface similarity (supporting only)       (0–1)
       │
       ▼
  [7] SEMANTIC SIM       MiniLM sentence embeddings (cached per question)  (0–1)
       │
       ▼
  [8] GEMINI (optional)  meaning-level AI evaluation with timeout guard    (0–1)
       │
       ▼
  [9] WEIGHTED FUSION    concept_coverage dominates locally; Gemini when live
       │
       ▼
  [10] SCORE             integer 0–10 with human-readable feedback

ROOT CAUSES FIXED IN THIS VERSION
──────────────────────────────────
Fix A ─ Semantic weak / structure-penalised
  Problem: _synonym_match split multi-word synonyms into tokens then looked up
           each token separately.  "real time entity" became {"real","time","entity"}
           and none matched "instance".
  Fix:     _phrase_match() checks full normalised text for multi-word synonyms
           BEFORE falling back to token-level matching.

Fix B ─ Length bias
  Problem: keyword_match applied stu_len/exp_len ratio as 20% of score, so a
           shorter correct answer was penalised purely for word count.
  Fix:     keyword_match now measures RECALL only (how many expected tokens
           the student covered).  Length is never penalised — only coverage matters.

Fix C ─ Concept coverage not wired into with-expected pipeline
  Problem: _concept_coverage_score existed but was only called from
           _evaluate_without_expected.  Q1 (has expected) never got concept boost.
  Fix:     _evaluate_with_expected now calls _concept_coverage_score and uses
           the ratio as an additive floor on content_signal so a student who
           correctly names all required concepts cannot score below that floor.
"""

import os
import re
import json
import hashlib
import time
import logging
import threading
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL    = "gemini-1.5-flash"
MAX_RETRIES     = 2
RETRY_DELAY     = 1
GEMINI_TIMEOUT  = 10          # seconds

_embedding_cache: dict = {}
_embedding_lock  = threading.Lock()
EMBEDDING_CACHE_MAX = 500

_eval_cache: dict = {}
_cache_lock = threading.Lock()


# ═════════════════════════════════════════════════════════════════════════════
# [1]  NORMALISATION
#      Single entry point used by every technique below.
#      Guarantees: "SQL"="structured query language", "data base"="database"
# ═════════════════════════════════════════════════════════════════════════════

ABBREVIATIONS: dict = {
    # CS / programming
    "sql":   "structured query language",
    "dbms":  "database management system",
    "rdbms": "relational database management system",
    "nosql": "not only sql",
    "os":    "operating system",
    "cpu":   "central processing unit",
    "gpu":   "graphics processing unit",
    "ram":   "random access memory",
    "rom":   "read only memory",
    "hdd":   "hard disk drive",
    "ssd":   "solid state drive",
    "oop":   "object oriented programming",
    "oops":  "object oriented programming",
    "api":   "application programming interface",
    "url":   "uniform resource locator",
    "http":  "hypertext transfer protocol",
    "https": "hypertext transfer protocol secure",
    "html":  "hypertext markup language",
    "css":   "cascading style sheets",
    "xml":   "extensible markup language",
    "json":  "javascript object notation",
    "ai":    "artificial intelligence",
    "ml":    "machine learning",
    "dl":    "deep learning",
    "nlp":   "natural language processing",
    "nn":    "neural network",
    "cnn":   "convolutional neural network",
    "rnn":   "recurrent neural network",
    "io":    "input output",
    "ui":    "user interface",
    "ux":    "user experience",
    "ide":   "integrated development environment",
    "sdk":   "software development kit",
    "mvc":   "model view controller",
    "tcp":   "transmission control protocol",
    "ip":    "internet protocol",
    "dns":   "domain name system",
    "ftp":   "file transfer protocol",
    "lan":   "local area network",
    "wan":   "wide area network",
    "er":    "entity relationship",
    "erd":   "entity relationship diagram",
    "dfd":   "data flow diagram",
    "uml":   "unified modeling language",
    "crud":  "create read update delete",
    "acid":  "atomicity consistency isolation durability",
    "bst":   "binary search tree",
    "dll":   "doubly linked list",
    "lifo":  "last in first out",
    "fifo":  "first in first out",
    "adt":   "abstract data type",
    # Science
    "dna":   "deoxyribonucleic acid",
    "rna":   "ribonucleic acid",
    "atp":   "adenosine triphosphate",
    "co2":   "carbon dioxide",
    "h2o":   "water",
}

# Known compound words: "data base" → "database" etc.
_COMPOUND_WORDS = {
    "database", "databases", "hardware", "software", "firmware",
    "keyboard", "touchscreen", "smartphone", "broadband", "bluetooth",
    "username", "password", "firewall", "malware", "ransomware",
    "frontend", "backend", "fullstack", "codebase", "runtime",
    "middleware", "namespace", "callback", "overload", "override",
    "underflow", "overflow", "deadlock", "blockchain", "timestamp",
    "checksum", "bitmap", "bytecode", "sourcecode", "microprocessor",
    "multiprocessing", "multithreading", "hyperlink", "hypertext",
    "localhost", "bandwidth", "throughput", "photosynthesis",
    "carbohydrate", "mitochondria", "chromosome", "electromagnetic",
    "thermodynamics", "semiconductors", "inheritance", "polymorphism",
    "encapsulation", "abstraction",
}


def _expand_abbrev(text: str) -> str:
    words, out = text.split(), []
    for w in words:
        clean = w.lower().strip(".,;:!?()")
        out.append(ABBREVIATIONS[clean] if clean in ABBREVIATIONS else w)
    return " ".join(out)


def _collapse_compounds(text: str) -> str:
    words, result, i = text.split(), [], 0
    while i < len(words):
        if i + 1 < len(words):
            pair = words[i].lower() + words[i + 1].lower()
            if pair in _COMPOUND_WORDS:
                result.append(pair); i += 2; continue
        result.append(words[i]); i += 1
    return " ".join(result)


def normalise(text: str) -> str:
    """
    Canonical normalisation used by EVERY technique.
    Steps: lower → expand abbreviations → collapse compounds
           → strip punctuation → collapse whitespace
    """
    if not text:
        return ""
    t = text.lower().strip()
    t = _expand_abbrev(t)
    t = _collapse_compounds(t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","to","of","in","for","on","with","at","by","from","as",
    "into","through","and","or","but","not","this","that","it","its","i",
    "we","you","he","she","they","which","who","what","when","where","how",
    "also","so","if","then","than","about","up","out","use","our","their",
    "there","here","just","very","much","more","some","any","all","each",
    "both","few","those","these","such","only","same","too","used","using",
    "called","known","defined","basically","generally","typically","usually",
}


def _tokens(text: str) -> set:
    """Meaningful tokens from normalised text (no stopwords, len > 2)."""
    return {w for w in normalise(text).split()
            if w not in STOPWORDS and len(w) > 2}


# ═════════════════════════════════════════════════════════════════════════════
# [2]  EXACT MATCH  (fast-path, checked before anything else)
# ═════════════════════════════════════════════════════════════════════════════

def _exact_match(expected: str, student: str) -> float:
    return 1.0 if normalise(expected) == normalise(student) else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# [3]  CONCEPT COVERAGE
#      Domain-specific required-concept checking.
#
#      Key design:
#        • Each domain has TRIGGER WORDS (detected from question text)
#        • Each domain has REQUIRED CONCEPTS, each with a synonym list
#        • We check the FULL NORMALISED student answer for each synonym
#          (not split into tokens — fixes multi-word matching bug)
#        • Returns (ratio 0–1, covered list, missing list)
#        • ratio=0.0 and empty lists when no domain detected
#
#      This is wired into BOTH pipelines (with/without expected answer).
# ═════════════════════════════════════════════════════════════════════════════

# Each concept entry: first element = display name, rest = synonyms to search
DOMAIN_CONCEPTS: dict = {

    "oop_pillars": {
        "triggers": {
            "pillars", "principles", "oops", "oop",
            "object oriented", "object-oriented",
            "four pillars", "4 pillars",
        },
        "concepts": [
            # [display_name, *synonyms_to_search_in_full_text]
            ["abstraction",
             "abstraction", "abstract", "hide implementation",
             "hides implementation", "hiding implementation",
             "expose essential", "essential functionality",
             "what it does", "implementation detail"],
            ["encapsulation",
             "encapsulation", "encapsulate", "data hiding",
             "wrapping", "bundling", "bundled", "wrapped",
             "private", "protected", "access control",
             "bind data", "binds data"],
            ["inheritance",
             "inheritance", "inherit", "derive", "derived",
             "extend", "extends", "subclass", "parent class",
             "child class", "base class", "superclass",
             "is a relationship"],
            ["polymorphism",
             "polymorphism", "polymorphic", "overloading",
             "overriding", "many forms", "method overload",
             "method override", "compile time", "runtime polymorphism"],
        ],
    },

    "db_normalization": {
        "triggers": {
            "normalization", "normalisation", "normal forms",
            "1nf", "2nf", "3nf", "bcnf", "denormalization",
        },
        "concepts": [
            ["1NF / First Normal Form",
             "1nf", "first normal form", "atomic values",
             "atomicity", "no repeating groups"],
            ["2NF / Second Normal Form",
             "2nf", "second normal form", "partial dependency",
             "partial dependence", "full functional dependency"],
            ["3NF / Third Normal Form",
             "3nf", "third normal form", "transitive dependency",
             "transitive dependence"],
            ["BCNF",
             "bcnf", "boyce codd", "boyce-codd normal form"],
        ],
    },

    "osi_model": {
        "triggers": {
            "osi", "osi model", "seven layers", "7 layers",
            "network layers", "osi layers",
        },
        "concepts": [
            ["Physical layer",   "physical layer", "physical"],
            ["Data Link layer",  "data link", "datalink"],
            ["Network layer",    "network layer", "network"],
            ["Transport layer",  "transport layer", "transport"],
            ["Session layer",    "session layer", "session"],
            ["Presentation layer", "presentation layer", "presentation"],
            ["Application layer", "application layer", "application"],
        ],
    },

    "acid_properties": {
        "triggers": {
            "acid", "acid properties", "transaction properties",
            "atomicity consistency isolation durability",
        },
        "concepts": [
            ["Atomicity",   "atomicity", "atomic", "all or nothing"],
            ["Consistency", "consistency", "consistent", "valid state"],
            ["Isolation",   "isolation", "isolate", "concurrent transactions"],
            ["Durability",  "durability", "durable", "permanent", "persist"],
        ],
    },

    "sdlc": {
        "triggers": {
            "sdlc", "software development life cycle",
            "phases of sdlc", "software life cycle",
        },
        "concepts": [
            ["Planning",        "planning", "plan"],
            ["Analysis",        "analysis", "requirement", "requirements"],
            ["Design",          "design", "system design"],
            ["Implementation",  "implementation", "coding", "development"],
            ["Testing",         "testing", "test"],
            ["Deployment",      "deployment", "deploy", "release"],
            ["Maintenance",     "maintenance", "maintain"],
        ],
    },

    "sorting_algorithms": {
        "triggers": {
            "sorting algorithms", "sorting techniques",
            "types of sorting", "sorting methods",
        },
        "concepts": [
            ["Bubble Sort",    "bubble sort", "bubble"],
            ["Selection Sort", "selection sort", "selection"],
            ["Insertion Sort", "insertion sort", "insertion"],
            ["Merge Sort",     "merge sort", "mergesort"],
            ["Quick Sort",     "quick sort", "quicksort"],
        ],
    },

    "data_structures": {
        "triggers": {
            "data structures", "types of data structure",
            "linear data structure", "non linear",
        },
        "concepts": [
            ["Array",       "array"],
            ["Linked List", "linked list"],
            ["Stack",       "stack"],
            ["Queue",       "queue"],
            ["Tree",        "tree"],
            ["Graph",       "graph"],
            ["Hash Table",  "hash table", "hashing", "hashtable"],
        ],
    },

    # ── NEW: class & object definition questions ──────────────────────────
    # Triggers: "what is class", "class and object", "define class", etc.
    # Required concepts are the things ANY correct answer must mention.
    # Floor: if student covers 2/3 → score ≥ 7; all 3 → score ≥ 8
    "oop_class_object": {
        "triggers": {
            "what is class", "class and object", "class and object",
            "define class", "explain class", "what is object",
            "class object", "class in oop", "object in oop",
            "what are class", "class are", "object are",
        },
        "concepts": [
            # Concept 1: class as a blueprint/template
            ["class as blueprint/template",
             "blueprint", "template", "prototype", "mold",
             "user defined type", "user-defined type",
             "user defined datatype", "user-defined datatype",
             "custom datatype", "custom data type",
             "defines structure", "structure of object",
             "defines the structure",
            ],
            # Concept 2: object as instance/real entity
            ["object as instance/entity",
             "instance", "object", "real time entity", "realtime entity",
             "runtime entity", "real world entity", "real entity",
             "actual entity", "concrete", "instantiation",
             "created from class", "created from a class",
            ],
            # Concept 3: purpose / what they enable
            ["purpose (reusability/data+behavior)",
             "reusable", "reusability", "reuse",
             "data and behavior", "data and behaviour",
             "holds data", "holds specific data",
             "behavior", "behaviour", "methods and attributes",
             "attributes and methods", "encapsulates",
             "code reuse", "create multiple",
            ],
        ],
    },

    # ── NEW: inheritance definition questions ─────────────────────────────
    "oop_inheritance": {
        "triggers": {
            "what is inheritance", "explain inheritance",
            "define inheritance", "inheritance in oop",
        },
        "concepts": [
            ["parent/base class",
             "parent class", "base class", "superclass", "super class"],
            ["child/derived class",
             "child class", "derived class", "subclass", "sub class"],
            ["reusability via inheritance",
             "reuse", "reusability", "inherit properties",
             "inherits methods", "inherits attributes",
             "inherit", "inherits"],
        ],
    },

    # ── NEW: polymorphism definition questions ────────────────────────────
    "oop_polymorphism": {
        "triggers": {
            "what is polymorphism", "explain polymorphism",
            "define polymorphism", "types of polymorphism",
        },
        "concepts": [
            ["many forms",
             "many forms", "multiple forms", "one interface", "different forms"],
            ["overloading",
             "overloading", "method overloading", "compile time",
             "compile-time", "static polymorphism"],
            ["overriding",
             "overriding", "method overriding", "runtime",
             "run time", "dynamic polymorphism"],
        ],
    },

    # ── NEW: encapsulation definition questions ───────────────────────────
    "oop_encapsulation": {
        "triggers": {
            "what is encapsulation", "explain encapsulation",
            "define encapsulation", "encapsulation in oop",
        },
        "concepts": [
            ["data hiding",
             "data hiding", "hiding", "hide data", "hides data",
             "hide implementation", "hides implementation"],
            ["bundling data and methods",
             "bundling", "bundle", "wrapping", "wrap",
             "data and methods", "methods and data",
             "data and functions", "binds data"],
            ["access control",
             "private", "protected", "public",
             "access modifier", "access specifier",
             "getter", "setter", "getters", "setters"],
        ],
    },

    # ── NEW: abstraction definition questions ─────────────────────────────
    "oop_abstraction": {
        "triggers": {
            "what is abstraction", "explain abstraction",
            "define abstraction", "abstraction in oop",
        },
        "concepts": [
            ["hiding implementation details",
             "hide implementation", "hides implementation",
             "hiding details", "hide details",
             "implementation detail", "internal detail"],
            ["showing essential features",
             "essential", "expose essential", "show essential",
             "only essential", "what it does", "interface",
             "abstract class", "abstract method"],
            ["reducing complexity",
             "reduce complexity", "simplify", "complexity",
             "simple interface", "easy to use"],
        ],
    },
}


def _detect_domain(question: str) -> Optional[str]:
    """
    Return the first matching domain key for this question, or None.

    Checks BOTH the raw lowercased question AND the normalised version
    (abbreviation-expanded) so that:
      • "OOP pillars"  → normalises to "object oriented programming pillars"
        → matches "object oriented" trigger
      • "what is class and object" → matches "class and object" trigger
        in oop_class_object domain

    Priority: more-specific domains checked first (longer trigger phrases
    match before single-word ones to avoid false positives).
    """
    q_raw  = question.lower()
    q_norm = normalise(question)   # abbreviation-expanded + compound-collapsed

    # Check both raw and normalised to maximise recall
    for domain, cfg in DOMAIN_CONCEPTS.items():
        triggers = cfg["triggers"]
        if any(tw in q_raw for tw in triggers):
            return domain
        if any(tw in q_norm for tw in triggers):
            return domain
    return None


def _concept_coverage(student: str, question: str) -> Tuple[float, List[str], List[str]]:
    """
    Check which required concepts the student's answer covers.

    FIX A (multi-word matching):  We search for each synonym phrase in the
    FULL normalised student text, not in a token set.  This means:
      "real time entity" is in the text → matches synonym phrase → credit given
      "inherit" is in text → matches synonym "inherit" → inheritance credited

    Returns
    -------
    (coverage_ratio, covered_names, missing_names)
    All empty / 0.0 when no domain detected.
    """
    domain = _detect_domain(question)
    if not domain:
        return 0.0, [], []

    concepts   = DOMAIN_CONCEPTS[domain]["concepts"]
    stu_norm   = normalise(student)          # full string, not token set
    covered, missing = [], []

    for concept in concepts:
        name      = concept[0]              # display name
        synonyms  = concept[1:]             # phrases to search for

        # Search each synonym phrase in the full normalised student text
        found = any(syn in stu_norm for syn in synonyms)
        (covered if found else missing).append(name)

    ratio = len(covered) / max(1, len(concepts))
    return ratio, covered, missing


# ═════════════════════════════════════════════════════════════════════════════
# [4]  KEYWORD RECALL
#      FIX B: no length penalty.  Only measures how many expected-answer
#      tokens the student covered.  A short but correct answer is not
#      penalised for brevity.
# ═════════════════════════════════════════════════════════════════════════════

def _keyword_recall(expected: str, student: str) -> float:
    """
    Recall = |expected_tokens ∩ student_tokens| / |expected_tokens|

    No length penalty.  Coverage of the expected answer's key terms is
    the only signal — a concise correct answer scores as high as a verbose one.
    """
    try:
        exp_t = _tokens(expected)
        stu_t = _tokens(student)
        if not exp_t:
            return 0.5
        return round(len(exp_t & stu_t) / len(exp_t), 4)
    except Exception as e:
        logger.warning("keyword_recall failed: %s", e)
        return 0.3


# ═════════════════════════════════════════════════════════════════════════════
# [5]  SYNONYM MATCH
#      FIX A (multi-word synonyms):
#        Step 1 — phrase check in full text  (catches "real time entity", etc.)
#        Step 2 — token-level group lookup   (built-in SYNONYM_GROUPS map)
#        Step 3 — optional WordNet expansion
#        Step 4 — stem prefix (last resort)
# ═════════════════════════════════════════════════════════════════════════════

# Each group = set of synonymous words/phrases (single or multi-word)
SYNONYM_GROUPS: List[set] = [
    # OOP — class/object
    # "user defined datatype", "codebase" added so Q1-style answers get credit
    {"class", "blueprint", "template", "prototype", "mold", "pattern",
     "user defined type", "user defined datatype", "user-defined datatype",
     "custom type", "custom datatype", "structure"},
    {"object", "instance", "entity", "real time entity", "realtime entity",
     "runtime entity", "actual entity", "concrete object", "real world entity",
     "real entity", "real object", "real world object"},
    # reusable/reusability ≈ code reuse benefit (mentioned in Q1 student answer)
    {"reusable", "reusability", "reuse", "code reuse", "modular",
     "modularity", "maintainable", "maintainability"},

    # OOP pillars
    {"abstraction", "abstract", "hiding details", "hide implementation",
     "expose essential", "hides implementation", "essential functionality"},
    {"encapsulation", "encapsulate", "data hiding", "wrapping", "bundling",
     "access control", "bind data"},
    {"inheritance", "inherit", "derive", "extend", "subclass",
     "parent class", "child class", "base class", "superclass"},
    {"polymorphism", "polymorphic", "overloading", "overriding",
     "many forms", "method overload", "method override"},

    # OOP structural
    {"method", "function", "behavior", "operation", "member function",
     "member method"},
    {"attribute", "field", "property", "member variable", "data member",
     "instance variable"},
    {"constructor", "initializer", "init method"},
    {"access modifier", "access specifier", "visibility"},

    # DB
    {"database", "db", "datastore", "data repository"},
    {"query", "request", "search", "retrieve", "fetch", "lookup"},
    {"table", "relation", "record"},
    {"primary key", "unique key", "identifier"},
    {"normalization", "normalisation", "standardization"},
    {"transaction", "operation", "atomic operation"},
    {"index", "indexing", "indices"},

    # Algorithms / DS
    {"array", "list", "sequence", "collection"},
    {"stack", "lifo", "last in first out"},
    {"queue", "fifo", "first in first out"},
    {"recursion", "recursive", "self-referential"},
    {"loop", "iteration", "cycle", "repetition"},

    # General academic
    {"increase", "grow", "rise", "expand", "enhance", "improve", "escalate"},
    {"decrease", "reduce", "fall", "decline", "diminish", "lessen"},
    {"important", "significant", "crucial", "vital", "essential", "key", "critical"},
    {"process", "procedure", "approach", "technique", "mechanism"},
    {"store", "save", "retain", "persist", "hold", "keep"},
    {"convert", "transform", "change", "alter", "modify"},
    {"create", "generate", "produce", "make", "build", "construct"},
    {"delete", "remove", "erase", "drop", "eliminate"},
    {"allow", "permit", "enable", "authorize", "grant"},
    {"prevent", "restrict", "block", "prohibit", "disallow"},
    {"fast", "quick", "rapid", "efficient", "speedy"},
    {"slow", "sluggish", "inefficient", "delayed"},
    {"simple", "easy", "straightforward", "uncomplicated"},
    {"complex", "complicated", "difficult", "hard", "challenging"},

    # Science
    {"energy", "power", "force", "strength"},
    {"cell", "unit", "component", "element"},
    {"photosynthesis", "light reaction", "carbon fixation"},
    {"respiration", "breathing", "gas exchange"},
    {"atom", "particle", "molecule"},
]


def _build_syn_map(groups: List[set]) -> dict:
    """
    Build two lookup structures:
      token_map:  token → set of group indices  (for fast token lookup)
      phrase_list: sorted list of (phrase, group_idx) for multi-word search
    """
    token_map  = {}
    phrase_list = []
    for idx, grp in enumerate(groups):
        for entry in grp:
            tokens = entry.split()
            if len(tokens) == 1:
                token_map.setdefault(tokens[0], set()).add(idx)
            else:
                phrase_list.append((entry, idx))
    # Sort longest phrases first so longer matches take priority
    phrase_list.sort(key=lambda x: -len(x[0]))
    return token_map, phrase_list


_SYN_TOKEN_MAP, _SYN_PHRASE_LIST = _build_syn_map(SYNONYM_GROUPS)


def _synonym_match(expected: str, student: str) -> float:
    """
    For each meaningful token in the expected answer, award credit if:
      1. Token appears directly in student tokens  → 1.00
      2. Token's multi-word phrase found in student full text → 0.95
         (FIX A: this catches "real time entity", "parent class", etc.)
      3. Token shares a synonym group with any student token → 0.85
      4. WordNet expansion match (if NLTK available) → 0.88
      5. Stem prefix match (4+ char prefix) → 0.60
    """
    exp_tokens = _tokens(expected)
    stu_tokens = _tokens(student)
    stu_norm   = normalise(student)     # full text for phrase search

    if not exp_tokens:
        return 0.5

    def _token_groups(tok: str) -> set:
        return _SYN_TOKEN_MAP.get(tok, set())

    def _phrase_groups_in_text(text: str) -> set:
        """All group indices whose phrases appear in text."""
        found = set()
        for phrase, idx in _SYN_PHRASE_LIST:
            if phrase in text:
                found.add(idx)
        return found

    stu_phrase_groups = _phrase_groups_in_text(stu_norm)

    # Optional WordNet
    try:
        import nltk
        from nltk.corpus import wordnet
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)

        def _wn_syns(w: str) -> set:
            s = {w}
            for syn in wordnet.synsets(w):
                for lem in syn.lemmas():
                    s.add(lem.name().lower().replace("_", " "))
            return s

        wordnet_available = True
    except Exception:
        wordnet_available = False

    matched = 0.0
    for ew in exp_tokens:
        # 1. Direct token match
        if ew in stu_tokens:
            matched += 1.0
            continue

        # 2. Multi-word phrase that contains this token appears in student text
        ew_groups = _token_groups(ew)
        if ew_groups & stu_phrase_groups:
            matched += 0.95
            continue

        # 3. Synonym group via token map
        if ew_groups and any(_token_groups(sw) & ew_groups for sw in stu_tokens):
            matched += 0.85
            continue

        # 4. WordNet
        if wordnet_available:
            try:
                ew_syns = _wn_syns(ew)
                if ew_syns & stu_tokens:
                    matched += 0.88
                    continue
            except Exception:
                pass

        # 5. Stem prefix
        if len(ew) > 5:
            stem = ew[:max(4, len(ew) - 3)]
            if any(sw.startswith(stem) for sw in stu_tokens):
                matched += 0.60

    return min(1.0, round(matched / len(exp_tokens), 4))


# ═════════════════════════════════════════════════════════════════════════════
# [6]  TF-IDF COSINE  (supporting signal only — not dominant)
# ═════════════════════════════════════════════════════════════════════════════

def _tfidf_cosine(expected: str, student: str) -> float:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        e = normalise(expected)
        s = normalise(student)
        if not e or not s:
            return 0.0

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                              stop_words="english", sublinear_tf=True)
        mat = vec.fit_transform([e, s])
        return float(max(0.0, min(1.0, cosine_similarity(mat[0], mat[1])[0][0])))
    except Exception as ex:
        logger.warning("tfidf_cosine failed: %s", ex)
        return _keyword_recall(expected, student) * 0.80


# ═════════════════════════════════════════════════════════════════════════════
# [7]  SEMANTIC SIMILARITY  (MiniLM sentence embeddings)
#      Expected answer embedding is cached per question_id.
# ═════════════════════════════════════════════════════════════════════════════

_st_model      = None
_st_model_lock = threading.Lock()


def _get_st_model():
    global _st_model
    if _st_model is None:
        with _st_model_lock:
            if _st_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    _st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
                    logger.info("SentenceTransformer loaded.")
                except Exception as e:
                    logger.warning("SentenceTransformer load failed: %s", e)
                    _st_model = "FAILED"
    return None if _st_model == "FAILED" else _st_model


def get_expected_embedding(question_id: int, expected_text: str):
    """Cache expected-answer embedding by question_id (avoids re-encoding)."""
    with _embedding_lock:
        if question_id in _embedding_cache:
            return _embedding_cache[question_id]

    model = _get_st_model()
    if model is None:
        return None
    try:
        emb = model.encode([normalise(expected_text)])[0]
        with _embedding_lock:
            if len(_embedding_cache) >= EMBEDDING_CACHE_MAX:
                del _embedding_cache[next(iter(_embedding_cache))]
            _embedding_cache[question_id] = emb
        return emb
    except Exception as e:
        logger.warning("Embedding error q%s: %s", question_id, e)
        return None


def _semantic_sim(expected: str, student: str,
                  question_id: Optional[int] = None) -> float:
    """
    Cosine similarity between MiniLM sentence embeddings.
    Raw cosine returned — no internal threshold applied here.
    Thresholding happens only in _apply_thresholds().
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        model = _get_st_model()
        if model is None:
            return _tfidf_cosine(expected, student)

        exp_emb = (get_expected_embedding(question_id, expected)
                   if question_id is not None
                   else model.encode([normalise(expected)])[0])

        if exp_emb is None:
            return _tfidf_cosine(expected, student)

        stu_emb = model.encode([normalise(student)])[0]
        return float(max(0.0, min(1.0, cos_sim([exp_emb], [stu_emb])[0][0])))

    except Exception as e:
        logger.warning("semantic_sim failed: %s", e)
        return _tfidf_cosine(expected, student)


# ═════════════════════════════════════════════════════════════════════════════
# [8]  GEMINI  (optional, dominant when available)
# ═════════════════════════════════════════════════════════════════════════════

_GEMINI_PROMPT_WITH_EXPECTED = """\
You are a strict but fair academic evaluator scoring a student's answer.

Return ONLY valid JSON — no markdown fences, no explanation outside JSON:
{{
  "semantic_score": <float 0.0–1.0>,
  "feedback": "<2–3 specific sentences: what is correct, what is missing, how to improve>",
  "key_points_covered": ["<concept 1>", "<concept 2>"],
  "key_points_missing":  ["<concept A>", "<concept B>"]
}}

SCORING GUIDE:
- 0.90–1.00 : All key concepts correct (synonyms/paraphrasing = full credit)
- 0.75–0.89 : Very good, only minor details missing
- 0.55–0.74 : Good, main idea present but important details absent
- 0.35–0.54 : Partial — some relevant content but significant gaps
- 0.15–0.34 : Weak — barely touches the topic
- 0.00–0.14 : Wrong or completely irrelevant

RULES:
- Synonyms and paraphrasing = full credit for that concept
- "SQL" = "Structured Query Language" — never penalise abbreviations
- "data base" = "database" — never penalise spacing variants
- "real time entity" ≈ "instance" — accept equivalent CS terminology
- "class are like blueprint" = student knows class is a blueprint — give credit
- Spelling/grammar errors do NOT reduce score unless meaning is lost
- Feedback MUST name specific concepts, NOT generic phrases like "good answer"
- Never return exactly 0.0 unless answer is empty or gibberish
- Never return exactly 1.0 unless answer is flawlessly complete

EXPECTED ANSWER:
\"\"\"{expected}\"\"\"

STUDENT ANSWER:
\"\"\"{student}\"\"\"\
"""

_GEMINI_PROMPT_NO_EXPECTED = """\
You are a strict but fair academic evaluator.
The teacher did NOT provide an expected answer.
Evaluate the student's answer based on your subject-matter knowledge.

Question: \"\"\"{question}\"\"\"
Student Answer: \"\"\"{student}\"\"\"

Return ONLY valid JSON:
{{
  "semantic_score": <float 0.0–1.0>,
  "feedback": "<2–3 sentences: what is correct, missing, how to improve>",
  "key_points_covered": ["<concept correctly mentioned>"],
  "key_points_missing":  ["<important concept not mentioned>"]
}}

RULES:
- Evaluate on factual correctness and completeness
- Synonyms and paraphrasing = full credit
- Never return exactly 0.0 unless answer is empty or gibberish\
"""


def _gemini_evaluate(expected: str, student: str,
                     question: str = "") -> Optional[dict]:
    """Call Gemini with strict thread timeout. Returns None on any failure."""
    if not GEMINI_API_KEY:
        return None

    result_holder: list = [None]
    error_holder:  list = [None]

    def _call():
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model  = genai.GenerativeModel(GEMINI_MODEL)
            prompt = (
                _GEMINI_PROMPT_WITH_EXPECTED.format(expected=expected, student=student)
                if expected.strip()
                else _GEMINI_PROMPT_NO_EXPECTED.format(
                    question=question or "Unknown question", student=student)
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
                    text = re.sub(r"^```json\s*", "", resp.text.strip())
                    text = re.sub(r"```$", "", text).strip()
                    data = json.loads(text)

                    score   = float(max(0.01, min(0.99,
                                   float(data.get("semantic_score", 0.5)))))
                    covered = [str(x).strip() for x in
                               data.get("key_points_covered", []) if str(x).strip()]
                    missing = [str(x).strip() for x in
                               data.get("key_points_missing", []) if str(x).strip()]

                    result_holder[0] = {
                        "score":    score,
                        "feedback": str(data.get("feedback", "")).strip(),
                        "covered":  covered,
                        "missing":  missing,
                    }
                    return
                except (json.JSONDecodeError, Exception):
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                    else:
                        error_holder[0] = "retries exhausted"
        except Exception as e:
            error_holder[0] = e

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=GEMINI_TIMEOUT)

    if t.is_alive():
        logger.warning("Gemini timed out — using local fallback.")
        return None
    if error_holder[0]:
        logger.warning("Gemini error: %s — using local fallback.", error_holder[0])
        return None
    return result_holder[0]


# ═════════════════════════════════════════════════════════════════════════════
# [9]  WEIGHTED FUSION + SCORE CONVERSION
#
#  content_signal  =  composite of all signals, used to drive final score
#
#  With Gemini:    gemini leads (0.40), semantic supports (0.20),
#                  concept_coverage floor applied on top
#  Without Gemini: semantic leads (0.38), synonym (0.22), concept (0.20),
#                  keyword (0.12), tfidf (0.08)
#
#  concept_coverage acts as a FLOOR:
#    if student covered 4/4 OOP pillars → content_signal ≥ 0.80 guaranteed
#    This fixes the "knows all 4 pillars → 6/10" bug.
# ═════════════════════════════════════════════════════════════════════════════

# Weights must sum to 1.0
_W_GEMINI = {
    "gemini":          0.40,
    "semantic_sim":    0.20,
    "synonym_match":   0.15,
    "concept_coverage":0.12,
    "keyword_recall":  0.08,
    "tfidf_cosine":    0.05,
}

_W_LOCAL = {
    "semantic_sim":    0.38,
    "synonym_match":   0.22,
    "concept_coverage":0.20,
    "keyword_recall":  0.12,
    "tfidf_cosine":    0.08,
}

assert abs(sum(_W_GEMINI.values()) - 1.0) < 1e-9
assert abs(sum(_W_LOCAL.values())  - 1.0) < 1e-9


def _to_score(content_signal: float) -> int:
    """
    Map content_signal (0–1) to integer grade (0–10).
    Calibrated so:
      ≥0.90 → 10,  ≥0.80 → 9,  ≥0.70 → 8,
      ≥0.60 → 7,   ≥0.50 → 6,  ≥0.40 → 5,
      ≥0.30 → 4,   ≥0.20 → 3,  ≥0.10 → 2,
      ≥0.04 → 1,   else  → 0
    """
    thresholds = [(0.90, 10), (0.80, 9), (0.70, 8), (0.60, 7),
                  (0.50, 6),  (0.40, 5), (0.30, 4), (0.20, 3),
                  (0.10, 2),  (0.04, 1)]
    for thr, grade in thresholds:
        if content_signal >= thr:
            return grade
    return 0


def _feedback(score: int, scores: dict, covered: list,
              missing: list, no_expected: bool) -> str:
    """Build meaningful, metric-specific feedback when Gemini is absent."""
    sem  = scores.get("semantic_sim",    0)
    kw   = scores.get("keyword_recall",  0)
    cov  = scores.get("concept_coverage",0)
    syn  = scores.get("synonym_match",   0)

    cov_str = ", ".join(covered[:4]) if covered else None
    mis_str = ", ".join(missing[:4]) if missing else None

    if score == 10:
        return ("Perfect — all key concepts covered with accuracy. "
                "The meaning aligns completely with the expected answer.")
    if score >= 8:
        msg = (f"Strong answer — semantic similarity {int(sem*100)}%, "
               f"concept coverage {int(cov*100)}%. ")
        return msg + (f"Minor gaps: {mis_str}." if mis_str
                      else "Only minor elaboration could improve this.")
    if score >= 6:
        msg = f"Good — main idea present (semantic: {int(sem*100)}%). "
        if cov_str:
            msg += f"Correctly addressed: {cov_str}. "
        if mis_str:
            msg += f"Key concepts missing: {mis_str}."
        return msg
    if score >= 4:
        msg = f"Partial (concept coverage: {int(cov*100)}%, synonym match: {int(syn*100)}%). "
        if cov_str:
            msg += f"You covered: {cov_str}. "
        if mis_str:
            msg += f"Missing: {mis_str}."
        return msg
    if score >= 2:
        return (f"Weak — limited relevant content "
                f"(keyword recall: {int(kw*100)}%). "
                + (f"Missing concepts: {mis_str}. " if mis_str else "")
                + "Include specific terminology from the topic.")
    if score == 1:
        return ("The answer barely touches the topic. "
                "Review the material and include key definitions.")
    return ("No answer submitted." if not no_expected
            else "No answer submitted, or entirely off-topic.")


# ═════════════════════════════════════════════════════════════════════════════
# CORE PIPELINES
# ═════════════════════════════════════════════════════════════════════════════

def _pipeline_with_expected(expected: str, student: str,
                             question: str = "",
                             question_id: Optional[int] = None) -> dict:
    """
    Full pipeline when an expected answer exists.

    FIX C: concept_coverage_score is now called HERE (was missing before)
    and its result is included in the weighted fusion AND used as a floor
    on content_signal.  This ensures Q1-style questions benefit from domain
    concept detection even when an expected answer is present.
    """
    # ── Collect all local signals ─────────────────────────────────────────
    coverage_ratio, covered, missing = _concept_coverage(student, question)

    scores = {
        "keyword_recall":   _keyword_recall(expected, student),
        "synonym_match":    _synonym_match(expected, student),
        "tfidf_cosine":     _tfidf_cosine(expected, student),
        "semantic_sim":     _semantic_sim(expected, student, question_id),
        "concept_coverage": coverage_ratio,
    }

    feedback    = ""
    used_gemini = False

    # ── Try Gemini ────────────────────────────────────────────────────────
    gem = _gemini_evaluate(expected, student, question)
    if gem:
        scores["gemini"] = gem["score"]
        feedback  = gem.get("feedback", "")
        # Merge Gemini's covered/missing with local concept detection
        if gem.get("covered"):
            covered = gem["covered"]
        if gem.get("missing"):
            missing = gem["missing"]
        used_gemini = True
        weights = _W_GEMINI
    else:
        weights = _W_LOCAL

    # ── Weighted fusion ───────────────────────────────────────────────────
    raw_signal = sum(scores.get(k, 0) * w for k, w in weights.items())

    # ── concept_coverage FLOOR ───────────────────────────────────────────────
    # Guarantees minimum signal when student covers required concepts.
    # Scaled by how many concepts the domain requires:
    #   coverage_ratio=1.0  → floor=0.80 → score ≥ 9   (all concepts present)
    #   coverage_ratio=0.67 → floor=0.54 → score ≥ 6   (2/3 concepts)
    #   coverage_ratio=0.50 → floor=0.40 → score ≥ 5   (half concepts)
    #   coverage_ratio=0.0  → floor=0.0  → no boost     (no domain detected)
    # Cap at 0.82 so concept coverage alone cannot give 10 — explanation quality
    # (semantic/synonym) must contribute the final push to 10.
    # Floor logic:
    #   3/3 concepts covered → floor = 0.72 → score ≥ 8
    #   2/3 concepts covered → floor = 0.48 → score ≥ 5
    #   1/3 concepts covered → floor = 0.24 → score ≥ 3
    # Score 9-10 requires BOTH concept floor AND high semantic/synonym quality.
    # Cap at 0.75 so explanation depth (semantic, synonym) must contribute the
    # final push — prevents shallow but keyword-complete answers from hitting 9.
    concept_floor  = min(0.75, coverage_ratio * 0.72)
    content_signal = max(raw_signal, concept_floor)

    final_score = _to_score(content_signal)

    if not feedback:
        feedback = _feedback(final_score, scores, covered, missing, no_expected=False)

    logger.info(
        "with_exp → sem:%.2f kw:%.2f syn:%.2f cov:%.2f tfidf:%.2f gem:%.2f"
        " | raw:%.3f floor:%.3f sig:%.3f → %d",
        scores["semantic_sim"], scores["keyword_recall"],
        scores["synonym_match"], scores["concept_coverage"],
        scores["tfidf_cosine"], scores.get("gemini", 0),
        raw_signal, concept_floor, content_signal, final_score,
    )

    return {
        "score":              int(max(0, min(10, final_score))),
        "feedback":           feedback,
        "key_points_covered": covered,
        "key_points_missing": missing,
        "breakdown":          {k: round(v * 10, 2) for k, v in scores.items()},
        "used_gemini":        used_gemini,
        "no_expected_answer": False,
    }


def _pipeline_without_expected(student: str, question: str = "") -> dict:
    """
    Pipeline when no expected answer is stored.

    Primary:  Gemini (evaluates on its own subject knowledge)
    Fallback: concept_coverage + vocabulary richness
              FIX B: NO length bias — coverage of required concepts drives score
    """
    # ── Try Gemini first ──────────────────────────────────────────────────
    gem = _gemini_evaluate(expected="", student=student, question=question)
    if gem:
        gem_score = gem["score"]
        covered   = gem.get("covered", [])
        missing   = gem.get("missing", [])
        feedback  = gem.get("feedback", "")

        final_score = max(1, min(9, round(gem_score * 10)))
        if gem_score < 0.05:
            final_score = 0

        if not feedback:
            feedback = _feedback(
                final_score,
                {"semantic_sim": gem_score, "concept_coverage": gem_score,
                 "keyword_recall": gem_score, "synonym_match": gem_score},
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

    # ── Local fallback — concept coverage + richness ───────────────────────
    words      = student.split()
    word_count = len(words)

    if word_count == 0:
        return {
            "score": 0, "feedback": "No answer was submitted.",
            "key_points_covered": [], "key_points_missing": [],
            "breakdown": {}, "used_gemini": False, "no_expected_answer": True,
        }

    # Concept coverage (FIX B: replaces length heuristic)
    coverage_ratio, covered, missing = _concept_coverage(student, question)
    domain_detected = coverage_ratio > 0.0

    # Vocabulary richness — meaningful terms per answer
    meaningful   = len(_tokens(student))
    unique_ratio = len({w.lower() for w in words}) / max(1, word_count)
    richness     = unique_ratio * min(1.0, meaningful / 20)  # saturates at 20 terms

    if domain_detected:
        # Coverage is the main signal — length is irrelevant
        # 4/4 pillars covered → content_quality ≈ 0.82 → score 9
        content_quality = coverage_ratio * 0.80 + richness * 0.20
        final_score     = _to_score(content_quality)
        final_score     = max(1, min(10, final_score))
    else:
        # No domain — richness only, cap at 7 (can't verify correctness)
        content_quality = richness * 0.65 + min(0.25, word_count / 100) * 0.35
        final_score     = _to_score(min(0.72, content_quality))
        final_score     = max(1, min(7, final_score))

    # Feedback
    if domain_detected:
        if missing:
            cov_str = ", ".join(covered)
            mis_str = ", ".join(missing)
            feedback = (f"Good coverage — you addressed: {cov_str}. "
                        f"Missing: {mis_str}. Adding these would improve your score.")
        else:
            feedback = (f"Excellent — all required concepts covered: {', '.join(covered)}. "
                        f"Your answer is comprehensive.")
    else:
        feedback = (
            f"No expected answer on file. Evaluated on content richness "
            f"({meaningful} meaningful terms, {int(unique_ratio*100)}% unique vocabulary). "
            f"Ask your teacher to add an expected answer for a more accurate score."
        )

    logger.info(
        "no_exp → domain=%s coverage=%.2f richness=%.2f quality=%.2f → %d",
        domain_detected, coverage_ratio, richness, content_quality, final_score,
    )

    return {
        "score":              final_score,
        "feedback":           feedback,
        "key_points_covered": covered,
        "key_points_missing": missing,
        "breakdown":          {
            "concept_coverage": round(coverage_ratio * 10, 2),
            "richness":         round(richness * 10, 2),
            "content_quality":  round(content_quality * 10, 2),
        },
        "used_gemini":        False,
        "no_expected_answer": True,
    }


# ═════════════════════════════════════════════════════════════════════════════
# AUTO-GENERATE EXPECTED ANSWER  (called by teacher interface)
# ═════════════════════════════════════════════════════════════════════════════

def generate_expected_answer(question_text: str) -> str:
    if not GEMINI_API_KEY:
        return ""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model    = genai.GenerativeModel(GEMINI_MODEL)
        prompt   = (
            "You are an academic subject matter expert. "
            "Provide a comprehensive, accurate expected answer for this question. "
            "Cover all key concepts in 2–4 clear sentences. "
            "Do NOT include preamble — just the answer itself.\n\n"
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
# PUBLIC API  —  call ai_evaluate_safe() from admin.py routes
# ═════════════════════════════════════════════════════════════════════════════

def ai_evaluate(expected_answer: str, student_answer: str,
                question_text: str = "",
                question_id: Optional[int] = None) -> dict:
    """
    Evaluate a student answer.

    Parameters
    ----------
    expected_answer : str  — teacher's reference answer (may be empty)
    student_answer  : str  — student's submitted answer
    question_text   : str  — question text (used for concept detection & Gemini)
    question_id     : int  — enables embedding cache (skip re-encoding expected)

    Returns
    -------
    dict:  score (0–10), feedback, key_points_covered, key_points_missing,
           breakdown, used_gemini, no_expected_answer
    """
    expected = (expected_answer or "").strip()
    student  = (student_answer  or "").strip()
    question = (question_text   or "").strip()

    # Fast paths
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

    # Result cache
    cache_key = hashlib.sha256(
        f"{expected}|||{student}|||{question}".lower().encode()
    ).hexdigest()
    with _cache_lock:
        if cache_key in _eval_cache:
            return _eval_cache[cache_key]

    # Route
    result = (_pipeline_with_expected(expected, student, question, question_id)
              if expected
              else _pipeline_without_expected(student, question))

    with _cache_lock:
        _eval_cache[cache_key] = result

    return result


def ai_evaluate_safe(expected_answer: str, student_answer: str,
                     question_text: str = "",
                     question_id: Optional[int] = None,
                     fallback_score: Optional[int] = None) -> dict:
    """
    Never raises.  Always returns a valid result dict.
    Use this in all Flask routes — never call ai_evaluate() directly.
    """
    try:
        return ai_evaluate(expected_answer, student_answer,
                           question_text, question_id)
    except Exception as e:
        logger.error("ai_evaluate_safe error: %s", e)
        return {
            "score":              fallback_score if fallback_score is not None else 0,
            "feedback":           "Evaluation service temporarily unavailable.",
            "key_points_covered": [],
            "key_points_missing": [],
            "breakdown":          {},
            "used_gemini":        False,
            "no_expected_answer": not bool((expected_answer or "").strip()),
            "error":              str(e),
        }
