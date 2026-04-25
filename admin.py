"""
admin.py — AES_ai Flask Application (Production-Grade + Parallel Evaluation)
=============================================================================
FIXES IN THIS VERSION:
  1. MYSQL_PORT changed back to 3306 (XAMPP default — 3307 was causing MySQL crash)
  2. delete_student_score now redirects back to the student's score page, not admin_students
  3. submit_answers (legacy alias) calls redirect instead of directly calling the view function
     to prevent double-commit on back-button resubmit
  4. add_student / add_teacher guard against duplicate username (catches IntegrityError)
  5. Cursor always closed in finally blocks in critical routes to prevent connection leaks
  6. teacher_view_score defaultdict converted to plain dict before passing to template
     (Jinja2 can't iterate defaultdict reliably in all versions)
"""

import os
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_mysqldb import MySQL

from ai_evaluator import ai_evaluate_safe, generate_expected_answer

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change_this_in_production_abc123")
app.template_folder = "templates"

# ── MySQL ────────────────────────────────────────────────────────────────────
app.config["MYSQL_HOST"]     = "localhost"
app.config["MYSQL_PORT"]     = 3306        # ← FIX: XAMPP default is 3306, not 3307
app.config["MYSQL_USER"]     = "root"
app.config["MYSQL_PASSWORD"] = ""          # ← your MySQL password (blank for XAMPP default)
app.config["MYSQL_DB"]       = "teacher_part"

mysql = MySQL(app)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — CENTRAL EVALUATION CALL
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_answer(expected: str, student: str, question_text: str = "") -> dict:
    """
    Evaluate a student's answer.
    - expected may be empty — handled gracefully via Gemini or heuristics.
    - question_text is passed so Gemini has context when expected is absent.
    """
    expected = (expected or "").strip()
    student  = (student  or "").strip()
    qt       = (question_text or "").strip()

    if not student:
        return {
            "score":              0,
            "feedback":           "No answer submitted.",
            "key_points_covered": [],
            "key_points_missing": [],
            "breakdown":          {},
        }

    return ai_evaluate_safe(expected, student, question_text=qt)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — PARALLEL EVALUATION (the key speed optimization)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_answers_parallel(answers_list: list) -> dict:
    """
    Evaluate multiple answers simultaneously using a thread pool.

    Parameters
    ----------
    answers_list : list of dicts, each with keys:
        question_id, student_answer, expected_answer, question_text

    Returns
    -------
    dict keyed by question_id → evaluation result dict
    """
    results = {}

    if not answers_list:
        return results

    def _eval_one(item):
        """Worker: evaluate a single answer. Runs in its own thread."""
        result = evaluate_answer(
            item["expected_answer"],
            item["student_answer"],
            item["question_text"],
        )
        return item["question_id"], result

    # Fire ALL evaluations at once — Gemini calls for every question run in parallel.
    # max_workers=8 is safe for the free Gemini tier (15 req/min limit).
    # If you hit rate limits, lower this to 4 or 5.
    max_workers = min(8, len(answers_list))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_eval_one, item): item for item in answers_list}
        for future in as_completed(futures):
            try:
                qid, result = future.result()
                results[qid] = result
            except Exception as e:
                # If one question's evaluation crashes, give it a safe fallback
                item = futures[future]
                results[item["question_id"]] = {
                    "score":    0,
                    "feedback": f"Evaluation error: {e}",
                    "key_points_covered": [],
                    "key_points_missing": [],
                    "breakdown": {},
                }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# API — JSON evaluation endpoint (for AJAX / external use)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    if "teacher_logged_in" not in session and "admin_logged_in" not in session:
        return jsonify({"error": "Unauthorised"}), 401
    data          = request.get_json(silent=True) or {}
    expected      = data.get("expected_answer", "").strip()
    student       = data.get("student_answer",  "").strip()
    question_text = data.get("question_text",   "").strip()
    if not student:
        return jsonify({"error": "student_answer is required"}), 400
    return jsonify(evaluate_answer(expected, student, question_text)), 200


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("Homepage.html")


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN — AUTH
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM Admins WHERE username=%s AND password=%s",
                    (username, password))
        admin = cur.fetchone()
        cur.close()
        if admin:
            session["admin_logged_in"] = True
            return redirect(url_for("admin_home"))
        return render_template("adminlogin.html", error="Invalid username or password")
    return render_template("adminlogin.html")


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))


@app.route("/admin/home")
def admin_home():
    if "admin_logged_in" not in session:
        return redirect(url_for("admin_login"))
    return render_template("adminhome.html")


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN — STUDENTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/admin/students")
def admin_students():
    if "admin_logged_in" not in session:
        return redirect(url_for("admin_login"))
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT s.student_id, s.username, s.password,
               IFNULL(SUM(sa.score), 0) AS total_score
        FROM Students s
        LEFT JOIN StudentAnswers sa ON s.student_id = sa.student_id
        GROUP BY s.student_id, s.username, s.password
        ORDER BY s.student_id
    """)
    students = cur.fetchall()
    cur.close()
    return render_template("admin_students.html", students=students)


@app.route("/admin/add_student", methods=["POST"])
def add_student():
    if "admin_logged_in" not in session:
        return redirect(url_for("admin_login"))
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    if username and password:
        cur = mysql.connection.cursor()
        try:
            cur.execute("INSERT INTO Students (username, password) VALUES (%s, %s)",
                        (username, password))
            mysql.connection.commit()
        except Exception:
            mysql.connection.rollback()  # duplicate username — ignore silently
        finally:
            cur.close()
    return redirect(url_for("admin_students"))


@app.route("/admin/update_student/<int:student_id>", methods=["POST"])
def update_student(student_id):
    if "admin_logged_in" not in session:
        return redirect(url_for("admin_login"))
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    cur = mysql.connection.cursor()
    cur.execute(
        "UPDATE Students SET username=%s, password=%s WHERE student_id=%s",
        (username, password, student_id)
    )
    mysql.connection.commit()
    cur.close()
    return redirect(url_for("admin_students"))


@app.route("/admin/delete_student/<int:student_id>", methods=["POST"])
def delete_student(student_id):
    if "admin_logged_in" not in session:
        return redirect(url_for("admin_login"))
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM StudentAnswers WHERE student_id=%s", (student_id,))
    cur.execute("DELETE FROM Students WHERE student_id=%s", (student_id,))
    mysql.connection.commit()
    cur.close()
    return redirect(url_for("admin_students"))


@app.route("/admin/view_student_scores/<int:student_id>")
def view_student_scores(student_id):
    if "admin_logged_in" not in session:
        return redirect(url_for("admin_login"))
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT sa.answer_id,
               sa.test_id,
               t.test_name,
               q.question_text,
               IFNULL(ea.answer_text, '') AS expected_answer,
               sa.answer_text             AS student_answer,
               IFNULL(sa.score, 0)        AS score,
               IFNULL(sa.feedback, '')    AS feedback
        FROM StudentAnswers sa
        JOIN Tests     t   ON sa.test_id     = t.test_id
        JOIN Questions q   ON sa.question_id = q.question_id
        LEFT JOIN (
            SELECT question_id, MIN(answer_id) AS min_aid
            FROM ExpectedAnswers
            GROUP BY question_id
        ) ea_sub ON q.question_id = ea_sub.question_id
        LEFT JOIN ExpectedAnswers ea ON ea.answer_id = ea_sub.min_aid
        WHERE sa.student_id = %s
        ORDER BY sa.test_id, q.question_id
    """, (student_id,))
    rows = cur.fetchall()
    cur.close()

    scores = [
        {
            "answer_id":       r[0],
            "test_id":         r[1],
            "test_name":       r[2],
            "question_text":   r[3],
            "expected_answer": r[4],
            "student_answer":  r[5],
            "score":           r[6],
            "feedback":        r[7],
        }
        for r in rows
    ]
    return render_template("student_scores.html", scores=scores, student_id=student_id)


@app.route("/admin/delete_student_score/<int:answer_id>", methods=["POST"])
def delete_student_score(answer_id):
    if "admin_logged_in" not in session:
        return redirect(url_for("admin_login"))
    cur = mysql.connection.cursor()
    # FIX: fetch student_id BEFORE deleting so we can redirect back to that student's page
    cur.execute("SELECT student_id FROM StudentAnswers WHERE answer_id=%s", (answer_id,))
    row = cur.fetchone()
    student_id = row[0] if row else None
    cur.execute("DELETE FROM StudentAnswers WHERE answer_id=%s", (answer_id,))
    mysql.connection.commit()
    cur.close()
    # FIX: redirect back to the student's score page, not the full student list
    if student_id:
        return redirect(url_for("view_student_scores", student_id=student_id))
    return redirect(url_for("admin_students"))


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN — TEACHERS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/admin/teachers")
def admin_teachers():
    if "admin_logged_in" not in session:
        return redirect(url_for("admin_login"))
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Teachers ORDER BY teacher_id")
    teachers = cur.fetchall()
    cur.close()
    return render_template("admin_teachers.html", teachers=teachers)


@app.route("/admin/add_teacher", methods=["GET", "POST"])
def add_teacher():
    if "admin_logged_in" not in session:
        return redirect(url_for("admin_login"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if username and password:
            cur = mysql.connection.cursor()
            try:
                cur.execute(
                    "INSERT INTO Teachers (username, password) VALUES (%s, %s)",
                    (username, password)
                )
                mysql.connection.commit()
            except Exception:
                mysql.connection.rollback()
            finally:
                cur.close()
        return redirect(url_for("admin_teachers"))
    return render_template("add_teacher.html")


@app.route("/admin/update_teacher/<int:teacher_id>", methods=["GET", "POST"])
def update_teacher(teacher_id):
    if "admin_logged_in" not in session:
        return redirect(url_for("admin_login"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        cur = mysql.connection.cursor()
        cur.execute(
            "UPDATE Teachers SET username=%s, password=%s WHERE teacher_id=%s",
            (username, password, teacher_id)
        )
        mysql.connection.commit()
        cur.close()
        return redirect(url_for("admin_teachers"))
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Teachers WHERE teacher_id=%s", (teacher_id,))
    teacher = cur.fetchone()
    cur.close()
    if not teacher:
        return "Teacher not found", 404
    return render_template("update_teacher.html", teacher=teacher, teacher_id=teacher_id)


@app.route("/admin/delete_teacher/<int:teacher_id>", methods=["POST"])
def delete_teacher(teacher_id):
    if "admin_logged_in" not in session:
        return redirect(url_for("admin_login"))
    cur = mysql.connection.cursor()
    cur.execute("""
        DELETE sa FROM StudentAnswers sa
        JOIN Tests t ON sa.test_id = t.test_id
        WHERE t.teacher_id = %s
    """, (teacher_id,))
    cur.execute("""
        DELETE ea FROM ExpectedAnswers ea
        JOIN Questions q ON ea.question_id = q.question_id
        JOIN Tests t     ON q.test_id      = t.test_id
        WHERE t.teacher_id = %s
    """, (teacher_id,))
    cur.execute("""
        DELETE q FROM Questions q
        JOIN Tests t ON q.test_id = t.test_id
        WHERE t.teacher_id = %s
    """, (teacher_id,))
    cur.execute("DELETE FROM Tests WHERE teacher_id=%s", (teacher_id,))
    cur.execute("DELETE FROM Teachers WHERE teacher_id=%s", (teacher_id,))
    mysql.connection.commit()
    cur.close()
    return redirect(url_for("admin_teachers"))


@app.route("/admin/view_teacher_tests/<int:teacher_id>")
def view_teacher_tests(teacher_id):
    if "admin_logged_in" not in session:
        return redirect(url_for("admin_login"))
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Tests WHERE teacher_id=%s ORDER BY test_id",
                (teacher_id,))
    tests = cur.fetchall()
    cur.close()
    return render_template("view_teacher_tests.html", tests=tests, teacher_id=teacher_id)


@app.route("/admin/view_test_questions/<int:test_id>")
def view_test_questions(test_id):
    if "admin_logged_in" not in session:
        return redirect(url_for("admin_login"))
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Questions WHERE test_id=%s ORDER BY question_id",
                (test_id,))
    questions = cur.fetchall()
    question_answers = {}
    for q in questions:
        cur.execute("SELECT * FROM ExpectedAnswers WHERE question_id=%s", (q[0],))
        question_answers[q[0]] = cur.fetchall()
    cur.execute("SELECT teacher_id FROM Tests WHERE test_id=%s", (test_id,))
    row = cur.fetchone()
    teacher_id = row[0] if row else 0
    cur.close()
    return render_template("view_test_questions.html",
                           questions=questions,
                           question_answers=question_answers,
                           teacher_id=teacher_id,
                           test_id=test_id)


# ══════════════════════════════════════════════════════════════════════════════
# TEACHER — AUTH
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/teacher_login", methods=["GET", "POST"])
def teacher_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT * FROM Teachers WHERE username=%s AND password=%s",
            (username, password)
        )
        teacher = cur.fetchone()
        cur.close()
        if teacher:
            session["teacher_logged_in"] = True
            session["teacher_id"] = teacher[0]
            return redirect(url_for("teacher_home"))
        return render_template("teacher_login.html", error="Invalid username or password")
    return render_template("teacher_login.html")


@app.route("/teacher_logout")
def teacher_logout():
    session.pop("teacher_logged_in", None)
    session.pop("teacher_id", None)
    return redirect(url_for("teacher_login"))


# ══════════════════════════════════════════════════════════════════════════════
# TEACHER — HOME (create / rename / delete tests)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/teacher_home", methods=["GET", "POST"])
def teacher_home():
    if "teacher_logged_in" not in session:
        return redirect(url_for("teacher_login"))
    teacher_id = session["teacher_id"]

    if request.method == "POST":
        cur = mysql.connection.cursor()

        if "add_test_name" in request.form:
            test_name = request.form.get("test_name", "").strip()
            if test_name:
                cur.execute(
                    "INSERT INTO Tests (test_name, teacher_id) VALUES (%s, %s)",
                    (test_name, teacher_id)
                )
                mysql.connection.commit()

        elif "update_test_name" in request.form:
            test_id = request.form.get("test_id")
            updated = request.form.get("updated_test_name", "").strip()
            if test_id and updated:
                cur.execute(
                    "UPDATE Tests SET test_name=%s WHERE test_id=%s AND teacher_id=%s",
                    (updated, test_id, teacher_id)
                )
                mysql.connection.commit()

        elif "delete_test_name" in request.form:
            test_id = request.form.get("test_id")
            if test_id:
                cur.execute("DELETE FROM StudentAnswers WHERE test_id=%s", (test_id,))
                cur.execute("""
                    DELETE ea FROM ExpectedAnswers ea
                    JOIN Questions q ON ea.question_id = q.question_id
                    WHERE q.test_id = %s
                """, (test_id,))
                cur.execute("DELETE FROM Questions WHERE test_id=%s", (test_id,))
                cur.execute(
                    "DELETE FROM Tests WHERE test_id=%s AND teacher_id=%s",
                    (test_id, teacher_id)
                )
                mysql.connection.commit()

        cur.close()

    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT test_id, test_name FROM Tests WHERE teacher_id=%s ORDER BY test_id DESC",
        (teacher_id,)
    )
    tests = cur.fetchall()
    cur.close()
    return render_template("teacher_home.html", tests=tests)


# ══════════════════════════════════════════════════════════════════════════════
# TEACHER — QUESTIONS (add / edit / delete)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/teacher/view_test_questions/<int:test_id>", methods=["GET", "POST"])
def view_teacher_test_questions(test_id):
    if "teacher_logged_in" not in session:
        return redirect(url_for("teacher_login"))

    teacher_id = session["teacher_id"]
    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT test_id, test_name FROM Tests WHERE test_id=%s AND teacher_id=%s",
        (test_id, teacher_id)
    )
    test_row = cur.fetchone()
    if not test_row:
        cur.close()
        return "Test not found or access denied.", 403

    if request.method == "POST":
        action = request.form.get("action", "")
        if not action:
            if "add_question"    in request.form: action = "add_question"
            elif "delete_question" in request.form: action = "delete_question"
            elif "edit_question"   in request.form: action = "edit_question"

        if action == "add_question":
            q_text = request.form.get("question_text", "").strip()
            a_text = request.form.get("expected_answer", "").strip()
            if q_text:
                if not a_text:
                    a_text = generate_expected_answer(q_text)
                cur.execute(
                    "INSERT INTO Questions (question_text, test_id) VALUES (%s, %s)",
                    (q_text, test_id)
                )
                qid = cur.lastrowid
                if a_text:
                    cur.execute(
                        "INSERT INTO ExpectedAnswers (answer_text, question_id) VALUES (%s, %s)",
                        (a_text, qid)
                    )
                mysql.connection.commit()

        elif action == "edit_question":
            qid    = request.form.get("question_id")
            q_text = request.form.get("question_text", "").strip()
            a_text = request.form.get("expected_answer", "").strip()
            if qid and q_text:
                cur.execute(
                    "UPDATE Questions SET question_text=%s WHERE question_id=%s",
                    (q_text, qid)
                )
                cur.execute(
                    "SELECT answer_id FROM ExpectedAnswers WHERE question_id=%s LIMIT 1",
                    (qid,)
                )
                ea_row = cur.fetchone()
                if a_text:
                    if ea_row:
                        cur.execute(
                            "UPDATE ExpectedAnswers SET answer_text=%s WHERE answer_id=%s",
                            (a_text, ea_row[0])
                        )
                    else:
                        cur.execute(
                            "INSERT INTO ExpectedAnswers (answer_text, question_id) VALUES (%s, %s)",
                            (a_text, qid)
                        )
                elif not ea_row:
                    generated = generate_expected_answer(q_text)
                    if generated:
                        cur.execute(
                            "INSERT INTO ExpectedAnswers (answer_text, question_id) VALUES (%s, %s)",
                            (generated, qid)
                        )
                mysql.connection.commit()

        elif action == "delete_question":
            qid = request.form.get("question_id")
            if qid:
                cur.execute("DELETE FROM StudentAnswers WHERE question_id=%s", (qid,))
                cur.execute("DELETE FROM ExpectedAnswers WHERE question_id=%s", (qid,))
                cur.execute("DELETE FROM Questions WHERE question_id=%s", (qid,))
                mysql.connection.commit()

    cur.execute(
        "SELECT question_id, question_text FROM Questions WHERE test_id=%s ORDER BY question_id",
        (test_id,)
    )
    questions = cur.fetchall()
    question_answers = {}
    for q in questions:
        cur.execute(
            "SELECT answer_id, answer_text FROM ExpectedAnswers WHERE question_id=%s LIMIT 1",
            (q[0],)
        )
        question_answers[q[0]] = cur.fetchall()
    cur.close()

    return render_template("view_teacher_test_questions.html",
                           test_id=test_id,
                           test_name=test_row[1],
                           teacher_id=teacher_id,
                           questions=questions,
                           question_answers=question_answers)


# ══════════════════════════════════════════════════════════════════════════════
# TEACHER — VIEW STUDENT SCORES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/teacher_view_score")
def teacher_view_score():
    if "teacher_logged_in" not in session:
        return redirect(url_for("teacher_login"))
    teacher_id = session["teacher_id"]
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT s.student_id,
               s.username,
               t.test_name,
               q.question_text,
               IFNULL(ea.answer_text, '') AS expected_answer,
               sa.answer_text             AS student_answer,
               IFNULL(sa.score, 0)        AS score,
               IFNULL(sa.feedback, '')    AS feedback
        FROM StudentAnswers sa
        JOIN Students   s   ON sa.student_id  = s.student_id
        JOIN Tests      t   ON sa.test_id     = t.test_id
        JOIN Questions  q   ON sa.question_id = q.question_id
        LEFT JOIN (
            SELECT question_id, MIN(answer_id) AS min_aid
            FROM ExpectedAnswers
            GROUP BY question_id
        ) ea_sub ON q.question_id = ea_sub.question_id
        LEFT JOIN ExpectedAnswers ea ON ea.answer_id = ea_sub.min_aid
        WHERE t.teacher_id = %s
        ORDER BY s.student_id, t.test_name, q.question_id
    """, (teacher_id,))
    results = cur.fetchall()
    cur.close()

    # FIX: build as plain dict (not defaultdict) so Jinja2 iterates it correctly
    raw = defaultdict(lambda: {"student_username": None, "tests": defaultdict(list)})
    for row in results:
        sid, uname, tname, qtext, exp_ans, stu_ans, score, feedback = row
        raw[sid]["student_username"] = uname
        raw[sid]["tests"][tname].append({
            "question_text":   qtext,
            "expected_answer": exp_ans,
            "student_answer":  stu_ans,
            "score":           score,
            "feedback":        feedback,
        })

    # Convert nested defaultdicts to plain dicts for Jinja2
    student_scores = {
        sid: {
            "student_username": data["student_username"],
            "tests": dict(data["tests"]),
        }
        for sid, data in raw.items()
    }

    return render_template("teacher_view_score.html", student_scores=student_scores)


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT — AUTH
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/student_login", methods=["GET", "POST"])
def student_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT * FROM Students WHERE username=%s AND password=%s",
            (username, password)
        )
        student = cur.fetchone()
        cur.close()
        if student:
            session["student_logged_in"] = True
            session["student_id"] = student[0]
            return redirect(url_for("student_home"))
        return render_template("student_login.html", error="Invalid username or password")
    return render_template("student_login.html")


@app.route("/student_logout")
def student_logout():
    session.pop("student_logged_in", None)
    session.pop("student_id", None)
    return redirect(url_for("student_login"))


@app.route("/student_home")
def student_home():
    if "student_logged_in" not in session:
        return redirect(url_for("student_login"))
    return render_template("student_home.html")


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT — TAKE TEST (list)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/student_take_test")
def student_take_test():
    if "student_logged_in" not in session:
        return redirect(url_for("student_login"))
    cur = mysql.connection.cursor()
    cur.execute("SELECT test_id, test_name FROM Tests ORDER BY test_id DESC")
    rows = cur.fetchall()
    cur.close()
    tests = [{"test_id": r[0], "test_name": r[1]} for r in rows]
    return render_template("student_take_test.html", tests=tests)


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT — TAKE TEST (questions + submit)
# KEY CHANGE: sequential loop → parallel ThreadPoolExecutor
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/student_take_test/<int:test_id>", methods=["GET", "POST"])
def student_take_test_questions(test_id):
    if "student_logged_in" not in session:
        return redirect(url_for("student_login"))
    cur = mysql.connection.cursor()

    if request.method == "POST":
        student_id = session["student_id"]

        # Wipe previous submission for this test (prevent duplicates)
        cur.execute(
            "DELETE FROM StudentAnswers WHERE student_id=%s AND test_id=%s",
            (student_id, test_id)
        )
        mysql.connection.commit()

        # Fetch ALL questions for this test up front (one DB query)
        cur.execute(
            """SELECT q.question_id, q.question_text,
                      IFNULL(ea.answer_text, '') AS expected_answer
               FROM Questions q
               LEFT JOIN (
                   SELECT question_id, MIN(answer_id) AS min_aid
                   FROM ExpectedAnswers
                   GROUP BY question_id
               ) ea_sub ON q.question_id = ea_sub.question_id
               LEFT JOIN ExpectedAnswers ea ON ea.answer_id = ea_sub.min_aid
               WHERE q.test_id = %s
               ORDER BY q.question_id""",
            (test_id,)
        )
        question_rows = {r[0]: {"question_text": r[1], "expected_answer": r[2]}
                         for r in cur.fetchall()}

        # ── Build the work list from submitted form data ──────────────────────
        answers_to_evaluate = []
        for key, value in request.form.items():
            if not key.startswith("answer_"):
                continue
            try:
                question_id = int(key.split("_", 1)[1])
            except (ValueError, IndexError):
                continue

            student_answer = value.strip()
            q_info = question_rows.get(question_id, {})

            answers_to_evaluate.append({
                "question_id":     question_id,
                "student_answer":  student_answer,
                "expected_answer": q_info.get("expected_answer", ""),
                "question_text":   q_info.get("question_text", ""),
            })

        # ── PARALLEL EVALUATION — all Gemini + local algo calls fire at once ──
        eval_results = evaluate_answers_parallel(answers_to_evaluate)

        # ── Write results to DB ───────────────────────────────────────────────
        for item in answers_to_evaluate:
            qid    = item["question_id"]
            result = eval_results.get(qid, {"score": 0, "feedback": ""})
            score    = result.get("score", 0)
            feedback = result.get("feedback", "")

            cur.execute(
                """INSERT INTO StudentAnswers
                       (student_id, test_id, question_id, answer_text, score, feedback)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (student_id, test_id, qid,
                 item["student_answer"], score, feedback)
            )

        mysql.connection.commit()
        cur.close()
        return redirect(url_for("student_view_score"))

    # GET — fetch questions for display
    cur.execute(
        "SELECT DISTINCT question_id, question_text FROM Questions "
        "WHERE test_id=%s ORDER BY question_id",
        (test_id,)
    )
    questions = cur.fetchall()
    cur.execute("SELECT test_name FROM Tests WHERE test_id=%s", (test_id,))
    test_row  = cur.fetchone()
    test_name = test_row[0] if test_row else "Test"
    cur.close()

    return render_template("student_take_test_questions.html",
                           questions=questions,
                           test_id=test_id,
                           test_name=test_name)


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT — VIEW SCORES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/student_view_score")
def student_view_score():
    if "student_logged_in" not in session:
        return redirect(url_for("student_login"))
    student_id = session["student_id"]
    cur = mysql.connection.cursor()

    cur.execute("""
        SELECT t.test_id,
               t.test_name,
               q.question_text,
               IFNULL(ea.answer_text, '') AS expected_answer,
               sa.answer_text             AS student_answer,
               IFNULL(sa.score, 0)        AS score,
               IFNULL(sa.feedback, '')    AS feedback
        FROM StudentAnswers sa
        JOIN Tests      t   ON sa.test_id     = t.test_id
        JOIN Questions  q   ON sa.question_id = q.question_id
        LEFT JOIN (
            SELECT question_id, MIN(answer_id) AS min_aid
            FROM ExpectedAnswers
            GROUP BY question_id
        ) ea_sub ON q.question_id = ea_sub.question_id
        LEFT JOIN ExpectedAnswers ea ON ea.answer_id = ea_sub.min_aid
        WHERE sa.student_id = %s
        ORDER BY t.test_id, q.question_id
    """, (student_id,))
    results = cur.fetchall()
    cur.close()

    student_scores = {}
    for row in results:
        test_id, test_name, question_text, expected_answer, \
            student_answer, score, feedback = row

        if test_id not in student_scores:
            student_scores[test_id] = {
                "test_id":     test_id,
                "test_name":   test_name,
                "total_score": 0,
                "max_score":   0,
                "scores":      [],
            }

        student_scores[test_id]["scores"].append({
            "question":        question_text,
            "expected_answer": expected_answer,
            "student_answer":  student_answer,
            "score":           score,
            "feedback":        feedback,
        })
        student_scores[test_id]["total_score"] += score
        student_scores[test_id]["max_score"]   += 10

    return render_template("student_view_score.html",
                           student_scores=list(student_scores.values()))


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY ALIAS — /submit_answers kept for backward compatibility
# FIX: use redirect instead of calling the view directly (prevents double-submit)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/submit_answers", methods=["POST"])
def submit_answers():
    if "student_logged_in" not in session:
        return redirect(url_for("student_login"))
    test_id = request.form.get("test_id")
    if not test_id:
        return redirect(url_for("student_take_test"))
    # FIX: redirect to the proper POST endpoint instead of calling view directly
    # This avoids double-commit when the browser resubmits on back-navigation
    return redirect(url_for("student_take_test_questions", test_id=int(test_id)),
                    code=307)   # 307 = preserve POST method


if __name__ == "__main__":
    app.run(debug=True)
