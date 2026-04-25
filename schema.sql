-- =============================================================
-- AES_ai — Complete Database Schema
-- Run this ONCE to set up / repair the database.
-- =============================================================

CREATE DATABASE IF NOT EXISTS teacher_part
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE teacher_part;

-- ── Admins ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Admins (
    admin_id   INT          AUTO_INCREMENT PRIMARY KEY,
    username   VARCHAR(100) NOT NULL UNIQUE,
    password   VARCHAR(255) NOT NULL
) ENGINE=InnoDB;

-- Insert a default admin (change password before production)
INSERT IGNORE INTO Admins (username, password) VALUES ('admin', 'admin123');

-- ── Teachers ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Teachers (
    teacher_id INT          AUTO_INCREMENT PRIMARY KEY,
    username   VARCHAR(100) NOT NULL UNIQUE,
    password   VARCHAR(255) NOT NULL
) ENGINE=InnoDB;

-- ── Students ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Students (
    student_id INT          AUTO_INCREMENT PRIMARY KEY,
    username   VARCHAR(100) NOT NULL UNIQUE,
    password   VARCHAR(255) NOT NULL
) ENGINE=InnoDB;

-- ── Tests ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Tests (
    test_id    INT          AUTO_INCREMENT PRIMARY KEY,
    test_name  VARCHAR(255) NOT NULL,
    teacher_id INT          NOT NULL,
    created_at TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (teacher_id) REFERENCES Teachers(teacher_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ── Questions ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Questions (
    question_id   INT           AUTO_INCREMENT PRIMARY KEY,
    question_text TEXT          NOT NULL,
    test_id       INT           NOT NULL,
    FOREIGN KEY (test_id) REFERENCES Tests(test_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ── ExpectedAnswers ───────────────────────────────────────────
-- One row per question (one expected answer).
-- A question may have ZERO rows here — evaluated by AI on its own knowledge.
CREATE TABLE IF NOT EXISTS ExpectedAnswers (
    answer_id   INT  AUTO_INCREMENT PRIMARY KEY,
    answer_text TEXT NOT NULL,
    question_id INT  NOT NULL,
    FOREIGN KEY (question_id) REFERENCES Questions(question_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ── StudentAnswers ────────────────────────────────────────────
-- SINGLE SOURCE OF TRUTH for student submissions, scores, and feedback.
-- score    : 0–10  (set by ai_evaluator)
-- feedback : AI-generated explanation (stored here, not recomputed each view)
CREATE TABLE IF NOT EXISTS StudentAnswers (
    answer_id   INT          AUTO_INCREMENT PRIMARY KEY,
    student_id  INT          NOT NULL,
    test_id     INT          NOT NULL,
    question_id INT          NOT NULL,
    answer_text TEXT,
    score       INT          DEFAULT 0,
    feedback    TEXT         DEFAULT NULL,
    submitted_at TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id)  REFERENCES Students(student_id)  ON DELETE CASCADE,
    FOREIGN KEY (test_id)     REFERENCES Tests(test_id)         ON DELETE CASCADE,
    FOREIGN KEY (question_id) REFERENCES Questions(question_id) ON DELETE CASCADE,
    UNIQUE KEY uq_student_test_question (student_id, test_id, question_id)
) ENGINE=InnoDB;

-- =============================================================
-- Migration (run if StudentAnswers already exists without feedback column)
-- =============================================================

-- Add feedback column if missing
ALTER TABLE StudentAnswers
    ADD COLUMN IF NOT EXISTS feedback TEXT DEFAULT NULL;

-- Add question_id column if missing (legacy schema may lack it)
ALTER TABLE StudentAnswers
    ADD COLUMN IF NOT EXISTS question_id INT DEFAULT NULL;

-- Verify
-- DESCRIBE StudentAnswers;
-- DESCRIBE Questions;
-- DESCRIBE ExpectedAnswers;
