
-- ============================================================
-- Migration: Add feedback column to StudentAnswers
-- Run this ONCE against your teacher_part database
-- ============================================================

ALTER TABLE studentanswers
ADD COLUMN IF NOT EXISTS feedback TEXT DEFAULT NULL;

-- ============================================================
-- Verify
-- ============================================================
-- DESCRIBE studentanswers;
