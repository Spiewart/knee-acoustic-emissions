-- Migration: Add biomechanics_import_id column to audio_processing table
-- Date: 2026-01-30
-- Description: Adds foreign key linking audio processing records to biomechanics import records

-- Add the column (allowing NULL for existing records)
ALTER TABLE audio_processing
ADD COLUMN IF NOT EXISTS biomechanics_import_id INTEGER;

-- Add foreign key constraint
ALTER TABLE audio_processing
DROP CONSTRAINT IF EXISTS fk_audio_biomechanics;

ALTER TABLE audio_processing
ADD CONSTRAINT fk_audio_biomechanics
FOREIGN KEY (biomechanics_import_id)
REFERENCES biomechanics_import(id)
ON DELETE SET NULL;

-- Verify the column was added
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'audio_processing'
AND column_name = 'biomechanics_import_id';
