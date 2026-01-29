#!/bin/bash
cd /Users/spiewart/acoustic_emissions_processing
.venv/bin/python -m pytest tests/test_database.py -v --tb=short
