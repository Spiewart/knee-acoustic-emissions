"""Reports module for generating Excel exports from database queries.

Provides on-demand Excel report generation for:
- Audio Processing sheets
- Biomechanics Import sheets
- Synchronization sheets
- Movement Cycle sheets
- Summary sheets

All data is queried directly from the database at report generation time,
ensuring accuracy and eliminating data duplication.
"""

from src.reports.report_generator import ReportGenerator

__all__ = ["ReportGenerator"]
