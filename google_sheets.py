"""Google Sheets integration for storing interview screening results.

Uses gcloud Application Default Credentials — run
  gcloud auth application-default login \\
    --scopes=https://www.googleapis.com/auth/spreadsheets,https://www.googleapis.com/auth/drive
once on your machine, then set GOOGLE_SHEETS_SPREADSHEET_ID in .env.local.
See README.md for the full setup.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional

import google.auth
import gspread

logger = logging.getLogger("google-sheets")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

HEADERS = [
    "Timestamp",
    "Session ID",
    "Candidate Name",
    "Score (1-10)",
    "Recommendation",
    "Strengths",
    "Areas for Improvement",
    "Summary",
    "Full Conversation",
]


class GoogleSheetsService:
    """Manages Google Sheets operations for interview results.

    Authenticates via gcloud Application Default Credentials. Synchronous
    gspread calls are run in a thread via asyncio.to_thread() to avoid
    blocking the event loop.
    """

    def __init__(self, spreadsheet_id: Optional[str] = None):
        self.spreadsheet_id = spreadsheet_id or os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", "")
        self.client: Optional[gspread.Client] = None
        self.worksheet: Optional[gspread.Worksheet] = None

    def initialize(self) -> None:
        """Initialize Google Sheets connection synchronously (call once at startup)."""
        if not self.spreadsheet_id:
            raise ValueError(
                "GOOGLE_SHEETS_SPREADSHEET_ID is not set. "
                "Add it to .env.local or pass spreadsheet_id= to GoogleSheetsService."
            )

        creds, _ = google.auth.default(scopes=SCOPES)
        self.client = gspread.authorize(creds)

        spreadsheet = self.client.open_by_key(self.spreadsheet_id)

        try:
            self.worksheet = spreadsheet.worksheet("Interview Results")
        except gspread.exceptions.WorksheetNotFound:
            self.worksheet = spreadsheet.add_worksheet(
                title="Interview Results", rows=1000, cols=len(HEADERS)
            )

        self._ensure_headers()
        logger.info("Google Sheets service initialized successfully")

    def _ensure_headers(self) -> None:
        existing = self.worksheet.row_values(1)
        if existing != HEADERS:
            col_letter = chr(ord("A") + len(HEADERS) - 1)
            self.worksheet.update(f"A1:{col_letter}1", [HEADERS])
            logger.info("Interview Results sheet headers initialized")

    def _sync_append_row(self, row: list) -> None:
        self.worksheet.append_row(row, value_input_option="USER_ENTERED")

    async def save_interview_result(self, session_id: str, data: dict) -> bool:
        """Append one row of interview evaluation data to the sheet."""
        if self.worksheet is None:
            logger.error("Google Sheets not initialized. Call initialize() first.")
            return False

        try:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            row = [
                timestamp,
                session_id,
                data.get("candidate_name", "Unknown"),
                data.get("score", ""),
                data.get("recommendation", ""),
                data.get("strengths", ""),
                data.get("areas_for_improvement", ""),
                data.get("summary", ""),
                data.get("conversation_transcript", ""),
            ]
            await asyncio.to_thread(self._sync_append_row, row)
            logger.info(
                f"Interview result saved for candidate '{data.get('candidate_name')}' "
                f"(session: {session_id})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save interview result to Google Sheets: {e}")
            return False


sheets_service = GoogleSheetsService()
