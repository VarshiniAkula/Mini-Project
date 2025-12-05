"""
CustomData Loader for LongMemEval Dataset
Handles loading and processing of LongMemEval conversational data.
"""

import json
from typing import Dict, List, Optional


class CustomData:
    """Load and process LongMemEval dataset for retrieval experiments."""

    def __init__(self, path_locomo: Optional[str] = None, path_longmem_eval: Optional[str] = None):
        """
        Initialize CustomData loader.

        Args:
            path_locomo: Path to Locomo dataset (not used in this pipeline)
            path_longmem_eval: Path to LongMemEval cleaned JSON file
        """
        self.path_locomo = path_locomo
        self.path_longmem_eval = path_longmem_eval
        self.longmem_eval_data = []

        if path_longmem_eval:
            self._load_longmem_eval()

    def _load_longmem_eval(self):
        """Load LongMemEval dataset from JSON file."""
        print(f"Loading LongMemEval data from {self.path_longmem_eval}...")
        with open(self.path_longmem_eval, 'r', encoding='utf-8') as f:
            self.longmem_eval_data = json.load(f)
        print(f"Loaded {len(self.longmem_eval_data)} LongMemEval items")

    def process_longmem_eval(self) -> Dict[str, List[Dict]]:
        """
        Process LongMemEval data into collections organized by question_id.

        Returns:
            Dict mapping collection_key (vectordb_collection_{question_id}) to list of session entries.
            Each entry contains:
                - haystack_session: List of conversation messages
                - haystack_date: Session date
                - haystack_session_id: Session identifier
        """
        collections = {}

        for item in self.longmem_eval_data:
            question_id = item.get('question_id', '')
            collection_key = f"vectordb_collection_{question_id}"

            haystack_sessions = item.get('haystack_sessions', [])
            haystack_dates = item.get('haystack_dates', [])
            haystack_session_ids = item.get('haystack_session_ids', [])

            # Create entries for each session
            entries = []
            for idx, session in enumerate(haystack_sessions):
                date = haystack_dates[idx] if idx < len(haystack_dates) else ''
                session_id = haystack_session_ids[idx] if idx < len(haystack_session_ids) else ''

                entries.append({
                    'haystack_session': session,
                    'haystack_date': date,
                    'haystack_session_id': session_id
                })

            collections[collection_key] = entries

        print(f"Processed {len(collections)} collections from LongMemEval data")
        return collections

    def load_source_qa_longmem_eval(self) -> Dict[str, List[Dict]]:
        """
        Load Q&A pairs organized by question_id.

        Returns:
            Dict mapping question_id to list of Q&A dictionaries.
            Each Q&A dict contains:
                - question: The query text
                - answer: Ground truth answer
                - evidence_ids: List of relevant session IDs
        """
        qa_dict = {}

        for item in self.longmem_eval_data:
            question_id = item.get('question_id', '')

            qa_entry = {
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'evidence_ids': item.get('answer_session_ids', [])
            }

            # Organize by question_id
            if question_id not in qa_dict:
                qa_dict[question_id] = []
            qa_dict[question_id].append(qa_entry)

        total_questions = sum(len(questions) for questions in qa_dict.values())
        print(f"Loaded {total_questions} Q&A pairs across {len(qa_dict)} question_ids")
        return qa_dict
