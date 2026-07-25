import os
import tempfile
import unittest
from unittest.mock import patch

from llm_file_assistant import _run_with_rules


class ResumeRoleQueryTests(unittest.TestCase):
    def test_role_query_returns_matching_resume_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_analyst_resume = os.path.join(temp_dir, "alice_data_analyst.txt")
            with open(data_analyst_resume, "w", encoding="utf-8") as file_handle:
                file_handle.write("Alice Smith\nData Analyst\nSQL and Python\n")

            manager_resume = os.path.join(temp_dir, "bob_manager.txt")
            with open(manager_resume, "w", encoding="utf-8") as file_handle:
                file_handle.write("Bob Jones\nProject Manager\n")

            with patch("llm_file_assistant._default_resumes_directory", return_value=temp_dir):
                result = _run_with_rules("give me resumes for data analyst")

            self.assertIn("alice_data_analyst.txt", result)
            self.assertNotIn("bob_manager.txt", result)

    def test_role_question_uses_resume_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            resume_path = os.path.join(temp_dir, "sam_patel_resume.txt")
            with open(resume_path, "w", encoding="utf-8") as file_handle:
                file_handle.write("Sam Patel\nData Analyst\nSQL, Python, Tableau\n")

            with patch("llm_file_assistant._default_resumes_directory", return_value=temp_dir):
                result = _run_with_rules("what is role of sam patel")

            self.assertIn("Data Analyst", result)

    def test_read_query_resolves_resume_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            resume_path = os.path.join(temp_dir, "alex_kim_resume.txt")
            with open(resume_path, "w", encoding="utf-8") as file_handle:
                file_handle.write("Alex Kim\nData Analyst\nPython, SQL\n")

            with patch("llm_file_assistant._default_resumes_directory", return_value=temp_dir):
                result = _run_with_rules("read alex kim resume")

            self.assertIn("Alex Kim", result)
            self.assertIn("Data Analyst", result)


if __name__ == "__main__":
    unittest.main()
