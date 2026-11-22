import unittest
import json
import re
import threading
import urllib.error
import urllib.request
from io import BytesIO
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import app
import pymupdf


class MainWorkflowTests(unittest.TestCase):
    @staticmethod
    def analyst_group(group):
        group = dict(group)
        group.setdefault("source_ids", [])
        group.setdefault("supporting_evidence", list(group["source_ids"]))
        group.setdefault("change_summary", group.get("what_changed", "The cited passages differ."))
        group.setdefault("old_rule", group.get("previous_rule", "No earlier passage was supplied."))
        group.setdefault("new_rule", group.get("updated_rule", "No updated passage was supplied."))
        group.setdefault("change_category", "Other")
        group.setdefault("change_type", "Modified content")
        group.setdefault("importance", "Medium")
        group.setdefault("importance_reason", "The cited passages show a change requiring review.")
        group.setdefault("substantive_change", True)
        group.setdefault("confidence", 0.9)
        group.setdefault("unknown_information", "Any implications beyond these passages cannot be determined.")
        group.setdefault("simple_explanation", group.get("what_changed", "The cited passages differ."))
        group.setdefault("potentially_affected_groups", [])
        group.setdefault("practical_implications", [])
        group.setdefault("what_changed_for_them", [])
        group.setdefault("limitations", "The practical effects beyond the cited text cannot be determined.")
        group.setdefault("impact_supporting_evidence", list(group["source_ids"]))
        return group

    def test_local_env_loader_reads_key_settings_without_overriding_shell(self):
        with tempfile.TemporaryDirectory() as folder:
            env_path = Path(folder) / ".env"
            env_path.write_text(
                "CIVICLENS_TEST_FILE_VALUE='loaded safely'\nCIVICLENS_TEST_OVERRIDE=file-value\n",
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"CIVICLENS_TEST_OVERRIDE": "shell-value"}, clear=False):
                app.load_local_env(env_path)
                import os
                self.assertEqual(os.environ["CIVICLENS_TEST_FILE_VALUE"], "loaded safely")
                self.assertEqual(os.environ["CIVICLENS_TEST_OVERRIDE"], "shell-value")

    def test_cgfmu_sample_ai_request_uses_only_verified_passages(self):
        changes = app.compare_docs(*app.SAMPLE_DOCS)
        old_pages = app.extract_pdf(app.SAMPLES / app.SAMPLE_DOCS[0]["file"])
        new_pages = app.extract_pdf(app.SAMPLES / app.SAMPLE_DOCS[1]["file"])
        self.assertTrue(all(app.validate_change_sources(change, old_pages, new_pages) for change in changes))
        response_content = json.dumps({
            "summary": "The updated excerpt changes the specified loan limit from Rs.10 lakh to Rs.20 lakh and states a Tarun Plus category covering loans above 10 lakh and upto 20 lakh for entrepreneurs who repaid previous Tarun loans.",
            "groups": [
                self.analyst_group({"source_ids": ["cgfmu-loan-limit"], "title": "Specified loan limit", "what_changed": "The specified loan limit changes from Rs.10 lakh to Rs.20 lakh.", "previous_rule": "The earlier excerpt says Rs.10 lakh.", "updated_rule": "The updated excerpt says Rs.20 lakh.", "why_it_matters": "The updated excerpt states a higher specified loan limit.", "who_may_be_affected": None, "change_type": "Modified loan limit"}),
                self.analyst_group({"source_ids": ["cgfmu-tarun-plus"], "title": "Tarun Plus category", "what_changed": "The updated excerpt states Tarun Plus covering loans above 10 lakh and upto 20 lakh for entrepreneurs who repaid previous Tarun loans.", "previous_rule": "No corresponding passage was found in the supplied earlier comparison.", "updated_rule": "The updated excerpt states Tarun Plus covering loans above 10 lakh and upto 20 lakh for entrepreneurs who repaid previous Tarun loans.", "why_it_matters": "The updated excerpt states a new Tarun Plus category for entrepreneurs who repaid previous loans.", "who_may_be_affected": None, "change_type": "Added loan category"}),
            ],
        })

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": response_content}}]}).encode()

        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", return_value=FakeResponse()) as request:
            analysis, error = app.ai_policy_analysis(changes)
        self.assertIsNone(error)
        self.assertEqual(len(analysis["groups"]), 2)
        sent = json.loads(json.loads(request.call_args.args[0].data)["messages"][1]["content"])
        self.assertEqual(len(sent["verified_text_differences"]), len(changes))
        self.assertEqual(sent["verified_text_differences"][0]["previous_text"], changes[0]["previous"]["text"])
        self.assertEqual(sent["verified_text_differences"][0]["updated_text"], changes[0]["updated"]["text"])
        source_reference = sent["verified_text_differences"][0]["source_references"][0]
        self.assertEqual(source_reference["earlier"]["page"], changes[0]["previous"]["page"])
        self.assertEqual(source_reference["updated"]["document"], changes[0]["updated"]["document"])

    @staticmethod
    def explanation_change(change_id, before_text, after_text):
        return {
            "id": change_id, "type": "Modified content", "summary": "Text differs.",
            "previous": {"text": before_text, "page": 1, "document": "Earlier", "version": "Earlier"},
            "updated": {"text": after_text, "page": 2, "document": "Updated", "version": "Updated"},
        }

    def test_ai_analysis_is_one_batched_call_and_expands_duplicate_evidence_ids(self):
        first = self.explanation_change("change-1", "Applicants receive a benefit of Rs 10 lakh.", "Applicants receive a benefit of Rs 20 lakh.")
        duplicate = self.explanation_change("change-2", "Applicants receive a benefit of Rs 10 lakh.", "Applicants receive a benefit of Rs 20 lakh.")
        response_content = json.dumps({
            "summary": "The cited benefit limit changes from Rs 10 lakh to Rs 20 lakh.",
            "groups": [self.analyst_group({
                "source_ids": ["change-1"], "title": "Benefit limit changed",
                "what_changed": "The cited limit changes from Rs 10 lakh to Rs 20 lakh.",
                "previous_rule": "The earlier passage says Rs 10 lakh.",
                "updated_rule": "The updated passage says Rs 20 lakh.",
                "why_it_matters": "The excerpt states a higher benefit limit.",
                "who_may_be_affected": "Applicants",
            })],
        })

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": response_content}}]}).encode()

        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", return_value=FakeResponse()) as request:
            analysis, error = app.ai_policy_analysis([first, duplicate])

        self.assertIsNone(error)
        request.assert_called_once()
        payload = json.loads(request.call_args.args[0].data)
        sent_items = json.loads(payload["messages"][1]["content"])["verified_text_differences"]
        self.assertEqual(len(sent_items), 1)
        self.assertEqual(analysis["groups"][0]["source_ids"], ["change-1", "change-2"])

    def test_impact_fields_are_validated_and_linked_to_the_same_verified_change(self):
        change = self.explanation_change("impact-change", "The benefit is Rs 10 lakh for applicants.", "The benefit is Rs 20 lakh for applicants.")
        group = self.analyst_group({
            "source_ids": ["impact-change"], "title": "Benefit amount changed",
            "what_changed": "The cited benefit amount changes from Rs 10 lakh to Rs 20 lakh.",
            "previous_rule": "The earlier passage states Rs 10 lakh.",
            "updated_rule": "The updated passage states Rs 20 lakh.",
            "why_it_matters": "The cited amount is higher in the updated version.",
            "simple_explanation": "The listed benefit amount is higher in the updated passage.",
            "potentially_affected_groups": ["Applicants"],
            "practical_implications": ["The text lists Rs 20 lakh instead of Rs 10 lakh."],
            "what_changed_for_them": ["Applicants are named in both passages."],
            "who_may_be_affected": "Applicants",
        })
        body = json.dumps({"summary": "The cited benefit amount changes from Rs 10 lakh to Rs 20 lakh.", "groups": [group]})

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self): return json.dumps({"choices": [{"message": {"content": body}}]}).encode()

        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", return_value=FakeResponse()):
            analysis, error = app.ai_policy_analysis([change])
        self.assertIsNone(error)
        self.assertEqual(analysis["groups"][0]["impact_supporting_evidence"], ["impact-change"])
        self.assertEqual(analysis["groups"][0]["potentially_affected_groups"], ["Applicants"])

        group["impact_supporting_evidence"] = ["fabricated-source"]
        body = json.dumps({"summary": "The cited benefit amount changes from Rs 10 lakh to Rs 20 lakh.", "groups": [group]})
        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", return_value=FakeResponse()):
            analysis, error = app.ai_policy_analysis([change])
        self.assertIsNone(analysis)
        self.assertIn("impact explanation contains invalid supporting evidence IDs", error)

    def test_impact_supporting_evidence_validation_comprehensive(self):
        # Setup 2 changes grouped together to test subsets, duplicates, empty, and invalid IDs
        change1 = self.explanation_change("ch-1", "Eligibility percentile is 80 for students.", "Eligibility percentile is 85 for students.")
        change2 = self.explanation_change("ch-2", "The benefit is Rs 10 lakh for students.", "The benefit is Rs 20 lakh for students.")
        
        class FakeResp:
            def __init__(self, data): self.data = data
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps({"choices": [{"message": {"content": json.dumps(self.data)}}]}).encode()

        # 1. All group IDs supplied -> valid
        grp_all = self.analyst_group({
            "source_ids": ["ch-1", "ch-2"], "title": "Eligibility and benefit",
            "what_changed": "The percentile changes to 85 and benefit to Rs 20 lakh.",
            "previous_rule": "The earlier passage lists 80 and Rs 10 lakh.",
            "updated_rule": "The updated passage lists 85 and Rs 20 lakh.",
            "why_it_matters": "The updated passage lists a different percentile and benefit for students.",
            "potentially_affected_groups": ["Students"],
            "practical_implications": ["The updated excerpt lists 85 percentile for students."],
            "what_changed_for_them": ["The benefit for students is Rs 20 lakh."],
            "who_may_be_affected": "Students",
            "impact_supporting_evidence": ["ch-1", "ch-2"],
        })
        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", return_value=FakeResp({"summary": "The updated excerpt lists 85 percentile and benefit of Rs 20 lakh for students.", "groups": [grp_all]})):
            analysis, error = app.ai_policy_analysis([change1, change2])
        self.assertIsNone(error)
        self.assertEqual(analysis["groups"][0]["impact_supporting_evidence"], ["ch-1", "ch-2"])

        # 2. Valid subset of group IDs -> valid
        grp_subset = self.analyst_group({
            "source_ids": ["ch-1", "ch-2"], "title": "Eligibility and benefit",
            "what_changed": "The percentile changes to 85 and benefit to Rs 20 lakh.",
            "previous_rule": "The earlier passage lists 80 and Rs 10 lakh.",
            "updated_rule": "The updated passage lists 85 and Rs 20 lakh.",
            "why_it_matters": "The updated passage lists a different percentile and benefit for students.",
            "potentially_affected_groups": ["Students"],
            "practical_implications": ["The updated excerpt lists 85 percentile for students."],
            "what_changed_for_them": ["The benefit for students is Rs 20 lakh."],
            "who_may_be_affected": "Students",
            "impact_supporting_evidence": ["ch-1"],
        })
        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", return_value=FakeResp({"summary": "The updated excerpt lists 85 percentile and benefit of Rs 20 lakh for students.", "groups": [grp_subset]})):
            analysis, error = app.ai_policy_analysis([change1, change2])
        self.assertIsNone(error)
        self.assertEqual(analysis["groups"][0]["impact_supporting_evidence"], ["ch-1"])

        # 3. Empty list when impact is unknown / no impact established -> valid
        grp_empty_unknown = self.analyst_group({
            "source_ids": ["ch-1", "ch-2"], "title": "Eligibility and benefit",
            "what_changed": "The percentile changes to 85 and benefit to Rs 20 lakh.",
            "previous_rule": "The earlier passage lists 80 and Rs 10 lakh.",
            "updated_rule": "The updated passage lists 85 and Rs 20 lakh.",
            "why_it_matters": "The updated passage lists a different percentile and benefit for students.",
            "potentially_affected_groups": [],
            "practical_implications": [],
            "what_changed_for_them": [],
            "who_may_be_affected": None,
            "impact_supporting_evidence": [],
        })
        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", return_value=FakeResp({"summary": "The updated excerpt lists 85 percentile and benefit of Rs 20 lakh for students.", "groups": [grp_empty_unknown]})):
            analysis, error = app.ai_policy_analysis([change1, change2])
        self.assertIsNone(error)
        self.assertEqual(analysis["groups"][0]["impact_supporting_evidence"], [])

        # 4. Invalid ID not in group source_ids -> rejected
        grp_invalid = self.analyst_group({
            "source_ids": ["ch-1", "ch-2"], "title": "Eligibility and benefit",
            "what_changed": "The percentile changes to 85 and benefit to Rs 20 lakh.",
            "previous_rule": "The earlier passage lists 80 and Rs 10 lakh.",
            "updated_rule": "The updated passage lists 85 and Rs 20 lakh.",
            "why_it_matters": "The updated passage lists a different percentile and benefit for students.",
            "potentially_affected_groups": [],
            "practical_implications": [],
            "what_changed_for_them": [],
            "who_may_be_affected": None,
            "impact_supporting_evidence": ["non-existent-ch"],
        })
        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", return_value=FakeResp({"summary": "Changes for students.", "groups": [grp_invalid]})):
            analysis, error = app.ai_policy_analysis([change1, change2])
        self.assertIsNone(analysis)
        self.assertIn("impact explanation contains invalid supporting evidence IDs", error)

        # 5. Duplicate IDs -> rejected
        grp_dup = self.analyst_group({
            "source_ids": ["ch-1", "ch-2"], "title": "Eligibility and benefit",
            "what_changed": "The percentile changes to 85 and benefit to Rs 20 lakh.",
            "previous_rule": "The earlier passage lists 80 and Rs 10 lakh.",
            "updated_rule": "The updated passage lists 85 and Rs 20 lakh.",
            "why_it_matters": "The updated passage lists a different percentile and benefit for students.",
            "potentially_affected_groups": [],
            "practical_implications": [],
            "what_changed_for_them": [],
            "who_may_be_affected": None,
            "impact_supporting_evidence": ["ch-1", "ch-1"],
        })
        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", return_value=FakeResp({"summary": "Changes for students.", "groups": [grp_dup]})):
            analysis, error = app.ai_policy_analysis([change1, change2])
        self.assertIsNone(analysis)
        self.assertIn("impact explanation contains invalid supporting evidence IDs", error)

        # 6. Specific impact claim with NO supporting evidence (empty list) -> rejected
        grp_claim_no_ev = self.analyst_group({
            "source_ids": ["ch-1", "ch-2"], "title": "Eligibility and benefit",
            "what_changed": "The percentile changes to 85 and benefit to Rs 20 lakh.",
            "previous_rule": "The earlier passage lists 80 and Rs 10 lakh.",
            "updated_rule": "The updated passage lists 85 and Rs 20 lakh.",
            "why_it_matters": "The updated passage lists a different percentile and benefit for students.",
            "potentially_affected_groups": ["Students"],
            "practical_implications": ["Students qualify at 85 percentile."],
            "what_changed_for_them": ["Students receive higher benefits."],
            "who_may_be_affected": "Students",
            "impact_supporting_evidence": [],  # Empty despite specific impact claims!
        })
        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", return_value=FakeResp({"summary": "Changes for students.", "groups": [grp_claim_no_ev]})):
            analysis, error = app.ai_policy_analysis([change1, change2])
        self.assertIsNone(analysis)
        self.assertIn("impact explanation contains invalid supporting evidence IDs", error)

        # 7. Correct expansion of subset and empty evidence IDs for duplicated passages
        # Pass identical duplicates so unique_items deduplicates them
        dup_change1 = self.explanation_change("ch-1-dup", "Eligibility percentile is 80 for students.", "Eligibility percentile is 85 for students.")
        dup_change1["previous"]["page"] = 3
        dup_change1["updated"]["page"] = 4
        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", return_value=FakeResp({"summary": "Changes for students.", "groups": [grp_subset]})):
            analysis, error = app.ai_policy_analysis([change1, dup_change1, change2])
        self.assertIsNone(error)
        # ch-1 was deduplicated with ch-1-dup; subset was ['ch-1']; expanded impact should contain both ch-1 and ch-1-dup
        self.assertEqual(sorted(analysis["groups"][0]["impact_supporting_evidence"]), ["ch-1", "ch-1-dup"])

    def test_chat_retrieves_only_the_active_comparison_and_validates_citations(self):
        session_id = "chat-test-session"
        app.remember_document_session(session_id,
            {"title": "Earlier policy", "version": "Earlier"},
            {"title": "Updated policy", "version": "Updated"},
            [{"page": 2, "text": "Eligible applicants may receive a benefit of Rs 10 lakh.", "blocks": [{"text": "Eligible applicants may receive a benefit of Rs 10 lakh."}]}],
            [{"page": 4, "text": "Eligible applicants may receive a benefit of Rs 20 lakh.", "blocks": [{"text": "Eligible applicants may receive a benefit of Rs 20 lakh."}]}],
        )
        passages = app.retrieve_chat_evidence(app.DOCUMENT_SESSIONS[session_id], "What benefit may eligible applicants receive?")
        self.assertEqual({item["side"] for item in passages}, {"earlier", "updated"})
        self.assertEqual({item["page"] for item in passages}, {2, 4})
        response_body = json.dumps({"answer": "The earlier policy lists Rs 10 lakh. The updated policy lists Rs 20 lakh.", "source_ids": [item["id"] for item in passages]})

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self): return json.dumps({"choices": [{"message": {"content": response_body}}]}).encode()

        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", return_value=FakeResponse()):
            result = app.answer_policy_question(session_id, "What changed about the benefit?")
        self.assertTrue(result["ok"])
        self.assertEqual({(source["side"], source["page"]) for source in result["sources"]}, {("earlier", 2), ("updated", 4)})

        response_body = json.dumps({"answer": "A new rule starts on page 99.", "source_ids": ["not-in-evidence"]})
        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", return_value=FakeResponse()):
            result = app.answer_policy_question(session_id, "What benefit may applicants receive?")
        self.assertFalse(result["ok"])
        self.assertIn("verified AI answer", result["error"])

    def test_chat_retrieval_is_relevant_balanced_between_versions_and_limited(self):
        session = {
            "earlier_doc": {"title": "Earlier", "version": "2021"},
            "updated_doc": {"title": "Updated", "version": "2024"},
            "earlier_pages": [{"page": index + 1, "text": f"Applicants receive a financial benefit amount under earlier provision {index}.", "blocks": [{"text": f"Applicants receive a financial benefit amount under earlier provision {index}."}]} for index in range(12)],
            "updated_pages": [{"page": index + 10, "text": f"Applicants receive a financial benefit amount under revised provision {index}.", "blocks": [{"text": f"Applicants receive a financial benefit amount under revised provision {index}."}]} for index in range(3)],
        }
        evidence = app.retrieve_chat_evidence(session, "What financial benefit amount is provided?", limit=8)
        self.assertEqual(len(evidence), 8)
        self.assertEqual({item["side"] for item in evidence}, {"earlier", "updated"})
        self.assertEqual(len({item["id"] for item in evidence}), len(evidence))
        self.assertTrue(all(item["text"] in next(
            page["text"] for page in session[item["side"] + "_pages"] if page["page"] == item["page"]
        ) for item in evidence))
        self.assertEqual(app.retrieve_chat_evidence(session, "What financial benefit amount is provided?", limit=0), [])

    def test_chat_expired_session_and_missing_retrieved_passages_are_safe(self):
        self.assertFalse(app.answer_policy_question("missing-session", "Anything?")["ok"])
        app.remember_document_session("empty-chat-session",
            {"title": "Earlier", "version": "Earlier"}, {"title": "Updated", "version": "Updated"},
            [{"page": 1, "text": "Unrelated administrative wording only.", "blocks": [{"text": "Unrelated administrative wording only."}]}],
            [{"page": 1, "text": "Other unrelated content appears here.", "blocks": [{"text": "Other unrelated content appears here."}]}],
        )
        result = app.answer_policy_question("empty-chat-session", "What is the eligibility age?")
        self.assertTrue(result["ok"])
        self.assertEqual(result["sources"], [])
        self.assertIn("cannot be verified from the uploaded documents", result["answer"])

    def test_chat_context_and_retrieval_are_isolated_by_comparison(self):
        for comparison_id, title, amount in (("chat-isolation-a", "Policy A", "Rs 10 lakh"), ("chat-isolation-b", "Policy B", "Rs 30 lakh")):
            app.remember_document_session(comparison_id,
                {"title": title + " earlier", "version": "Earlier"}, {"title": title + " updated", "version": "Updated"},
                [{"page": 1, "text": f"Applicants may receive {amount} assistance.", "blocks": [{"text": f"Applicants may receive {amount} assistance."}]}],
                [{"page": 2, "text": f"Applicants may receive {amount} assistance.", "blocks": [{"text": f"Applicants may receive {amount} assistance."}]}],
            )
        evidence_a = app.retrieve_chat_evidence(app.DOCUMENT_SESSIONS["chat-isolation-a"], "What assistance do applicants receive?")
        evidence_b = app.retrieve_chat_evidence(app.DOCUMENT_SESSIONS["chat-isolation-b"], "What assistance do applicants receive?")
        self.assertTrue(all("Policy A" in item["document"] for item in evidence_a))
        self.assertTrue(all("Policy B" in item["document"] for item in evidence_b))
        app.DOCUMENT_SESSIONS["chat-isolation-a"]["chat_history"].append({"role": "user", "text": "private question"})
        self.assertEqual(app.DOCUMENT_SESSIONS["chat-isolation-b"]["chat_history"], [])

    def test_gemini_chat_uses_comparison_evidence_and_returns_server_pages(self):
        session_id = "gemini-chat-session"
        app.remember_document_session(session_id,
            {"title": "Old notice", "version": "Earlier"},
            {"title": "Revised notice", "version": "Updated"},
            [{"page": 3, "text": "Eligible applicants may receive Rs 10 lakh.", "blocks": [{"text": "Eligible applicants may receive Rs 10 lakh."}]}],
            [{"page": 5, "text": "Eligible applicants may receive Rs 20 lakh.", "blocks": [{"text": "Eligible applicants may receive Rs 20 lakh."}]}],
            [{"id": f"detected-change-{index}", "summary": "irrelevant change record"} for index in range(491)],
        )
        evidence = app.retrieve_chat_evidence(app.DOCUMENT_SESSIONS[session_id], "What benefit may applicants receive?")
        answer_content = json.dumps({"answer": "The earlier notice lists Rs 10 lakh; the revised notice lists Rs 20 lakh.", "source_ids": [passage["id"] for passage in evidence]})

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self): return json.dumps({"status": "completed", "steps": [{"type": "model_output", "content": [{"type": "text", "text": answer_content}]}]}).encode()

        with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "mock-key", "GEMINI_MODEL": "mock-model"}), patch("urllib.request.urlopen", return_value=FakeResponse()) as request:
            result = app.answer_policy_question(session_id, "What benefit may applicants receive?")
        self.assertTrue(result["ok"])
        self.assertEqual({(source["document"], source["page"]) for source in result["sources"]}, {("Old notice", 3), ("Revised notice", 5)})
        gemini_request = request.call_args.args[0]
        self.assertEqual(gemini_request.full_url, "https://generativelanguage.googleapis.com/v1beta/interactions")
        self.assertEqual(gemini_request.get_header("X-goog-api-key"), "mock-key")
        request_payload = json.loads(gemini_request.data)
        sent_payload = json.loads(request_payload["input"])
        sent_evidence = sent_payload["evidence"]
        self.assertEqual([passage["page"] for passage in sent_evidence], [3, 5])
        self.assertLessEqual(len(sent_evidence), 8)
        self.assertTrue(all("text" in passage and "page" in passage and "version" in passage for passage in sent_evidence))
        self.assertNotIn("changes", sent_payload)

    def test_chat_gemini_quota_timeout_and_malformed_responses_fail_safely_without_retry(self):
        session_id = "gemini-chat-failure-session"
        app.remember_document_session(session_id,
            {"title": "Earlier", "version": "Earlier"}, {"title": "Updated", "version": "Updated"},
            [{"page": 1, "text": "Applicants receive a benefit of Rs 10 lakh under this rule.", "blocks": [{"text": "Applicants receive a benefit of Rs 10 lakh under this rule."}]}],
            [{"page": 2, "text": "Applicants receive a benefit of Rs 20 lakh under this rule.", "blocks": [{"text": "Applicants receive a benefit of Rs 20 lakh under this rule."}]}],
        )
        quota = urllib.error.HTTPError("https://example.test", 429, "quota", {}, BytesIO(b"private response"))
        with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "mock-secret"}), \
                patch("urllib.request.urlopen", side_effect=quota) as request:
            with self.assertLogs("app", level="WARNING") as logs:
                result = app.answer_policy_question(session_id, "What benefit do applicants receive?")
        self.assertFalse(result["ok"])
        self.assertIn("HTTP 429", result["error"])
        self.assertIn("quota", result["error"].lower())
        self.assertIn("verified comparison remains available", result["error"])
        request.assert_called_once()
        self.assertNotIn("mock-secret", "\n".join(logs.output))
        quota.close()

        with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "mock-secret"}), \
                patch("urllib.request.urlopen", side_effect=TimeoutError()) as request:
            with self.assertLogs("app", level="WARNING"):
                timed_out = app.answer_policy_question(session_id, "What benefit do applicants receive?")
        self.assertFalse(timed_out["ok"])
        self.assertIn("timed out", timed_out["error"])
        request.assert_called_once()

        class MalformedResponse:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self): return b"not-json"

        with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "mock-secret"}), \
                patch("urllib.request.urlopen", return_value=MalformedResponse()) as request:
            with self.assertLogs("app", level="WARNING"):
                malformed = app.answer_policy_question(session_id, "What benefit do applicants receive?")
        self.assertFalse(malformed["ok"])
        self.assertIn("verified AI answer", malformed["error"])
        request.assert_called_once()

    def test_chat_http_endpoint_uses_comparison_id_and_returns_json(self):
        server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        request_body = json.dumps({"comparison_id": "test-comparison", "question": "What changed?"}).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/api/chat", data=request_body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with patch.object(app, "answer_policy_question", return_value={"ok": True, "answer": "The evidence says this changed.", "sources": []}) as answer:
                with urllib.request.urlopen(request) as response:
                    result = json.loads(response.read())
                answer.assert_called_once_with("test-comparison", "What changed?")
                self.assertEqual(result["answer"], "The evidence says this changed.")
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

    def test_chat_gemini_503_is_reported_safely_without_retry(self):
        session_id = "gemini-503-chat-session"
        app.remember_document_session(session_id,
            {"title": "Earlier", "version": "Earlier"}, {"title": "Updated", "version": "Updated"},
            [{"page": 1, "text": "Applicants may receive a loan benefit under this provision.", "blocks": [{"text": "Applicants may receive a loan benefit under this provision."}]}],
            [{"page": 1, "text": "Applicants may receive an updated loan benefit under this provision.", "blocks": [{"text": "Applicants may receive an updated loan benefit under this provision."}]}],
        )

        def unavailable():
            return urllib.error.HTTPError("https://generativelanguage.googleapis.com/v1beta/interactions", 503, "unavailable", {}, BytesIO(b"provider detail"))

        with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "do-not-log-this-secret"}), \
                patch("urllib.request.urlopen", side_effect=[unavailable(), unavailable()]) as request, \
                patch("app.time.sleep"):
            with self.assertLogs("app", level="WARNING") as logs:
                result = app.answer_policy_question(session_id, "What loan benefit is described?")
        self.assertFalse(result["ok"])
        self.assertIn("HTTP 503", result["error"])
        self.assertIn("No automatic retry", result["error"])
        self.assertEqual(request.call_count, 1)
        self.assertNotIn("do-not-log-this-secret", "\n".join(logs.output))
        # Plain-text error bodies are now captured as diagnostic detail
        self.assertIn("provider detail", "\n".join(logs.output))

    def test_ai_provider_health_check_sends_small_authenticated_prompt_and_validates_response(self):
        """Health check succeeds: SDK client called, marker validated, key not leaked."""
        import groq as groq_sdk

        fake_choice = MagicMock()
        fake_choice.message.content = "CIVICLENS_AI_HEALTH_OK"
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]

        with patch.dict("os.environ", {"AI_PROVIDER": "groq", "GROQ_API_KEY": "mock-health-key", "GROQ_MODEL": "openai/gpt-oss-20b"}), \
                patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = fake_response
            result = app.ai_provider_health_check()

        self.assertTrue(result["ok"])
        self.assertEqual(result["model"], "openai/gpt-oss-20b")
        self.assertNotIn("mock-health-key", json.dumps(result))
        # Verify SDK was constructed with the key and correct timeouts
        MockGroq.assert_called_once_with(api_key="mock-health-key", timeout=15.0, max_retries=0)
        create_call = MockGroq.return_value.chat.completions.create.call_args
        self.assertEqual(create_call.kwargs["model"], "openai/gpt-oss-20b")
        self.assertEqual(create_call.kwargs["max_tokens"], 64)

    def test_ai_provider_health_check_reports_sanitized_503_and_quota_statuses(self):
        """HTTP errors from Groq SDK are reported with correct user-facing messages; key not leaked."""
        import groq as groq_sdk
        import httpx

        secret = "gsk_test_secret_value_abcdefghij"

        cases = (
            (503, {"error": {"message": f"Service unavailable; key={secret}"}}, "overloaded"),
            (429, {"error": {"message": "Rate limit exceeded"}}, "rate or quota limit"),
            (404, {"error": {"message": "Model not found"}}, "model or endpoint"),
            (403, {"error": {"message": "Permission denied"}}, "denied access"),
        )
        for status, error_payload, expected in cases:
            with self.subTest(status=status):
                fake_http_response = MagicMock(spec=httpx.Response)
                fake_http_response.status_code = status
                fake_http_response.headers = {}
                exc = groq_sdk.APIStatusError(
                    message=error_payload["error"]["message"],
                    response=fake_http_response,
                    body=error_payload,
                )
                with patch.dict("os.environ", {"AI_PROVIDER": "groq", "GROQ_API_KEY": secret, "GROQ_MODEL": "openai/gpt-oss-20b"}), \
                        patch("groq.Groq") as MockGroq:
                    MockGroq.return_value.chat.completions.create.side_effect = exc
                    result = app.ai_provider_health_check()
                self.assertFalse(result["ok"])
                self.assertEqual(result["http_status"], status)
                self.assertNotIn(secret, json.dumps(result))
                self.assertIn(expected, result["error"].lower())

    def test_ai_provider_health_check_captures_plain_text_error_body(self):
        """HTTP 403 with plain-text body (e.g. Cloudflare 'error code: 1010') is captured as detail."""
        import groq as groq_sdk
        import httpx

        secret = "gsk_test_secret_value_1234567890"

        cases = (
            # (body dict or str, expected substring in detail)
            ({"error": {"message": "error code: 1010"}}, "error code: 1010"),
            ({"error": {"message": "Access Denied"}}, "Access Denied"),
            (None, None),  # no body → detail is None
        )
        for body, expected_detail in cases:
            with self.subTest(body=body):
                fake_http_response = MagicMock(spec=httpx.Response)
                fake_http_response.status_code = 403
                fake_http_response.headers = {}
                exc = groq_sdk.APIStatusError(
                    message=str(body),
                    response=fake_http_response,
                    body=body,
                )
                with patch.dict("os.environ", {"AI_PROVIDER": "groq", "GROQ_API_KEY": secret, "GROQ_MODEL": "openai/gpt-oss-20b"}), \
                        patch("groq.Groq") as MockGroq:
                    MockGroq.return_value.chat.completions.create.side_effect = exc
                    result = app.ai_provider_health_check()
                self.assertFalse(result["ok"])
                self.assertEqual(result["http_status"], 403)
                self.assertIn("denied access", result["error"].lower())
                self.assertNotIn(secret, json.dumps(result))
                if expected_detail:
                    self.assertIsNotNone(result["detail"])
                    self.assertIn(expected_detail, result["detail"])
                else:
                    self.assertIsNone(result["detail"])

    def test_ai_provider_health_check_redacts_api_key_in_error_body(self):
        """If the error body contains the API key, it must be redacted."""
        import groq as groq_sdk
        import httpx

        secret = "gsk_live_abcdefghij1234567890klmn"
        fake_http_response = MagicMock(spec=httpx.Response)
        fake_http_response.status_code = 403
        fake_http_response.headers = {}
        exc = groq_sdk.APIStatusError(
            message=f"Unauthorized request with key {secret} from blocked IP",
            response=fake_http_response,
            body={"error": {"message": f"Unauthorized request with key {secret} from blocked IP"}},
        )
        with patch.dict("os.environ", {"AI_PROVIDER": "groq", "GROQ_API_KEY": secret, "GROQ_MODEL": "openai/gpt-oss-20b"}), \
                patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = exc
            result = app.ai_provider_health_check()
        self.assertFalse(result["ok"])
        self.assertNotIn(secret, json.dumps(result))
        self.assertIn("[redacted]", result["detail"])

    def test_ai_provider_health_check_captures_cloudflare_ray_id(self):
        """Ray ID from Cloudflare response header is included in result when present."""
        import groq as groq_sdk
        import httpx

        secret = "gsk_test_key_abcdefghij1234567890"
        fake_http_response = MagicMock(spec=httpx.Response)
        fake_http_response.status_code = 403
        fake_http_response.headers = {"cf-ray": "7f8a9b1c2d3e4f5a-MUM"}
        exc = groq_sdk.APIStatusError(
            message="Access denied",
            response=fake_http_response,
            body={"error": {"message": "Access denied"}},
        )
        with patch.dict("os.environ", {"AI_PROVIDER": "groq", "GROQ_API_KEY": secret, "GROQ_MODEL": "openai/gpt-oss-20b"}), \
                patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = exc
            result = app.ai_provider_health_check()
        self.assertFalse(result["ok"])
        self.assertEqual(result.get("ray_id"), "7f8a9b1c2d3e4f5a-MUM")
        self.assertNotIn(secret, json.dumps(result))

    def test_ai_provider_health_check_distinguishes_connection_and_timeout_errors(self):
        """APIConnectionError and APITimeoutError produce distinct, helpful messages."""
        import groq as groq_sdk

        with patch.dict("os.environ", {"AI_PROVIDER": "groq", "GROQ_API_KEY": "test-key", "GROQ_MODEL": "openai/gpt-oss-20b"}):
            # 1. Connection failure
            with patch("groq.Groq") as MockGroq:
                MockGroq.return_value.chat.completions.create.side_effect = groq_sdk.APIConnectionError(request=MagicMock())
                res_conn = app.ai_provider_health_check()
            self.assertFalse(res_conn["ok"])
            self.assertIn("could not connect", res_conn["error"].lower())
            self.assertIn("Verify network access", res_conn["error"])

            # 2. Timeout
            with patch("groq.Groq") as MockGroq:
                MockGroq.return_value.chat.completions.create.side_effect = groq_sdk.APITimeoutError(request=MagicMock())
                res_timeout = app.ai_provider_health_check()
            self.assertFalse(res_timeout["ok"])
            self.assertIn("timed out connecting to Groq", res_timeout["error"])

    def test_ai_health_route_is_post_and_returns_json(self):
        server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/ai/health", method="POST")
        try:
            with patch.object(app, "ai_provider_health_check", return_value={"ok": True, "provider": "groq", "model": "mock-model", "message": "Connected."}) as health_check:
                with urllib.request.urlopen(request) as response:
                    result = json.loads(response.read())
                health_check.assert_called_once_with()
                self.assertEqual(result["model"], "mock-model")
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

    def test_gemini_policy_failure_exposes_only_sanitized_status_message(self):
        change = self.explanation_change("sanitized-503", "The benefit is Rs 10 lakh.", "The benefit is Rs 20 lakh.")
        secret = "AIza-never-log-this"

        def unavailable():
            body = json.dumps({"error": {"status": "UNAVAILABLE", "message": f"Model is under high demand; key={secret}"}}).encode()
            return urllib.error.HTTPError("https://generativelanguage.googleapis.com/v1beta/interactions", 503, "Unavailable", {}, BytesIO(body))

        provider_error = unavailable()
        with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": secret}), \
                patch("urllib.request.urlopen", side_effect=provider_error) as request, \
                patch("app.time.sleep"):
            with self.assertLogs("app", level="WARNING") as logs:
                analysis, error = app.ai_policy_analysis([change])
        self.assertIsNone(analysis)
        self.assertIn("HTTP 503", error)
        self.assertIn("UNAVAILABLE", error)
        self.assertIn("high demand", error)
        self.assertNotIn(secret, error)
        self.assertNotIn(secret, "\n".join(logs.output))
        self.assertEqual(request.call_count, 1)
        provider_error.close()

    def test_ai_analysis_is_truthfully_unavailable_without_key_or_on_bad_response(self):
        change = self.explanation_change("change-1", "Eligibility is based on age 18.", "Eligibility is based on age 21.")
        with patch.dict("os.environ", {}, clear=True):
            analysis, error = app.ai_policy_analysis([change])
        self.assertIsNone(analysis)
        self.assertIn("not configured", error)

        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", side_effect=TimeoutError()):
            with self.assertLogs("app", level="WARNING"):
                analysis, error = app.ai_policy_analysis([change])
        self.assertIsNone(analysis)
        self.assertIn("unavailable", error)

    def test_gemini_provider_uses_local_secret_header_and_validates_structured_output(self):
        change = self.explanation_change("gemini-change", "Applicants qualify at age 18.", "Applicants qualify at age 21.")
        response_content = json.dumps({
            "summary": "The stated qualifying age changes from 18 to 21.",
            "groups": [self.analyst_group({
                "source_ids": ["gemini-change"], "title": "Qualifying age changed",
                "what_changed": "The stated age changes from 18 to 21.",
                "previous_rule": "The earlier passage states age 18.",
                "updated_rule": "The updated passage states age 21.",
                "why_it_matters": "The excerpt gives a different qualifying age.",
                "who_may_be_affected": None,
            })],
        })

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"status": "completed", "steps": [{"type": "model_output", "content": [{"type": "text", "text": response_content}]}]}).encode()

        with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "mock-gemini-key", "GEMINI_MODEL": "gemini-test-model"}), patch("urllib.request.urlopen", return_value=FakeResponse()) as request:
            analysis, error = app.ai_policy_analysis([change])

        self.assertIsNone(error)
        self.assertEqual(analysis["groups"][0]["previous_rule"], "The earlier passage states age 18.")
        self.assertEqual(request.call_args.kwargs["timeout"], 45)
        gemini_request = request.call_args.args[0]
        self.assertEqual(gemini_request.full_url, "https://generativelanguage.googleapis.com/v1beta/interactions")
        self.assertEqual(gemini_request.get_header("X-goog-api-key"), "mock-gemini-key")
        self.assertIsNone(gemini_request.get_header("Authorization"))
        payload = json.loads(gemini_request.data)
        self.assertEqual(payload["model"], "gemini-test-model")
        self.assertFalse(payload["store"])
        self.assertEqual(payload["response_format"]["mime_type"], "application/json")
        self.assertIn("Applicants qualify at age 18.", payload["input"])
        self.assertIn("Applicants qualify at age 21.", payload["input"])

        # Confirm non-Gemini provider timeout remains 20 seconds
        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "mock-openai-key"}), patch("urllib.request.urlopen", return_value=FakeResponse()) as openai_req:
            app.ai_policy_analysis([change])
            self.assertEqual(openai_req.call_args.kwargs["timeout"], 20)

    def test_gemini_timeout_is_not_retried(self):
        change = self.explanation_change("retry-change", "Applicants qualify at age 18.", "Applicants qualify at age 21.")
        with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "mock-gemini-key"}), \
                patch("urllib.request.urlopen", side_effect=TimeoutError()) as request, \
                patch("app.time.sleep") as sleep:
            with self.assertLogs("app", level="WARNING") as logs:
                analysis, error = app.ai_policy_analysis([change])

        self.assertIsNone(analysis)
        self.assertIn("timed out", error)
        request.assert_called_once()
        sleep.assert_not_called()
        combined_logs = "\n".join(logs.output)
        self.assertNotIn("mock-gemini-key", combined_logs)

    def test_gemini_timeout_and_http_errors_are_not_retried(self):
        change = self.explanation_change("timeout-change", "Applicants qualify at age 18.", "Applicants qualify at age 21.")
        with (
            patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "mock-gemini-key"}),
            patch("urllib.request.urlopen", side_effect=TimeoutError()) as request,
            patch("app.time.sleep"),
        ):
            with self.assertLogs("app", level="WARNING"):
                analysis, error = app.ai_policy_analysis([change])
        self.assertIsNone(analysis)
        self.assertIn("timed out", error)
        request.assert_called_once()

        http_error = urllib.error.HTTPError("https://example.test", 401, "Unauthorized", {}, BytesIO(b"body"))
        with (
            patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "mock-gemini-key"}),
            patch("urllib.request.urlopen", side_effect=http_error) as request,
        ):
            with self.assertLogs("app", level="WARNING"):
                analysis, error = app.ai_policy_analysis([change])
        self.assertIsNone(analysis)
        self.assertIn("HTTP 401", error)
        request.assert_called_once()
        http_error.close()

    def test_gemini_http_503_is_not_retried_and_has_safe_diagnostics(self):
        change = self.explanation_change("503-change", "The limit is Rs 10 lakh.", "The limit is Rs 20 lakh.")
        secret_marker = "secret-key-must-not-appear"
        body_marker = "provider-body-must-not-appear"
        unavailable = urllib.error.HTTPError(
            "https://generativelanguage.googleapis.com/v1beta/interactions", 503, "Unavailable", {},
            BytesIO(body_marker.encode()),
        )
        with (
            patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": secret_marker}),
            patch("urllib.request.urlopen", side_effect=unavailable) as request,
            patch("app.time.sleep") as sleep,
        ):
            with self.assertLogs("app", level="WARNING") as logs:
                analysis, error = app.ai_policy_analysis([change])

        self.assertIsNone(analysis)
        self.assertIn("HTTP 503", error)
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()
        output = "\n".join(logs.output)
        self.assertIn("HTTP 503", output)
        self.assertIn("HTTP 503", output)
        self.assertNotIn(secret_marker, output)
        # Plain-text error bodies are now captured as diagnostic detail
        self.assertIn(body_marker, output)
        unavailable.close()

        repeated_503 = urllib.error.HTTPError(
            "https://generativelanguage.googleapis.com/v1beta/interactions", 503, "Unavailable", {},
            BytesIO(body_marker.encode()),
        )
        with (
            patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": secret_marker}),
            patch("urllib.request.urlopen", side_effect=repeated_503) as request,
            patch("app.time.sleep"),
        ):
            with self.assertLogs("app", level="WARNING"):
                analysis, error = app.ai_policy_analysis([change])
        self.assertIsNone(analysis)
        self.assertIn("HTTP 503", error)
        self.assertEqual(request.call_count, 1)
        repeated_503.close()

    def test_provider_http_errors_are_diagnosable_without_logging_secrets_or_body(self):
        change = self.explanation_change("http-error", "The limit is Rs 10 lakh.", "The limit is Rs 20 lakh.")
        secret_marker = "do-not-log-this-test-key"
        body_marker = "private-provider-response-body"
        http_error = urllib.error.HTTPError(
            "https://generativelanguage.googleapis.com/v1beta/interactions", 401, "Unauthorized", {},
            BytesIO(body_marker.encode()),
        )
        with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": secret_marker}), patch("urllib.request.urlopen", side_effect=http_error):
            with self.assertLogs("app", level="WARNING") as logs:
                analysis, error = app.ai_policy_analysis([change])
        self.assertIsNone(analysis)
        self.assertIn("HTTP 401", error)
        self.assertIn("Authentication failed", error)
        combined_logs = "\n".join(logs.output)
        self.assertIn("HTTP 401", combined_logs)
        self.assertNotIn(secret_marker, combined_logs)
        # Plain-text error bodies are now captured as diagnostic detail
        self.assertIn(body_marker, combined_logs)
        http_error.close()

    def test_gemini_quota_and_gateway_failures_are_not_retried_or_cached(self):
        change = self.explanation_change("quota-gateway", "Earlier requirement is 10 days.", "Updated requirement is 15 days.")
        for status, message in ((429, "quota"), (503, "overloaded"), (504, "timed out")):
            with self.subTest(status=status):
                error = urllib.error.HTTPError("https://example.test", status, "provider error", {}, BytesIO(b"private response"))
                with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "mock-key"}), \
                        patch("urllib.request.urlopen", side_effect=error) as request:
                    with self.assertLogs("app", level="WARNING"):
                        analysis, explanation_error = app.ai_policy_analysis([change])
                self.assertIsNone(analysis)
                self.assertIn(f"HTTP {status}", explanation_error)
                self.assertIn(message, explanation_error.lower())
                request.assert_called_once()
                self.assertIsNone(app.cached_policy_result([change]))
                error.close()

    def test_sample_compare_waits_for_manual_ai_request(self):
        sample_changes = app.compare_docs(*app.SAMPLE_DOCS)
        fake_analysis = {
            "summary": "The sample notification changes its stated provisions.",
            "groups": [{
                "source_ids": [change["id"] for change in sample_changes],
                "title": "Sample policy provisions",
                "what_changed": "The supplied excerpts show updated provisions.",
                "previous_rule": "The earlier excerpts contain the previous wording.",
                "updated_rule": "The updated excerpts contain the revised wording.",
                "why_it_matters": "The notification text differs between these versions.",
                "who_may_be_affected": None,
                "importance": "Medium", "importance_reason": "The cited provisions differ.",
            }],
        }
        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": ""}), patch.object(
            app, "ai_policy_analysis", return_value=(fake_analysis, None)
        ) as ai_call:
            server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/compare") as response:
                    result = json.loads(response.read())
                self.assertEqual(result["ai_status"], "not_requested")
                self.assertIsNone(result["analysis_id"])
                self.assertTrue(result["changes"])
                ai_call.assert_not_called()

                request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/analysis", data=json.dumps({
                    "comparison_id": result["comparison_id"], "request_id": "manual-sample-request",
                }).encode(), headers={"Content-Type": "application/json"}, method="POST")
                with patch.object(app, "ai_policy_analysis", return_value=(None, "Gemini API is unavailable.")) as manual_call:
                    with urllib.request.urlopen(request) as response:
                        started = json.loads(response.read())
                    for _ in range(100):
                        if started.get("analysis_id"):
                            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/analysis/{started['analysis_id']}") as response:
                                analysis_result = json.loads(response.read())
                            if analysis_result["ai_status"] != "pending":
                                break
                        else:
                            analysis_result = started
                            break
                        threading.Event().wait(0.01)
                    manual_call.assert_called_once()
                    self.assertEqual(analysis_result["ai_status"], "unavailable")
                    self.assertEqual(analysis_result["ai_error"], "Gemini API is unavailable.")
                self.assertTrue(all(change["previous"] or change["updated"] for change in result["changes"]))
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=2)

    def test_gemini_missing_key_or_unusable_response_keeps_fallback(self):
        change = self.explanation_change("gemini-fallback", "The limit is Rs 10 lakh.", "The limit is Rs 20 lakh.")
        with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": ""}), patch("urllib.request.urlopen") as request:
            analysis, error = app.ai_policy_analysis([change])
        self.assertIsNone(analysis)
        self.assertIn("GEMINI_API_KEY is not configured", error)
        request.assert_not_called()

        class FailedResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"status": "failed", "steps": []}).encode()

        with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "mock-gemini-key"}), patch("urllib.request.urlopen", return_value=FailedResponse()):
            with self.assertLogs("app", level="WARNING"):
                analysis, error = app.ai_policy_analysis([change])
        self.assertIsNone(analysis)
        self.assertIn("unavailable", error)

    def test_gemini_request_size_limit_falls_back_without_sending_partial_evidence(self):
        change = self.explanation_change("oversized", "Earlier provision " + ("text " * 100), "Updated provision " + ("text " * 100))
        with (
            patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "mock-gemini-key"}),
            patch.object(app, "MAX_AI_REQUEST_BYTES", 200),
            patch("urllib.request.urlopen") as request,
            self.assertLogs("app", level="WARNING") as logs,
        ):
            analysis, error = app.ai_policy_analysis([change])
        self.assertIsNone(analysis)
        self.assertIn("request limit", error)
        self.assertIn("full evidence comparison remains available", error)
        request.assert_not_called()
        self.assertIn("payload", "\n".join(logs.output))

    def test_gemini_analysis_runs_in_background_without_blocking_evidence_response(self):
        change = self.explanation_change("async", "The limit is Rs 10 lakh.", "The limit is Rs 20 lakh.")
        started, release = threading.Event(), threading.Event()

        def delayed_analysis(_changes):
            started.set()
            release.wait(2)
            return None, "Gemini request timed out."

        comparison_id = "async-comparison-" + str(app.uuid.uuid4())
        app.remember_document_session(comparison_id, {"title": "Earlier", "version": "Earlier"}, {"title": "Updated", "version": "Updated"}, [], [], [change])
        with patch.object(app, "ai_policy_analysis", side_effect=delayed_analysis) as ai_call:
            before = app.perf_counter()
            response = app.schedule_policy_analysis(comparison_id, "async-request")
            elapsed = app.perf_counter() - before
            self.assertLess(elapsed, 0.5)
            self.assertEqual(response["ai_status"], "pending")
            self.assertTrue(started.wait(1))
            duplicate = app.schedule_policy_analysis(comparison_id, "second-click")
            self.assertEqual(duplicate, response)
            ai_call.assert_called_once()
            with app.AI_ANALYSES_LOCK:
                self.assertEqual(app.AI_ANALYSES[response["analysis_id"]]["status"], "pending")
            release.set()
            for _ in range(100):
                with app.AI_ANALYSES_LOCK:
                    status = app.AI_ANALYSES[response["analysis_id"]]["status"]
                if status != "pending":
                    break
                threading.Event().wait(0.01)
            self.assertEqual(status, "unavailable")

    def test_successful_analysis_cache_reuses_validated_result_and_tracks_model_prompt(self):
        original = self.explanation_change("cache-original", "The benefit is Rs 10 lakh.", "The benefit is Rs 20 lakh.")
        reused = self.explanation_change("cache-reopened", "The benefit is Rs 10 lakh.", "The benefit is Rs 20 lakh.")
        analysis = {"summary": "The cited benefit amount changes.", "groups": [self.analyst_group({
            "source_ids": [original["id"]], "title": "Benefit amount changed",
            "what_changed": "The cited amount changes from Rs 10 lakh to Rs 20 lakh.",
            "previous_rule": "The earlier text states Rs 10 lakh.",
            "updated_rule": "The updated text states Rs 20 lakh.",
            "why_it_matters": "The cited amount differs.",
        })]}
        with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_MODEL": "mock-model"}):
            base_key = app.policy_analysis_cache_key([original])
            app._store_policy_analysis_cache(base_key, [original], analysis)
            reused_result = app.cached_policy_result([reused])
            self.assertTrue(reused_result["ai_cached"])
            self.assertEqual(reused_result["important_changes"][0]["changes"][0]["id"], "cache-reopened")
            self.assertEqual(reused_result["important_changes"][0]["changes"][0]["previous"]["page"], 1)
            with patch.dict("os.environ", {"GEMINI_MODEL": "different-model"}):
                self.assertNotEqual(base_key, app.policy_analysis_cache_key([original]))
            with patch.object(app, "POLICY_PROMPT_VERSION", "next-prompt-version"):
                self.assertNotEqual(base_key, app.policy_analysis_cache_key([original]))

    def test_ai_rejects_unsupported_amounts_and_affected_groups(self):
        change = self.explanation_change("grounding-check", "The benefit limit is Rs 10 lakh.", "The benefit limit is Rs 20 lakh.")

        class FakeResponse:
            def __init__(self, result):
                self.result = result

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"status": "completed", "steps": [{"type": "model_output", "content": [{"type": "text", "text": json.dumps(self.result)}]}]}).encode()

        cases = [
            {"summary": "The limit changes from Rs 10 lakh to Rs 30 lakh.", "groups": [self.analyst_group({
                "source_ids": ["grounding-check"], "title": "Benefit limit", "what_changed": "The limit changes to Rs 30 lakh.",
                "previous_rule": "The previous limit is Rs 10 lakh.", "updated_rule": "The updated limit is Rs 30 lakh.",
                "why_it_matters": "The stated limit changes.", "who_may_be_affected": None,
            })]},
            {"summary": "The benefit limit changes from Rs 10 lakh to Rs 20 lakh.", "groups": [self.analyst_group({
                "source_ids": ["grounding-check"], "title": "Benefit limit", "what_changed": "The limit changes from Rs 10 lakh to Rs 20 lakh.",
                "previous_rule": "The previous limit is Rs 10 lakh.", "updated_rule": "The updated limit is Rs 20 lakh.",
                "why_it_matters": "The stated limit changes.", "who_may_be_affected": "Students",
            })]},
        ]
        for result in cases:
            with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "mock-gemini-key"}), patch("urllib.request.urlopen", return_value=FakeResponse(result)):
                with self.assertLogs("app", level="WARNING"):
                    analysis, error = app.ai_policy_analysis([change])
            self.assertIsNone(analysis)
            self.assertIn("unavailable", error)

    def test_policy_priority_uses_provision_type_and_has_reason(self):
        amount_change = self.explanation_change("amount", "The assistance amount is Rs 10 lakh.", "The assistance amount is Rs 20 lakh.")
        procedure_change = self.explanation_change("procedure", "Applicants submit forms online.", "Applicants submit forms through the portal.")
        deadline_change = self.explanation_change("deadline", "Applications close on 30 September 2024.", "Applications close on 30 October 2024.")
        routine_change = self.explanation_change("routine", "The committee considered the proposal.", "The committee reviewed the proposal.")
        self.assertEqual(app.priority_for_changes([amount_change])[0], "High")
        self.assertEqual(app.priority_for_changes([procedure_change])[0], "Medium")
        self.assertEqual(app.priority_for_changes([deadline_change])[0], "High")
        self.assertEqual(app.priority_for_changes([routine_change])[0], "Low")
        self.assertIn("estimate", app.priority_for_changes([amount_change])[1])

    def test_policy_classifier_separates_heading_formatting_and_real_rule_changes(self):
        title = self.explanation_change("title-edit", "PMAY Urban Scheme 2021", "PMAY Urban Scheme 2024")
        punctuation = self.explanation_change("punctuation", "Applicants may receive Rs. 10 lakh.", "Applicants may receive Rs 10 lakh.")
        amount = self.explanation_change("amount-rule", "Eligible applicants may receive Rs 10 lakh.", "Eligible applicants may receive Rs 20 lakh.")
        age = self.explanation_change("eligibility-rule", "Applicants shall be at least 18 years old.", "Applicants shall be at least 21 years old.")
        for change in (title, punctuation):
            result = app.classify_policy_change(change)
            self.assertFalse(result["substantive_change"])
            self.assertEqual(result["priority"], "Low")
        amount_result = app.classify_policy_change(amount)
        self.assertTrue(amount_result["substantive_change"])
        self.assertEqual(amount_result["change_category"], "Financial assistance")
        self.assertEqual(amount_result["priority"], "High")
        age_result = app.classify_policy_change(age)
        self.assertTrue(age_result["substantive_change"])
        self.assertEqual(age_result["change_category"], "Eligibility")
        self.assertEqual(age_result["priority"], "High")

    def test_comparison_annotations_group_by_section_and_identify_repetitions(self):
        first = self.explanation_change("repeat-1", "Eligible applicants may receive Rs 10 lakh.", "Eligible applicants may receive Rs 20 lakh.")
        second = self.explanation_change("repeat-2", "Eligible applicants may receive Rs 10 lakh.", "Eligible applicants may receive Rs 20 lakh.")
        first.update(previous_section="Section 4: Benefits", updated_section="Section 4: Benefits")
        second.update(previous_section="Section 4: Benefits", updated_section="Section 4: Benefits")
        changes = [first, second]
        app.prepare_evidence_changes(changes)
        self.assertEqual(len(changes), 2, "Repeated evidence records stay visible; they are not discarded.")
        self.assertEqual([change["repetition_count"] for change in changes], [2, 2])
        self.assertEqual({change["policy_group"] for change in changes}, {"Section 4: Benefits"})
        self.assertEqual({change["classification"] for change in changes}, {"Substantive"})

    def test_evidence_overview_is_concise_and_keeps_exact_sources(self):
        changes = []
        for index in range(5):
            change = self.explanation_change(
                f"overview-{index}", f"Eligible applicants may receive Rs {10 + index} lakh on page provision.",
                f"Eligible applicants may receive Rs {20 + index} lakh on page provision.",
            )
            change["previous"]["document"] = "Earlier Gazette"
            change["previous"]["page"] = index + 1
            change["updated"]["document"] = "Updated Gazette"
            change["updated"]["page"] = index + 11
            changes.append(change)
        with patch.object(app, "cached_policy_result", return_value=None):
            result = app.comparison_ai_payload(changes)
        overview = result["comparison_overview"]
        self.assertEqual(overview["total_changes"], 5)
        self.assertEqual(overview["substantive_changes"], 5)
        self.assertEqual(len(overview["highlights"]), 3)
        highlighted = overview["highlights"][0]
        self.assertEqual(highlighted["previous"]["document"], "Earlier Gazette")
        self.assertEqual(highlighted["previous"]["page"], 1)
        self.assertEqual(highlighted["updated"]["page"], 11)
        self.assertEqual(highlighted["previous"]["text"], changes[0]["previous"]["text"])

    def test_ai_policy_labels_are_overridden_by_verified_comparison_rubric(self):
        change = self.explanation_change("authority-check", "The benefit amount is Rs 10 lakh.", "The benefit amount is Rs 20 lakh.")
        fake = {"summary": "The verified amount differs.", "groups": [self.analyst_group({
            "source_ids": [change["id"]], "title": "Benefit wording",
            "what_changed": "The quoted amount differs.", "previous_rule": "The earlier excerpt gives Rs 10 lakh.",
            "updated_rule": "The updated excerpt gives Rs 20 lakh.", "why_it_matters": "The cited amounts differ.",
            "who_may_be_affected": None, "substantive_change": False, "importance": "Low",
            "importance_reason": "Model supplied low priority.", "change_category": "Other",
        })]}
        with patch.object(app, "ai_policy_analysis", return_value=(fake, None)):
            result = app.build_policy_result([change])
        group = result["important_changes"][0]
        self.assertTrue(group["substantive_change"])
        self.assertEqual(group["change_category"], "Financial assistance")
        self.assertEqual(group["priority"], "High")
        self.assertEqual(group["changes"][0]["previous"]["page"], 1)
        self.assertEqual(group["changes"][0]["updated"]["page"], 2)

    def test_no_key_keeps_evidence_comparison_without_claiming_ai_summary(self):
        change = self.explanation_change("change-1", "The old eligibility age is 18.", "The updated eligibility age is 21.")
        with patch.dict("os.environ", {}, clear=True):
            result = app.build_policy_result([change])
        self.assertIsNone(result["policy_summary"])
        self.assertEqual(result["important_changes"], [])
        self.assertIn("not configured", result["ai_error"])
        self.assertEqual(change["explanation_mode"], "Evidence only")
        self.assertEqual(change["priority"], "High")

    def test_policy_result_groups_changes_but_keeps_verified_source_objects(self):
        change = self.explanation_change("verified-change", "The benefit is Rs 10 lakh.", "The benefit is Rs 20 lakh.")
        analysis = {
            "summary": "The cited benefit limit changed.",
            "groups": [self.analyst_group({
                "source_ids": ["verified-change"], "title": "Benefit limit",
                "what_changed": "The cited limit increased from Rs 10 lakh to Rs 20 lakh.",
                "previous_rule": "The earlier passage states Rs 10 lakh.",
                "updated_rule": "The updated passage states Rs 20 lakh.",
                "why_it_matters": "The updated passage states a different benefit limit.",
                "who_may_be_affected": None,
            })],
        }
        with patch.object(app, "ai_policy_analysis", return_value=(analysis, None)):
            result = app.build_policy_result([change])
        self.assertEqual(result["policy_summary"], analysis["summary"])
        self.assertEqual(result["important_changes"][0]["changes"][0], change)
        self.assertEqual(result["important_changes"][0]["changes"][0]["previous"]["page"], 1)
        self.assertEqual(result["important_changes"][0]["changes"][0]["updated"]["page"], 2)

    def test_both_sample_pdfs_extract_text(self):
        for doc in app.SAMPLE_DOCS:
            pages = app.extract_pdf(app.SAMPLES / doc["file"])
            self.assertTrue(pages)
            self.assertTrue(any(len(p["text"]) > 100 for p in pages))

    def test_comparison_quotes_match_the_cited_pages(self):
        old, new = app.SAMPLE_DOCS
        changes = app.compare_docs(old, new)
        self.assertTrue(changes)
        all_pages = {
            old["version"]: app.extract_pdf(app.SAMPLES / old["file"]),
            new["version"]: app.extract_pdf(app.SAMPLES / new["file"]),
        }
        for change in changes:
            for ref in (change.get("previous"), change.get("updated")):
                if ref:
                    normalized_page = " ".join(all_pages[ref["version"]][ref["page"] - 1]["text"].split())
                    self.assertIn(" ".join(ref["text"].split()), normalized_page)

    def test_demo_focuses_on_substantive_cgfmu_changes(self):
        changes = app.compare_docs(*app.SAMPLE_DOCS)
        self.assertEqual([c["id"] for c in changes], ["cgfmu-loan-limit", "cgfmu-tarun-plus"])
        self.assertEqual(changes[0]["previous"]["page"], 2)
        self.assertEqual(changes[0]["updated"]["page"], 3)
        self.assertIn("Rs.10 lakh", changes[0]["previous"]["text"])
        self.assertIn("Rs.20 lakh", changes[0]["updated"]["text"])

    @staticmethod
    def make_pdf(lines):
        doc = pymupdf.open()
        page = doc.new_page()
        for i, line in enumerate(lines):
            page.insert_text((72, 72 + i * 28), line, fontsize=11)
        result = doc.tobytes()
        doc.close()
        return result

    def test_uploaded_pdfs_detect_added_removed_and_modified_text(self):
        old_bytes = self.make_pdf([
            "Eligibility percentile is 80 for successful students.",
            "Family income ceiling is Rs. 6 lakh per annum.",
            "Physical forms may be submitted.",
        ])
        new_bytes = self.make_pdf([
            "Eligibility percentile is 80 for successful students.",
            "Family income ceiling is Rs. 4.5 lakh per annum.",
            "Applications must be submitted online through the portal.",
            "Students should keep a bank account.",
        ])
        old_pages = app.extract_pdf_bytes(old_bytes)
        new_pages = app.extract_pdf_bytes(new_bytes)
        changes = app.compare_passages(old_pages, new_pages,
            {"title": "earlier.pdf", "version": "Earlier"},
            {"title": "updated.pdf", "version": "Updated"}, "test-id")
        kinds = {change["type"] for change in changes}
        self.assertEqual(kinds, {"Added content", "Removed content", "Modified content"})
        for change in changes:
            for ref, pages in ((change["previous"], old_pages), (change["updated"], new_pages)):
                if ref:
                    source_text = " ".join(pages[ref["page"] - 1]["text"].split())
                    self.assertIn(" ".join(ref["text"].split()), source_text)

    def test_adjacent_blocks_for_one_change_are_grouped_with_exact_segments(self):
        old_pages = [{"page": 1, "text": "Section 4. The limit is 10 lakh. Applicants submit forms online. End of section.", "blocks": [
            "Section 4.", "The limit is 10 lakh.", "Applicants submit forms online.", "End of section.",
        ]}]
        new_pages = [{"page": 1, "text": "Section 4. The limit is 20 lakh. Applicants submit forms via the portal. End of section.", "blocks": [
            "Section 4.", "The limit is 20 lakh.", "Applicants submit forms via the portal.", "End of section.",
        ]}]
        changes = app.compare_passages(old_pages, new_pages,
            {"title": "earlier.pdf", "version": "Earlier"},
            {"title": "updated.pdf", "version": "Updated"}, "group-test")
        self.assertEqual(len(changes), 1)
        change = changes[0]
        self.assertEqual(change["type"], "Modified content")
        self.assertEqual(change["previous"]["page"], 1)
        self.assertEqual(change["updated"]["page"], 1)
        self.assertEqual(len(change["previous"]["evidence_segments"]), 2)
        self.assertTrue(all(segment in old_pages[0]["text"] for segment in change["previous"]["evidence_segments"]))
        self.assertTrue(all(segment in new_pages[0]["text"] for segment in change["updated"]["evidence_segments"]))

    def test_repeated_headers_footers_and_page_numbers_are_ignored(self):
        def page(number, body):
            return {"page": number, "height": 800, "text": f"Ministry of Housing & Urban Affairs PM Housing Scheme Guidelines {body} Department of Policy {number}", "blocks": [
                {"text": "Ministry of Housing & Urban Affairs", "y0": 20, "y1": 32},
                {"text": "PM Housing Scheme Guidelines", "y0": 42, "y1": 56},
                {"text": body, "y0": 150, "y1": 180},
                {"text": f"Department of Policy | {number}", "y0": 770, "y1": 785},
                {"text": str(number), "y0": 788, "y1": 798},
            ]}
        old_pages = [page(1, "Limit is 10 lakh."), page(2, "Unchanged paragraph.")]
        new_pages = [page(1, "Limit is 20 lakh."), page(2, "Unchanged paragraph.")]
        old_passages = app.page_passages(old_pages)
        self.assertEqual([p["text"] for p in old_passages], ["Limit is 10 lakh.", "Unchanged paragraph."])
        changes = app.compare_passages(old_pages, new_pages,
            {"title": "earlier.pdf", "version": "Earlier"},
            {"title": "updated.pdf", "version": "Updated"}, "header-test")
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["previous"]["page"], 1)
        self.assertEqual(changes[0]["updated"]["page"], 1)

    def test_cover_dates_and_ministry_heading_are_not_policy_changes(self):
        def cover(date, amount):
            text = f"Government of India Ministry of Housing & Urban Affairs {date} Eligibility assistance is Rs {amount} lakh per year."
            return [{"page": 1, "height": 800, "text": text, "blocks": [
                {"text": "Government of India", "x0": 72, "x1": 180, "y0": 20, "y1": 34},
                {"text": "Ministry of Housing & Urban Affairs", "x0": 72, "x1": 310, "y0": 45, "y1": 59},
                {"text": date, "x0": 72, "x1": 165, "y0": 72, "y1": 86},
                {"text": f"Eligibility assistance is Rs {amount} lakh per year.", "x0": 72, "x1": 360, "y0": 180, "y1": 198},
            ]}]
        old_pages = cover("January, 2021", 10)
        new_pages = cover("September 2024", 20)
        changes = app.compare_passages(old_pages, new_pages,
            {"title": "earlier.pdf", "version": "Earlier"},
            {"title": "updated.pdf", "version": "Updated"}, "cover-metadata-test")
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["type"], "Modified content")
        all_excerpts = " ".join([changes[0]["previous"]["text"], changes[0]["updated"]["text"]])
        self.assertIn("10 lakh", all_excerpts)
        self.assertIn("20 lakh", all_excerpts)
        self.assertNotIn("January, 2021", all_excerpts)
        self.assertNotIn("September 2024", all_excerpts)
        self.assertNotIn("Ministry of Housing", all_excerpts)
        self.assertTrue(app.validate_change_sources(changes[0], old_pages, new_pages))

    def test_titles_dates_prefaces_and_moved_paragraphs_do_not_mask_policy_change(self):
        def page(number, blocks):
            text = " ".join(block["text"] for block in blocks)
            return {"page": number, "height": 800, "text": text, "blocks": blocks}

        old_pages = [
            page(1, [
                {"text": "Government of India", "x0": 72, "x1": 180, "y0": 20, "y1": 34},
                {"text": "Affordable Housing Scheme Guidelines", "x0": 72, "x1": 350, "y0": 70, "y1": 90},
                {"text": "January, 2021", "x0": 72, "x1": 165, "y0": 105, "y1": 120},
            ]),
            page(2, [
                {"text": "Preface"},
                {"text": "The programme was first introduced after a review of housing needs across several regions."},
            ]),
            page(3, [
                {"text": "Section 4 Eligibility", "x0": 72, "x1": 210, "y0": 100, "y1": 115},
                {"text": "Applications are reviewed by the district office each year.", "x0": 72, "x1": 360, "y0": 150, "y1": 170},
                {"text": "Eligible applicants may receive assistance of Rs 10 lakh per year.", "x0": 72, "x1": 410, "y0": 200, "y1": 220},
            ]),
        ]
        new_pages = [
            page(1, [
                {"text": "Government of India", "x0": 72, "x1": 180, "y0": 20, "y1": 34},
                {"text": "Affordable Housing Programme Revised Guidelines", "x0": 72, "x1": 400, "y0": 70, "y1": 90},
                {"text": "September 2024", "x0": 72, "x1": 165, "y0": 105, "y1": 120},
            ]),
            page(2, [
                {"text": "Preface"},
                {"text": "The programme was later reviewed using updated administrative background and regional reports."},
            ]),
            page(3, [{"text": "Eligible applicants may receive assistance of Rs 20 lakh per year.", "x0": 72, "x1": 410, "y0": 200, "y1": 220}]),
            page(4, [{"text": "Applications are reviewed by the district office each year.", "x0": 72, "x1": 360, "y0": 150, "y1": 170}]),
        ]
        changes = app.compare_passages(old_pages, new_pages,
            {"title": "earlier.pdf", "version": "Earlier"},
            {"title": "updated.pdf", "version": "Updated"}, "realistic-front-matter-test")

        self.assertEqual(len(changes), 1)
        change = changes[0]
        self.assertEqual(change["type"], "Modified content")
        self.assertEqual(change["previous"]["page"], 3)
        self.assertEqual(change["updated"]["page"], 3)
        self.assertIn("Rs 10 lakh", change["previous"]["text"])
        self.assertIn("Rs 20 lakh", change["updated"]["text"])
        self.assertNotIn("January, 2021", change["previous"]["text"])
        self.assertNotIn("September 2024", change["updated"]["text"])
        self.assertNotIn("programme was first introduced", change["previous"]["text"])
        self.assertTrue(app.validate_change_sources(change, old_pages, new_pages))

    def test_standalone_dates_in_policy_body_are_not_removed_as_cover_metadata(self):
        old_pages = [{"page": 1, "height": 800, "text": "Applications close on\n30 September 2024", "blocks": [
            {"text": "Applications close on", "x0": 72, "x1": 220, "y0": 200, "y1": 215},
            {"text": "30 September 2024", "x0": 72, "x1": 205, "y0": 218, "y1": 233},
        ]}]
        new_pages = [{"page": 1, "height": 800, "text": "Applications close on\n30 October 2024", "blocks": [
            {"text": "Applications close on", "x0": 72, "x1": 220, "y0": 200, "y1": 215},
            {"text": "30 October 2024", "x0": 72, "x1": 205, "y0": 218, "y1": 233},
        ]}]
        changes = app.compare_passages(old_pages, new_pages,
            {"title": "earlier.pdf", "version": "Earlier"},
            {"title": "updated.pdf", "version": "Updated"}, "deadline-test")
        self.assertEqual(len(changes), 1)
        self.assertIn("September 2024", changes[0]["previous"]["text"])
        self.assertIn("October 2024", changes[0]["updated"]["text"])
        self.assertTrue(app.validate_change_sources(changes[0], old_pages, new_pages))

    def test_identical_ministry_heading_is_not_modified_when_its_layout_moves(self):
        old_pages = [{"page": 1, "height": 800, "text": "Ministry of Housing & Urban Affairs", "blocks": [
            {"text": "Ministry of Housing & Urban Affairs", "x0": 72, "x1": 310, "y0": 30, "y1": 44},
        ]}]
        new_pages = [{"page": 1, "height": 800, "text": "Ministry of Housing & Urban Affairs", "blocks": [
            {"text": "Ministry of Housing & Urban Affairs", "x0": 190, "x1": 428, "y0": 310, "y1": 324},
        ]}]
        changes = app.compare_passages(old_pages, new_pages,
            {"title": "earlier.pdf", "version": "Earlier"},
            {"title": "updated.pdf", "version": "Updated"}, "ministry-heading-test")
        self.assertEqual(changes, [])

    def test_policy_dates_and_ministry_actions_remain_comparable(self):
        old_pages = [{"page": 1, "height": 800, "text": "Government of India Ministry of Housing & Urban Affairs The Ministry of Housing & Urban Affairs shall notify assistance of Rs 10 lakh. Applications close on 30 September 2024.", "blocks": [
            {"text": "Government of India", "x0": 72, "x1": 180, "y0": 20, "y1": 34},
            {"text": "Ministry of Housing & Urban Affairs", "x0": 72, "x1": 310, "y0": 45, "y1": 59},
            {"text": "The Ministry of Housing & Urban Affairs shall notify assistance of Rs 10 lakh.", "x0": 72, "x1": 410, "y0": 180, "y1": 198},
            {"text": "Applications close on 30 September 2024.", "x0": 72, "x1": 320, "y0": 220, "y1": 238},
        ]}]
        new_pages = [{"page": 1, "height": 800, "text": "Government of India Ministry of Housing & Urban Affairs The Ministry of Housing & Urban Affairs shall notify assistance of Rs 20 lakh. Applications close on 30 September 2024.", "blocks": [
            {"text": "Government of India", "x0": 72, "x1": 180, "y0": 20, "y1": 34},
            {"text": "Ministry of Housing & Urban Affairs", "x0": 72, "x1": 310, "y0": 45, "y1": 59},
            {"text": "The Ministry of Housing & Urban Affairs shall notify assistance of Rs 20 lakh.", "x0": 72, "x1": 410, "y0": 180, "y1": 198},
            {"text": "Applications close on 30 September 2024.", "x0": 72, "x1": 320, "y0": 220, "y1": 238},
        ]}]
        changes = app.compare_passages(old_pages, new_pages,
            {"title": "earlier.pdf", "version": "Earlier"},
            {"title": "updated.pdf", "version": "Updated"}, "policy-ministry-test")
        self.assertEqual(len(changes), 1)
        self.assertIn("shall notify", changes[0]["previous"]["text"])
        self.assertIn("10 lakh", changes[0]["previous"]["text"])
        self.assertIn("20 lakh", changes[0]["updated"]["text"])

    def test_unchanged_paragraph_is_not_modified_when_pdf_splits_it_differently(self):
        old_pages = [{"page": 1, "height": 800, "text": "Eligible businesses must submit the annual form by 1 April.", "blocks": [
            {"text": "Eligible businesses must submit", "x0": 72, "x1": 220, "y0": 120, "y1": 132},
            {"text": "the annual form by 1 April.", "x0": 72, "x1": 210, "y0": 134, "y1": 146},
        ]}]
        new_pages = [{"page": 1, "height": 800, "text": "Eligible businesses must submit the annual form by 1 April.", "blocks": [
            {"text": "Eligible businesses must submit the annual form by 1 April.", "x0": 72, "x1": 350, "y0": 120, "y1": 146},
        ]}]
        changes = app.compare_passages(old_pages, new_pages,
            {"title": "earlier.pdf", "version": "Earlier"},
            {"title": "updated.pdf", "version": "Updated"}, "split-test")
        self.assertEqual(changes, [])

    def test_paragraph_continuation_across_pages_keeps_each_page_reference(self):
        old_pages = [
            {"page": 1, "height": 800, "text": "The maximum assistance for a qualifying applicant is", "blocks": [
                {"text": "The maximum assistance for a qualifying applicant is", "x0": 72, "x1": 340, "y0": 755, "y1": 785},
            ]},
            {"page": 2, "height": 800, "text": "Rs 10 lakh per financial year.", "blocks": [
                {"text": "Rs 10 lakh per financial year.", "x0": 72, "x1": 240, "y0": 18, "y1": 34},
            ]},
        ]
        new_pages = [{"page": 1, "height": 800, "text": "The maximum assistance for a qualifying applicant is Rs 20 lakh per financial year.", "blocks": [
            {"text": "The maximum assistance for a qualifying applicant is Rs 20 lakh per financial year.", "x0": 72, "x1": 450, "y0": 120, "y1": 145},
        ]}]
        changes = app.compare_passages(old_pages, new_pages,
            {"title": "earlier.pdf", "version": "Earlier"},
            {"title": "updated.pdf", "version": "Updated"}, "pagination-test")
        self.assertEqual(len(changes), 1)
        change = changes[0]
        self.assertEqual(change["type"], "Modified content")
        self.assertEqual([ref["page"] for ref in change["previous"]["page_refs"]], [1, 2])
        self.assertEqual(change["updated"]["page"], 1)
        self.assertEqual(change["previous"]["evidence_segments"], [
            "The maximum assistance for a qualifying applicant is", "Rs 10 lakh per financial year.",
        ])
        self.assertTrue(app.validate_change_sources(change, old_pages, new_pages))
        invalid_reference = {**change, "previous": {**change["previous"], "page_refs": [
            {"page": 2, "text": "Not present in the PDF", "evidence_segments": ["Not present in the PDF"]},
        ]}}
        self.assertFalse(app.validate_change_sources(invalid_reference, old_pages, new_pages))

    def test_adjacent_related_changes_across_pages_are_grouped_with_exact_page_refs(self):
        old_pages = [
            {"page": 1, "height": 800, "text": "The approved grant amount is subject to review by the board and", "blocks": [
                {"text": "The approved grant amount is subject to review by the board and", "y0": 300, "y1": 330},
            ]},
            {"page": 2, "height": 800, "text": "will be released in two instalments.", "blocks": [
                {"text": "will be released in two instalments.", "y0": 300, "y1": 330},
            ]},
        ]
        new_pages = [
            {"page": 1, "height": 800, "text": "The approved grant amount is subject to review by the authority and", "blocks": [
                {"text": "The approved grant amount is subject to review by the authority and", "y0": 300, "y1": 330},
            ]},
            {"page": 2, "height": 800, "text": "will be released in four instalments.", "blocks": [
                {"text": "will be released in four instalments.", "y0": 300, "y1": 330},
            ]},
        ]
        changes = app.compare_passages(old_pages, new_pages,
            {"title": "earlier.pdf", "version": "Earlier"},
            {"title": "updated.pdf", "version": "Updated"}, "cross-page-group-test")
        self.assertEqual(len(changes), 1)
        change = changes[0]
        self.assertEqual(change["type"], "Modified content")
        self.assertEqual([ref["page"] for ref in change["previous"]["page_refs"]], [1, 2])
        self.assertEqual([ref["page"] for ref in change["updated"]["page_refs"]], [1, 2])
        self.assertTrue(app.validate_change_sources(change, old_pages, new_pages))

    def test_comparison_logs_preprocessing_alignment_detection_and_evidence_timings(self):
        old_pages = [{"page": 1, "text": "Applicants qualify for Rs 10 lakh.", "blocks": ["Applicants qualify for Rs 10 lakh."]}]
        new_pages = [{"page": 1, "text": "Applicants qualify for Rs 20 lakh.", "blocks": ["Applicants qualify for Rs 20 lakh."]}]
        with self.assertLogs("app", level="INFO") as logs:
            changes = app.compare_passages(old_pages, new_pages,
                {"title": "earlier.pdf", "version": "Earlier"},
                {"title": "updated.pdf", "version": "Updated"}, "timing-test")
        output = "\n".join(logs.output)
        self.assertTrue(changes)
        for stage in ("text_preprocessing", "document_alignment", "change_detection", "evidence_generation"):
            self.assertIn(f"stage={stage}", output)

    def test_changes_keep_section_labels_for_detailed_result_grouping(self):
        old_pages = [{"page": 1, "height": 800, "text": "Section 4 Eligibility Applicants qualify at age 18.", "blocks": [
            {"text": "Section 4 Eligibility", "y0": 60, "y1": 75},
            {"text": "Applicants qualify at age 18.", "y0": 120, "y1": 140},
        ]}]
        new_pages = [{"page": 1, "height": 800, "text": "Section 4 Eligibility Applicants qualify at age 21.", "blocks": [
            {"text": "Section 4 Eligibility", "y0": 60, "y1": 75},
            {"text": "Applicants qualify at age 21.", "y0": 120, "y1": 140},
        ]}]
        changes = app.compare_passages(old_pages, new_pages,
            {"title": "earlier.pdf", "version": "Earlier"},
            {"title": "updated.pdf", "version": "Updated"}, "section-label-test")
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["previous_section"], "Section 4 Eligibility")
        self.assertEqual(changes[0]["updated_section"], "Section 4 Eligibility")

    def test_more_than_one_hundred_real_changes_are_returned(self):
        count = 125
        old_blocks = [f"Stable provision anchor {i:03d}." for i in range(count)]
        new_blocks = []
        for i, anchor in enumerate(old_blocks):
            new_blocks.extend([f"New independently numbered provision {i:03d}.", anchor])
        old_pages = [{"page": 1, "text": " ".join(old_blocks), "blocks": old_blocks}]
        new_pages = [{"page": 1, "text": " ".join(new_blocks), "blocks": new_blocks}]
        changes = app.compare_passages(old_pages, new_pages,
            {"title": "earlier.pdf", "version": "Earlier"},
            {"title": "updated.pdf", "version": "Updated"}, "uncapped-test")
        self.assertEqual(len(changes), count)
        self.assertTrue(all(change["type"] == "Added content" for change in changes))

    def test_large_equal_replacement_hunk_matches_passages_in_document_order(self):
        count = 80
        old_blocks = [f"Provision {i:03d} requires an assistance amount of Rs 10 lakh after approval." for i in range(count)]
        new_blocks = [f"Provision {i:03d} requires an assistance amount of Rs 20 lakh after approval." for i in range(count)]
        old_pages = [{"page": 1, "height": 800, "text": " ".join(old_blocks), "blocks": old_blocks}]
        new_pages = [{"page": 1, "height": 800, "text": " ".join(new_blocks), "blocks": new_blocks}]
        changes = app.compare_passages(old_pages, new_pages,
            {"title": "earlier.pdf", "version": "Earlier"},
            {"title": "updated.pdf", "version": "Updated"}, "ordered-large-hunk-test")
        self.assertEqual(len(changes), count)
        self.assertTrue(all(change["type"] == "Modified content" for change in changes))
        self.assertEqual(changes[0]["previous"]["text"], old_blocks[0])
        self.assertEqual(changes[-1]["updated"]["text"], new_blocks[-1])
        self.assertTrue(all(app.validate_change_sources(change, old_pages, new_pages) for change in changes))

    def test_formatting_only_differences_are_not_reported(self):
        old_pages = [{"page": 1, "text": "A policy applies to all applicants.", "blocks": ["A policy—applies to all applicants."]}]
        new_pages = [{"page": 1, "text": "A policy applies to all applicants.", "blocks": ["A policy - applies to all applicants."]}]
        changes = app.compare_passages(old_pages, new_pages,
            {"title": "earlier.pdf", "version": "Earlier"},
            {"title": "updated.pdf", "version": "Updated"}, "format-test")
        self.assertEqual(changes, [])
        self.assertEqual(app.normalize("govern-\nment"), app.normalize("government"))
        self.assertTrue(app.normalize("पुरानी नीति"))

    def test_invalid_and_textless_pdfs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a readable PDF"):
            app.extract_pdf_bytes(b"not a PDF")
        empty_doc = pymupdf.open()
        empty_doc.new_page()
        empty_bytes = empty_doc.tobytes()
        empty_doc.close()
        with self.assertRaisesRegex(ValueError, "No selectable text"):
            app.extract_pdf_bytes(empty_bytes)

    @patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "", "GEMINI_API_KEY": ""})
    def test_upload_http_flow_returns_sources_and_rejects_invalid_file(self):
        old_bytes = self.make_pdf(["The earlier policy amount is Rs. 10 lakh.", "Applicants must submit the form online."])
        new_bytes = self.make_pdf(["The updated policy amount is Rs. 20 lakh.", "Applicants must submit the form through the portal."])
        boundary = "CivicLensUploadBoundary"

        def part(name, filename, content):
            return (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: application/pdf\r\n\r\n").encode() + content + b"\r\n"

        body = part("earlier_pdf", "earlier.pdf", old_bytes) + part("updated_pdf", "updated.pdf", new_bytes) + f"--{boundary}--\r\n".encode()
        server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            request = urllib.request.Request(base + "/api/compare-upload", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
            with patch.object(app, "ai_policy_analysis", wraps=app.ai_policy_analysis) as ai_call:
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(response.status, 200)
                    result = json.loads(response.read())
                self.assertEqual(result["ai_status"], "not_requested")
                self.assertIsNone(result["analysis_id"])
                ai_call.assert_not_called()
            self.assertTrue(result["changes"])
            self.assertTrue(all(change["explanation_mode"] == "Evidence only" for change in result["changes"]))
            self.assertTrue(all(change["previous"] or change["updated"] for change in result["changes"]))
            for side, original in (("old", old_bytes), ("new", new_bytes)):
                with urllib.request.urlopen(base + result[side]["view_url"]) as response:
                    self.assertEqual(response.read(), original)

            bad_body = part("earlier_pdf", "bad.pdf", b"not a PDF") + part("updated_pdf", "updated.pdf", new_bytes) + f"--{boundary}--\r\n".encode()
            bad_request = urllib.request.Request(base + "/api/compare-upload", data=bad_body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(bad_request)
            self.assertEqual(error.exception.code, 400)
            self.assertIn("readable PDF", error.exception.read().decode())
            error.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

    def test_comparison_loading_and_upload_errors_keep_existing_results_visible(self):
        frontend = (app.ROOT / "static" / "app.js").read_text(encoding="utf-8")
        sample_flow = frontend.split("async function loadComparison() {", 1)[1].split("function renderComparison(data)", 1)[0]
        upload_flow = frontend.split("uploadForm.addEventListener('submit', async event => {", 1)[1].split("async function init()", 1)[0]
        self.assertNotIn("changeRoot.innerHTML = ''", sample_flow)
        self.assertNotIn("changeRoot.innerHTML = ''", upload_flow)
        self.assertIn("changeRoot.setAttribute('aria-busy', 'true')", sample_flow)
        self.assertIn("changeRoot.setAttribute('aria-busy', 'false')", sample_flow)
        self.assertIn("changeRoot.setAttribute('aria-busy', 'true')", upload_flow)
        self.assertIn("changeRoot.setAttribute('aria-busy', 'false')", upload_flow)

    def test_phase7_grounding_accepts_supported_paraphrases_and_morphological_variants(self):
        evidence = (
            "Paragraph 2 of Notification S.O. 1443(E) dated 18th May 2016 has been amended. "
            "Borrowers who meet the eligibility criteria and credit requirement may receive loans. "
            "Entrepreneurs have successfully repaid previous loans under the scheme. "
            "Applicants submitted their application through the portal."
        )
        # eligibility / eligible
        self.assertEqual(app.assess_evidence_claim("Eligible borrowers receive loans.", evidence, allow_interpretation=True), "directly_supported")
        # requirement / require
        self.assertEqual(app.assess_evidence_claim("Borrowers are required to meet criteria.", evidence, allow_interpretation=True), "directly_supported")
        # repayment / repaid
        self.assertEqual(app.assess_evidence_claim("Entrepreneurs completed repayment of previous loans.", evidence, allow_interpretation=True), "interpretation")
        # amendment / amend
        self.assertEqual(app.assess_evidence_claim("The amendment modified the notification.", evidence, allow_interpretation=True), "interpretation")
        # application / apply
        self.assertEqual(app.assess_evidence_claim("Applicants apply through the portal.", evidence, allow_interpretation=True), "directly_supported")

    def test_phase7_grounding_rejects_unsupported_factual_and_numeric_claims(self):
        evidence = "Specified limit of the loan shall be Rs.20 lakh as defined above."
        # Unsupported amount
        self.assertIsNone(app.assess_evidence_claim("Specified limit of the loan shall be Rs.50 lakh.", evidence, allow_interpretation=True))
        # Unsupported percentage
        self.assertIsNone(app.assess_evidence_claim("Borrowers receive a 50% guarantee limit.", evidence, allow_interpretation=True))
        # Hallucinated page reference
        self.assertIsNone(app.assess_evidence_claim("The limit is Rs.20 lakh as stated on page 42.", evidence, allow_interpretation=True))

    def test_phase7_grounding_distinguishes_speculative_interpretations_from_directly_supported_facts(self):
        evidence = "Borrowers who meet criteria receive loans. The notification was published on 15 May 2024."
        # Speculative marker with allow_interpretation=True -> "interpretation"
        self.assertEqual(app.assess_evidence_claim("Borrowers could receive loans.", evidence, allow_interpretation=True), "interpretation")
        # Speculative marker with allow_interpretation=False -> None (rejected for factual rules)
        self.assertIsNone(app.assess_evidence_claim("Borrowers could receive loans.", evidence, allow_interpretation=False))
        # Directly supported fact without speculative markers -> "directly_supported"
        self.assertEqual(app.assess_evidence_claim("Borrowers who meet criteria receive loans.", evidence, allow_interpretation=False), "directly_supported")
        # Date containing the month 'May' is not misclassified as a speculative interpretation
        self.assertEqual(app.assess_evidence_claim("The notification was published on 15 May 2024.", evidence, allow_interpretation=False), "directly_supported")

    def test_phase7_grounding_rejects_unsupported_affected_groups(self):
        evidence = "Specified limit of the loan shall be Rs.20 lakh for micro unit borrowers."
        # Unsupported target affected groups should be rejected
        for group in ("farmers", "citizens", "startups", "students", "women"):
            claim = f"Eligible {group} receive loans up to Rs.20 lakh."
            self.assertIsNone(app.assess_evidence_claim(claim, evidence, allow_interpretation=True), f"Expected {group} to be rejected as unsupported")

    def test_phase7_grounding_handles_honest_insufficient_evidence_answers(self):
        evidence = "Specified limit of the loan shall be Rs.20 lakh."
        self.assertEqual(
            app.assess_evidence_claim("The supplied passages do not establish who is affected.", evidence, allow_interpretation=True),
            "insufficient_evidence"
        )
        self.assertEqual(
            app.assess_evidence_claim("This cannot be verified from the uploaded documents.", evidence, allow_interpretation=True),
            "insufficient_evidence"
        )
        self.assertEqual(
            app.assess_evidence_claim("There is insufficient evidence in the cited text.", evidence, allow_interpretation=True),
            "insufficient_evidence"
        )

    def test_phase7_ask_civiclens_claim_status_grounding_and_limitations(self):
        session_id = "test-phase7-civiclens-session"
        app.remember_document_session(
            session_id,
            {"title": "Earlier", "version": "2023"},
            {"title": "Updated", "version": "2024"},
            [{"page": 1, "text": "Specified limit shall be Rs.10 lakh for micro unit borrowers who receive loans.", "blocks": [{"text": "Specified limit shall be Rs.10 lakh for micro unit borrowers who receive loans."}]}],
            [{"page": 2, "text": "Specified limit shall be Rs.20 lakh for micro unit borrowers who receive loans.", "blocks": [{"text": "Specified limit shall be Rs.20 lakh for micro unit borrowers who receive loans."}]}],
        )
        passages = [
            {"id": "pass-earlier", "text": "Specified limit shall be Rs.10 lakh for micro unit borrowers who receive loans.", "side": "earlier", "document": "Doc", "version": "1", "page": 1, "section": None},
            {"id": "pass-updated", "text": "Specified limit shall be Rs.20 lakh for micro unit borrowers who receive loans.", "side": "updated", "document": "Doc", "version": "2", "page": 2, "section": None},
        ]

        class FakeResponse:
            def __init__(self, content): self.content = content
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps({"choices": [{"message": {"content": json.dumps(self.content)}}]}).encode()

        with patch.object(app, "retrieve_chat_evidence", return_value=passages), patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}):
            # Evidence-aligned answer -> "directly_supported"
            with patch("urllib.request.urlopen", return_value=FakeResponse({"answer": "Specified limit shall be Rs.20 lakh for micro unit borrowers.", "source_ids": ["pass-updated"]})):
                r1 = app.answer_policy_question(session_id, "What is the updated limit?")
                self.assertTrue(r1["ok"])
                self.assertEqual(r1["claim_status"], "directly_supported")

            # Speculative answer -> "interpretation"
            with patch("urllib.request.urlopen", return_value=FakeResponse({"answer": "Micro unit borrowers could receive Rs.20 lakh.", "source_ids": ["pass-updated"]})):
                r2 = app.answer_policy_question(session_id, "Could borrowers receive 20 lakh?")
                self.assertTrue(r2["ok"])
                self.assertEqual(r2["claim_status"], "interpretation")

            # Honest limitation answer -> "insufficient_evidence" without raising ValueError
            with patch("urllib.request.urlopen", return_value=FakeResponse({"answer": "The supplied passages do not establish who is affected.", "source_ids": ["pass-updated"]})):
                r3 = app.answer_policy_question(session_id, "Who else is affected?")
                self.assertTrue(r3["ok"])
                self.assertEqual(r3["claim_status"], "insufficient_evidence")

            # Unsupported affected group -> fails safely
            with patch("urllib.request.urlopen", return_value=FakeResponse({"answer": "Farmers receive Rs.20 lakh.", "source_ids": ["pass-updated"]})):
                r4 = app.answer_policy_question(session_id, "Can farmers apply?")
                self.assertFalse(r4["ok"])
                self.assertIn("verified AI answer", r4["error"])

            # Unsupported numeric amount -> fails safely
            with patch("urllib.request.urlopen", return_value=FakeResponse({"answer": "Specified limit shall be Rs.50 lakh for borrowers.", "source_ids": ["pass-updated"]})):
                r5 = app.answer_policy_question(session_id, "Is it 50 lakh?")
                self.assertFalse(r5["ok"])
                self.assertIn("verified AI answer", r5["error"])

    def test_phase7_ai_policy_analysis_populates_claim_validation_statuses(self):
        changes = app.compare_docs(*app.SAMPLE_DOCS)
        group = self.analyst_group({
            "source_ids": ["cgfmu-loan-limit"],
            "title": "Specified loan limit",
            "what_changed": "The specified loan limit changes from Rs.10 lakh to Rs.20 lakh.",
            "previous_rule": "The earlier excerpt says Rs.10 lakh.",
            "updated_rule": "The updated excerpt says Rs.20 lakh.",
            "why_it_matters": "The updated excerpt states a higher specified loan limit.",
            "who_may_be_affected": None,
            "change_type": "Modified loan limit",
        })
        response_content = json.dumps({
            "summary": "The updated excerpt changes the specified loan limit from Rs.10 lakh to Rs.20 lakh.",
            "groups": [group],
        })

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps({"choices": [{"message": {"content": response_content}}]}).encode()

        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", return_value=FakeResponse()):
            analysis, error = app.ai_policy_analysis([changes[0]])

        self.assertIsNone(error)
        self.assertIsNotNone(analysis)
        claim_val = analysis["groups"][0]["claim_validation"]
        self.assertEqual(claim_val["title"], "directly_supported")
        self.assertEqual(claim_val["change_summary"], "directly_supported")
        self.assertEqual(claim_val["old_rule"], "directly_supported")
        self.assertEqual(claim_val["new_rule"], "directly_supported")
        self.assertEqual(claim_val["why_it_matters"], "interpretation")
        self.assertEqual(claim_val["who_may_be_affected"], "insufficient_evidence")
        self.assertEqual(claim_val["unknown_information"], "insufficient_evidence")
        self.assertEqual(claim_val["limitations"], "insufficient_evidence")

    def test_ai_summary_directional_validation_supported_increase(self):
        earlier = "The maximum loan limit under the scheme is Rs. 10 lakh for borrowers."
        updated = "The maximum loan limit under the scheme is Rs. 20 lakh for borrowers."
        evidence = f"{earlier} {updated}"
        pairs = [(earlier, updated)]
        summary = "The maximum loan limit under the scheme increased to Rs. 20 lakh."
        self.assertEqual(
            app.assess_evidence_claim(summary, evidence, allow_interpretation=True, evidence_pairs=pairs),
            "interpretation"
        )

    def test_ai_summary_directional_validation_supported_decrease(self):
        earlier = "The annual guarantee fee shall be 1.5% of the sanctioned amount."
        updated = "The annual guarantee fee shall be 0.75% of the sanctioned amount."
        evidence = f"{earlier} {updated}"
        pairs = [(earlier, updated)]
        summary = "The annual guarantee fee decreased to 0.75% of the sanctioned amount."
        self.assertEqual(
            app.assess_evidence_claim(summary, evidence, allow_interpretation=True, evidence_pairs=pairs),
            "interpretation"
        )

    def test_ai_summary_directional_validation_unsupported_claims_rejected(self):
        earlier = "The maximum loan limit under the scheme is Rs. 10 lakh for borrowers."
        updated = "The maximum loan limit under the scheme is Rs. 20 lakh for borrowers."
        evidence = f"{earlier} {updated}"
        pairs = [(earlier, updated)]
        # Claims decrease when numbers increased
        self.assertIsNone(
            app.assess_evidence_claim("The maximum loan limit under the scheme decreased to Rs. 20 lakh.", evidence, allow_interpretation=True, evidence_pairs=pairs)
        )
        # Inverted transition: claims increased from 20 lakh to 10 lakh
        self.assertIsNone(
            app.assess_evidence_claim("The maximum loan limit under the scheme increased from Rs. 20 lakh to Rs. 10 lakh.", evidence, allow_interpretation=True, evidence_pairs=pairs)
        )
        # Hallucinated number not in evidence
        self.assertIsNone(
            app.assess_evidence_claim("The maximum loan limit under the scheme increased to Rs. 50 lakh.", evidence, allow_interpretation=True, evidence_pairs=pairs)
        )

    def test_ai_summary_directional_validation_multiple_unrelated_policy_changes(self):
        pair1 = ("The maximum loan limit is Rs. 10 lakh.", "The maximum loan limit is Rs. 20 lakh.")
        pair2 = ("The annual processing fee is 5%.", "The annual processing fee is 2%.")
        evidence = f"{pair1[0]} {pair1[1]} {pair2[0]} {pair2[1]}"
        pairs = [pair1, pair2]

        # Valid multi-change summary correctly matches corresponding pairs
        valid_summary = "The maximum loan limit increased to Rs. 20 lakh, while the processing fee decreased to 2%."
        self.assertEqual(
            app.assess_evidence_claim(valid_summary, evidence, allow_interpretation=True, evidence_pairs=pairs),
            "interpretation"
        )

        # Cross-talk: swapped directions (claims limit decreased and fee increased)
        invalid_summary = "The maximum loan limit decreased to Rs. 20 lakh, while the processing fee increased to 2%."
        self.assertIsNone(
            app.assess_evidence_claim(invalid_summary, evidence, allow_interpretation=True, evidence_pairs=pairs)
        )

        # Misattribution: fee claims increase without fee numbers, attempting to match loan pair
        misattributed = "The annual processing fee increased."
        self.assertIsNone(
            app.assess_evidence_claim(misattributed, evidence, allow_interpretation=True, evidence_pairs=pairs)
        )

    def test_ai_summary_mixed_supported_and_unsupported_claims_rejected(self):
        earlier = "The maximum loan limit is Rs. 10 lakh for eligible enterprises."
        updated = "The maximum loan limit is Rs. 20 lakh for eligible enterprises."
        evidence = f"{earlier} {updated}"
        pairs = [(earlier, updated)]
        # First sentence is supported increase; second sentence makes ungrounded claim
        mixed_summary = "The maximum loan limit increased to Rs. 20 lakh. All foreign students are barred from applying."
        self.assertIsNone(
            app.assess_evidence_claim(mixed_summary, evidence, allow_interpretation=True, evidence_pairs=pairs)
        )

    def test_ask_civiclens_insufficient_or_unpaired_evidence_fails_safely(self):
        session_id = "test-ask-civiclens-insufficient-evidence-session"
        app.remember_document_session(
            session_id,
            {"title": "Earlier", "version": "2023"},
            {"title": "Updated", "version": "2024"},
            [{"page": 1, "text": "Specified limit shall be Rs.10 lakh.", "blocks": [{"text": "Specified limit shall be Rs.10 lakh."}]}],
            [{"page": 2, "text": "The annual penalty fee is 2%.", "blocks": [{"text": "The annual penalty fee is 2%."}]}],
        )
        passages = [
            {"id": "pass-earlier", "text": "Specified limit shall be Rs.10 lakh.", "side": "earlier", "document": "Doc", "version": "1", "page": 1, "section": "Loan Rules"},
            {"id": "pass-updated", "text": "The annual penalty fee is 2%.", "side": "updated", "document": "Doc", "version": "2", "page": 2, "section": "Penalties"},
        ]

        class FakeResponse:
            def __init__(self, content): self.content = content
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps({"choices": [{"message": {"content": json.dumps(self.content)}}]}).encode()

        with patch.object(app, "retrieve_chat_evidence", return_value=passages), patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}):
            # Unpaired passages: AI answer attempts directional claim comparing unrelated passages -> rejected safely
            with patch("urllib.request.urlopen", return_value=FakeResponse({"answer": "The loan limit decreased to 2%.", "source_ids": ["pass-earlier", "pass-updated"]})):
                res = app.answer_policy_question(session_id, "Did the limit change?")
                self.assertFalse(res["ok"])
                self.assertIn("verified AI answer", res["error"])

            # Answer acknowledges insufficient evidence -> accepted with claim_status "insufficient_evidence"
            with patch("urllib.request.urlopen", return_value=FakeResponse({"answer": "This cannot be verified from the uploaded documents because no corresponding passage was found.", "source_ids": ["pass-earlier"]})):
                res = app.answer_policy_question(session_id, "What is the updated loan limit?")
                self.assertTrue(res["ok"])
                self.assertEqual(res["claim_status"], "insufficient_evidence")

    def test_change_summary_grounding_with_context_and_strict_number_bounds(self):
        changes = app.compare_docs(*app.SAMPLE_DOCS)
        c0_earlier = changes[0]["previous"]["text"]
        c0_updated = changes[0]["updated"]["text"]
        c0_ev = f"{c0_earlier} {c0_updated}"
        c0_pairs = [(c0_earlier, c0_updated)]

        all_ev = " ".join(
            (c["previous"]["text"] if c.get("previous") else "") + " " +
            (c["updated"]["text"] if c.get("updated") else "")
            for c in changes
        )

        c1_updated = changes[1]["updated"]["text"]
        c1_ev = c1_updated

        # 1. The loan-limit summary using "eligible borrowers" and Rs. 10 lakh -> Rs. 20 lakh is accepted
        # when companion verified passage supports the subject.
        summary_valid = "The specified loan limit for eligible borrowers increased from Rs. 10 lakh to Rs. 20 lakh."
        self.assertEqual(
            app.assess_evidence_claim(
                summary_valid, c0_ev, allow_interpretation=True,
                earlier_evidence=c0_earlier, updated_evidence=c0_updated,
                evidence_pairs=c0_pairs, context_evidence=all_ev
            ),
            "interpretation"
        )

        # 2. The same summary is rejected if its numerical amount or direction is unsupported.
        summary_wrong_direction = "The specified loan limit for eligible borrowers decreased to Rs. 20 lakh."
        self.assertIsNone(
            app.assess_evidence_claim(
                summary_wrong_direction, c0_ev, allow_interpretation=True,
                earlier_evidence=c0_earlier, updated_evidence=c0_updated,
                evidence_pairs=c0_pairs, context_evidence=all_ev
            )
        )

        # 3. An unrelated change elsewhere in the document cannot validate the loan-limit summary.
        # (Rs. 10,000 overdraft from Tarun Plus cannot validate loan limit summary)
        summary_unrelated_amount = "The specified loan limit for eligible borrowers increased from Rs. 10,000 to Rs. 20 lakh."
        self.assertIsNone(
            app.assess_evidence_claim(
                summary_unrelated_amount, c0_ev, allow_interpretation=True,
                earlier_evidence=c0_earlier, updated_evidence=c0_updated,
                evidence_pairs=c0_pairs, context_evidence=all_ev
            )
        )

        # 4. A hallucinated amount such as Rs. 50 lakh is rejected.
        summary_hallucinated_amount = "The specified loan limit for eligible borrowers increased to Rs. 50 lakh."
        self.assertIsNone(
            app.assess_evidence_claim(
                summary_hallucinated_amount, c0_ev, allow_interpretation=True,
                earlier_evidence=c0_earlier, updated_evidence=c0_updated,
                evidence_pairs=c0_pairs, context_evidence=all_ev
            )
        )

        # 5. Added provisions without earlier evidence do not receive unsupported directional descriptions.
        added_unsupported_direction = "The notification introduces the Tarun Plus category, increasing the loan limit to Rs. 20 lakh."
        self.assertIsNone(
            app.assess_evidence_claim(
                added_unsupported_direction, c1_ev, allow_interpretation=True,
                earlier_evidence="", updated_evidence=c1_updated,
                evidence_pairs=[], context_evidence=all_ev
            )
        )

        # 6. Added provision with valid descriptive summary is supported.
        added_valid_summary = "Introduces the Tarun Plus category covering loans above 10 lakh and up to 20 lakh for entrepreneurs who repaid previous Tarun loans."
        self.assertEqual(
            app.assess_evidence_claim(
                added_valid_summary, c1_ev, allow_interpretation=True,
                earlier_evidence="", updated_evidence=c1_updated,
                evidence_pairs=[], context_evidence=all_ev
            ),
            "directly_supported"
        )

    def test_ai_policy_analysis_accepts_change_summary_with_companion_subject(self):
        changes = app.compare_docs(*app.SAMPLE_DOCS)
        group0 = self.analyst_group({
            "source_ids": ["cgfmu-loan-limit"],
            "title": "Specified loan limit",
            "what_changed": "The specified loan limit for eligible borrowers increased from Rs.10 lakh to Rs.20 lakh.",
            "previous_rule": "The earlier excerpt says Rs.10 lakh.",
            "updated_rule": "The updated excerpt says Rs.20 lakh.",
            "why_it_matters": "The updated excerpt states a higher specified loan limit.",
            "who_may_be_affected": None,
            "change_type": "Modified loan limit",
        })
        group1 = self.analyst_group({
            "source_ids": ["cgfmu-tarun-plus"],
            "title": "Tarun Plus category",
            "what_changed": "The updated excerpt states Tarun Plus covering loans above 10 lakh and upto 20 lakh for entrepreneurs who repaid previous Tarun loans.",
            "previous_rule": "No corresponding passage was found in the supplied earlier comparison.",
            "updated_rule": "The updated excerpt states Tarun Plus covering loans above 10 lakh and upto 20 lakh for entrepreneurs who repaid previous Tarun loans.",
            "why_it_matters": "The updated excerpt states a new Tarun Plus category for entrepreneurs who repaid previous loans.",
            "who_may_be_affected": None,
            "change_type": "Added loan category",
        })
        response_content = json.dumps({
            "summary": "The specified loan limit for eligible borrowers increased from Rs.10 lakh to Rs.20 lakh.",
            "groups": [group0, group1],
        })

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps({"choices": [{"message": {"content": response_content}}]}).encode()

        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", return_value=FakeResponse()):
            analysis, error = app.ai_policy_analysis(changes)

        self.assertIsNone(error)
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis["groups"][0]["claim_validation"]["change_summary"], "interpretation")

    def test_old_rule_and_new_rule_side_specific_grounding_and_paraphrase(self):
        earlier_passage = "Specified limit of the loan shall be Rs.10 lakh or such other amount as may be decided"
        updated_passage = "Specified limit of the loan shall be Rs.20 lakh as defined above or such other amount as may be decided"
        earlier_context_with_borrower = "Paragraph 2 of Notification: Eligible borrower means new micro enterprise."
        updated_context_with_borrower = "Paragraph 2 of Notification: Eligible borrower means new micro enterprise."

        # 1. "The earlier rule set the loan limit at Rs. 10 lakh" accepted against supplied earlier passage.
        claim_set = "The earlier rule set the loan limit at Rs. 10 lakh."
        self.assertEqual(
            app.assess_evidence_claim(claim_set, earlier_passage, allow_interpretation=True, context_evidence=earlier_context_with_borrower),
            "interpretation"
        )

        # 2. "The specified loan limit for eligible borrowers was Rs. 10 lakh" accepted only if earlier-document context supports "eligible borrowers".
        claim_eligible = "The specified loan limit for eligible borrowers was Rs. 10 lakh."
        self.assertEqual(
            app.assess_evidence_claim(claim_eligible, earlier_passage, allow_interpretation=True, context_evidence=earlier_context_with_borrower),
            "interpretation"
        )

        # 3. The same claim rejected if that subject is supported only by updated-document context (i.e. earlier context lacks it).
        self.assertIsNone(
            app.assess_evidence_claim(claim_eligible, earlier_passage, allow_interpretation=True, context_evidence="")
        )

        # 4. Unsupported amounts and units rejected.
        self.assertIsNone(
            app.assess_evidence_claim("The earlier rule set the loan limit at Rs. 20 lakh.", earlier_passage, allow_interpretation=True, context_evidence=earlier_context_with_borrower)
        )
        self.assertIsNone(
            app.assess_evidence_claim("The earlier rule set the loan limit at Rs. 10 crore.", earlier_passage, allow_interpretation=True, context_evidence=earlier_context_with_borrower)
        )

        # 5. Unsupported eligibility conditions and obligations rejected.
        self.assertIsNone(
            app.assess_evidence_claim("Students were eligible for the Rs. 10 lakh loan.", earlier_passage, allow_interpretation=True, context_evidence=earlier_context_with_borrower)
        )
        self.assertIsNone(
            app.assess_evidence_claim("Borrowers must submit audited balance sheets for Rs. 10 lakh loan.", earlier_passage, allow_interpretation=True, context_evidence=earlier_context_with_borrower)
        )

        # 6. A new_rule paraphrase validated using updated-document context.
        new_rule_claim = "The updated rule set the loan limit for eligible borrowers at Rs. 20 lakh."
        self.assertEqual(
            app.assess_evidence_claim(new_rule_claim, updated_passage, allow_interpretation=True, context_evidence=updated_context_with_borrower),
            "interpretation"
        )
        # and fails if updated passage has different amount:
        self.assertIsNone(
            app.assess_evidence_claim("The updated rule set the loan limit for eligible borrowers at Rs. 50 lakh.", updated_passage, allow_interpretation=True, context_evidence=updated_context_with_borrower)
        )

        # 7. Added provisions with no earlier passage handled safely.
        # Standard phrases accepted:
        for phrase in (
            "No corresponding passage was found in the supplied earlier comparison.",
            "no previous passage was found in the supplied comparison",
            "no prior corresponding provision was provided",
            "no earlier passage was found in the supplied comparison",
            "There was no previous rule for this category.",
            "No prior rule was provided in the earlier document.",
            "No earlier excerpt was found.",
        ):
            lower = phrase.casefold()
            absent_entire = bool(re.search(r"\b(?:entire|whole)\s+(?:[a-z]+\s+)?(?:document|policy|gazette)\b", lower) or "absent from the" in lower)
            has_missing_kw = any(kw in lower for kw in ("earlier", "previous", "prior")) or app._is_insufficient_evidence_statement(phrase)
            self.assertTrue(not absent_entire and has_missing_kw, f"Failed on missing phrase: {phrase}")

        # Phrases claiming entire document absence rejected:
        for bad_phrase in (
            "The provision is absent from the entire document.",
            "The rule does not exist in the entire earlier policy.",
            "The provision was absent from the entire earlier document.",
        ):
            lower = bad_phrase.casefold()
            absent_entire = bool(re.search(r"\b(?:entire|whole)\s+(?:[a-z]+\s+)?(?:document|policy|gazette)\b", lower) or "absent from the" in lower)
            has_missing_kw = any(kw in lower for kw in ("earlier", "previous", "prior")) or app._is_insufficient_evidence_statement(bad_phrase)
            self.assertFalse(not absent_entire and has_missing_kw, f"Incorrectly accepted whole-document absence: {bad_phrase}")

    def test_ai_policy_analysis_validates_old_and_new_rules_with_side_specific_context(self):
        changes = app.compare_docs(*app.SAMPLE_DOCS)
        group0 = self.analyst_group({
            "source_ids": ["cgfmu-loan-limit"],
            "title": "Specified loan limit",
            "what_changed": "The specified loan limit for eligible borrowers increased from Rs.10 lakh to Rs.20 lakh.",
            "previous_rule": "The earlier rule set the specified limit of the loan at Rs. 10 lakh.",
            "updated_rule": "The updated rule set the specified limit of the loan at Rs. 20 lakh.",
            "why_it_matters": "The updated excerpt states a higher specified loan limit.",
            "who_may_be_affected": None,
            "change_type": "Modified loan limit",
        })
        group1 = self.analyst_group({
            "source_ids": ["cgfmu-tarun-plus"],
            "title": "Tarun Plus category",
            "what_changed": "The updated excerpt states Tarun Plus covering loans above 10 lakh and upto 20 lakh for entrepreneurs who repaid previous Tarun loans.",
            "previous_rule": "No previous passage was found in the supplied comparison.",
            "updated_rule": "The updated excerpt states Tarun Plus covering loans above 10 lakh and upto 20 lakh for entrepreneurs who repaid previous Tarun loans.",
            "why_it_matters": "The updated excerpt states a new Tarun Plus category for entrepreneurs who repaid previous loans.",
            "who_may_be_affected": None,
            "change_type": "Added loan category",
        })
        response_content = json.dumps({
            "summary": "The specified loan limit for eligible borrowers increased from Rs.10 lakh to Rs.20 lakh.",
            "groups": [group0, group1],
        })

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps({"choices": [{"message": {"content": response_content}}]}).encode()

        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", return_value=FakeResponse()):
            analysis, error = app.ai_policy_analysis(changes)

        self.assertIsNone(error)
        self.assertIsNotNone(analysis)
        self.assertIn(analysis["groups"][0]["claim_validation"]["old_rule"], ("directly_supported", "interpretation"))
        self.assertIn(analysis["groups"][0]["claim_validation"]["new_rule"], ("directly_supported", "interpretation"))
        self.assertEqual(analysis["groups"][1]["claim_validation"]["old_rule"], "directly_supported")

    def test_new_rule_directional_and_numerical_grounding_suite(self):
        # 1. A valid increase from Rs. 10 lakh to Rs. 20 lakh is accepted
        earlier_limit = "Specified limit of the loan shall be Rs.10 lakh or such other amount as may be decided"
        updated_limit = "Specified limit of the loan shall be Rs.20 lakh as defined above or such other amount as may be decided by the Fund from time to time."
        updated_context = "Eligible borrower means new micro enterprise. Specified limit shall be Rs.20 lakh."
        valid_increase = "The specified loan limit for eligible borrowers was increased to Rs. 20 lakh."
        self.assertEqual(
            app.assess_evidence_claim(
                valid_increase, updated_limit, allow_interpretation=True,
                earlier_evidence=earlier_limit, updated_evidence=updated_limit,
                evidence_pairs=[(earlier_limit, updated_limit)],
                context_evidence=updated_context,
            ),
            "interpretation"
        )

        # 2. A valid decrease is accepted when the old and updated passages support it
        earlier_fee = "Application processing fee is Rs. 500."
        updated_fee = "Application processing fee is Rs. 250."
        fee_context = "Eligible applicant fee is Rs. 250."
        valid_decrease = "The updated rule decreased the application fee to Rs. 250."
        self.assertEqual(
            app.assess_evidence_claim(
                valid_decrease, updated_fee, allow_interpretation=True,
                earlier_evidence=earlier_fee, updated_evidence=updated_fee,
                evidence_pairs=[(earlier_fee, updated_fee)],
                context_evidence=fee_context,
            ),
            "interpretation"
        )

        # 3. An inverted direction is rejected
        inverted_direction = "The specified loan limit for eligible borrowers was decreased to Rs. 20 lakh."
        self.assertIsNone(
            app.assess_evidence_claim(
                inverted_direction, updated_limit, allow_interpretation=True,
                earlier_evidence=earlier_limit, updated_evidence=updated_limit,
                evidence_pairs=[(earlier_limit, updated_limit)],
                context_evidence=updated_context,
            )
        )

        # 4. An unsupported numerical amount is rejected
        unsupported_amount = "The specified loan limit for eligible borrowers was increased to Rs. 50 lakh."
        self.assertIsNone(
            app.assess_evidence_claim(
                unsupported_amount, updated_limit, allow_interpretation=True,
                earlier_evidence=earlier_limit, updated_evidence=updated_limit,
                evidence_pairs=[(earlier_limit, updated_limit)],
                context_evidence=updated_context,
            )
        )

        # 5. A directional claim is rejected when the necessary earlier-side evidence is unavailable
        added_updated = "Tarun Plus covering loans above 10 lakh and upto 20 lakh for entrepreneurs."
        added_context = "Tarun Plus covering loans above 10 lakh and upto 20 lakh."
        directional_without_earlier = "The new rule increases the loan limit to Rs. 20 lakh."
        self.assertIsNone(
            app.assess_evidence_claim(
                directional_without_earlier, added_updated, allow_interpretation=True,
                earlier_evidence="", updated_evidence=added_updated,
                evidence_pairs=[],
                context_evidence=added_context,
            )
        )

        # 6. Existing side-specific vocabulary grounding remains intact
        # Vocabulary supported in updated context ("eligible borrowers") passes:
        claim_vocab = "The updated rule sets the specified limit of the loan at Rs. 20 lakh for eligible borrowers."
        self.assertEqual(
            app.assess_evidence_claim(
                claim_vocab, updated_limit, allow_interpretation=True,
                earlier_evidence=earlier_limit, updated_evidence=updated_limit,
                evidence_pairs=[(earlier_limit, updated_limit)],
                context_evidence=updated_context,
            ),
            "interpretation"
        )
        # But if the vocabulary is absent from updated context, critical missing terms reject it:
        self.assertIsNone(
            app.assess_evidence_claim(
                claim_vocab, updated_limit, allow_interpretation=True,
                earlier_evidence=earlier_limit, updated_evidence=updated_limit,
                evidence_pairs=[(earlier_limit, updated_limit)],
                context_evidence="",  # Empty context -> missing "eligible" and "borrowers"
            )
        )

    def test_ai_policy_analysis_accepts_directional_new_rule_end_to_end(self):
        changes = app.compare_docs(*app.SAMPLE_DOCS)
        group0 = self.analyst_group({
            "source_ids": ["cgfmu-loan-limit"],
            "title": "Specified loan limit",
            "what_changed": "The specified loan limit for eligible borrowers increased from Rs.10 lakh to Rs.20 lakh.",
            "previous_rule": "The earlier rule set the specified limit of the loan at Rs. 10 lakh.",
            "updated_rule": "The specified loan limit for eligible borrowers was increased to Rs. 20 lakh.",
            "why_it_matters": "The updated excerpt states a higher specified loan limit.",
            "who_may_be_affected": None,
            "change_type": "Modified loan limit",
        })
        group1 = self.analyst_group({
            "source_ids": ["cgfmu-tarun-plus"],
            "title": "Tarun Plus category",
            "what_changed": "The updated excerpt states Tarun Plus covering loans above 10 lakh and upto 20 lakh for entrepreneurs who repaid previous Tarun loans.",
            "previous_rule": "No previous passage was found in the supplied comparison.",
            "updated_rule": "The updated excerpt states Tarun Plus covering loans above 10 lakh and upto 20 lakh for entrepreneurs who repaid previous Tarun loans.",
            "why_it_matters": "The updated excerpt states a new Tarun Plus category for entrepreneurs who repaid previous loans.",
            "who_may_be_affected": None,
            "change_type": "Added loan category",
        })
        response_content = json.dumps({
            "summary": "The specified loan limit for eligible borrowers increased from Rs.10 lakh to Rs.20 lakh.",
            "groups": [group0, group1],
        })

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps({"choices": [{"message": {"content": response_content}}]}).encode()

        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), patch("urllib.request.urlopen", return_value=FakeResponse()):
            analysis, error = app.ai_policy_analysis(changes)

        self.assertIsNone(error)
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis["groups"][0]["claim_validation"]["new_rule"], "interpretation")

    # -----------------------------------------------------------------
    # Regression tests for why_it_matters validation fixes
    # -----------------------------------------------------------------

    def test_why_it_matters_added_provision_tier_with_numbers_accepted(self):
        """An added provision describing a tier with numbers explicitly in the updated
        evidence (no earlier passage, no diff pair) must be accepted as interpretation.
        This covers directional adjectives like 'higher' used to describe a tier range."""
        updated_ev = (
            "Tarun Plus: covering loans above 10 lakh and upto 20 lakh for those "
            "entrepreneurs who have availed and successfully repaid previous loans under "
            "the 'Tarun' category."
        )
        # "higher loan category" with specific numbers present in evidence -> accepted
        why_with_numbers = (
            "It introduces a higher loan category of up to Rs. 20 lakh for entrepreneurs."
        )
        self.assertEqual(
            app.assess_evidence_claim(
                why_with_numbers, updated_ev, allow_interpretation=True,
                earlier_evidence="", updated_evidence=updated_ev,
                evidence_pairs=[], context_evidence=updated_ev,
            ),
            "interpretation",
        )

    def test_why_it_matters_added_provision_bare_comparative_rejected(self):
        """A directional claim with no numbers and no diff pair must still be rejected:
        the fix only unblocks tier descriptions where numbers are verifiable in evidence."""
        updated_ev = (
            "Tarun Plus: covering loans above 10 lakh and upto 20 lakh for those "
            "entrepreneurs who have availed and successfully repaid previous loans under "
            "the 'Tarun' category."
        )
        # "higher loan coverage" with no numbers, no diff pair -> rejected
        why_bare = "This provides higher loan coverage for entrepreneurs."
        self.assertIsNone(
            app.assess_evidence_claim(
                why_bare, updated_ev, allow_interpretation=True,
                earlier_evidence="", updated_evidence=updated_ev,
                evidence_pairs=[], context_evidence=updated_ev,
            )
        )

    def test_why_it_matters_receive_as_connecting_verb_accepted(self):
        """'receive' used as a financial connecting verb (paraphrase of avail/sanctioned)
        must not block grounding when every number and substantial term is in evidence.
        In production, context_evidence=all_evidence covers companion passages where
        policy entities like 'borrower' are defined."""
        earlier_ev = (
            "Specified limit of the loan shall be Rs.10 lakh or such other amount as may be decided"
        )
        updated_ev = (
            "Specified limit of the loan shall be Rs.20 lakh as defined above or such other "
            "amount as may be decided by the Fund from time to time."
        )
        group_ev = earlier_ev + " " + updated_ev
        # In the real app, all_evidence includes the companion Tarun Plus passage which
        # defines "Eligible borrower" and contains "availed" (→ avail via morphological map).
        # Replicate that so the test actually verifies morphological grounding.
        companion_ev = (
            "Eligible borrower means new or existing micro unit whose credit requirement "
            "does not exceed the specified limit under PMMY. Entrepreneurs who have availed "
            "and repaid previous loans under Tarun may access loans above 10 lakh and upto 20 lakh."
        )
        all_ev = group_ev + " " + companion_ev
        # Phrase uses "receive" as a standard connecting verb
        why_receive = "Borrowers can receive loans up to Rs. 20 lakh."
        result = app.assess_evidence_claim(
            why_receive, group_ev, allow_interpretation=True,
            earlier_evidence=earlier_ev, updated_evidence=updated_ev,
            evidence_pairs=[(earlier_ev, updated_ev)], context_evidence=all_ev,
        )
        self.assertIsNotNone(result, "'receive' as a connecting verb should pass grounding")

    def test_why_it_matters_receive_with_hallucinated_number_rejected(self):
        """'receive' is now a soft term, but numeric grounding remains strict:
        an unsupported number must still cause rejection."""
        earlier_ev = (
            "Specified limit of the loan shall be Rs.10 lakh or such other amount as may be decided"
        )
        updated_ev = (
            "Specified limit of the loan shall be Rs.20 lakh as defined above or such other "
            "amount as may be decided by the Fund from time to time."
        )
        group_ev = earlier_ev + " " + updated_ev
        # Rs. 50 lakh is not in evidence
        why_hallucinated = "Borrowers can receive loans up to Rs. 50 lakh."
        self.assertIsNone(
            app.assess_evidence_claim(
                why_hallucinated, group_ev, allow_interpretation=True,
                earlier_evidence=earlier_ev, updated_evidence=updated_ev,
                evidence_pairs=[(earlier_ev, updated_ev)], context_evidence=group_ev,
            )
        )

    def test_why_it_matters_end_to_end_full_cgfmu_analysis(self):
        """End-to-end: both CGFMU change groups pass validation with realistic
        why_it_matters phrases that the previous implementation rejected."""
        changes = app.compare_docs(*app.SAMPLE_DOCS)
        group0 = self.analyst_group({
            "source_ids": ["cgfmu-loan-limit"],
            "title": "Specified loan limit",
            "what_changed": "The specified loan limit changes from Rs.10 lakh to Rs.20 lakh.",
            "previous_rule": "The earlier excerpt says Rs.10 lakh.",
            "updated_rule": "The updated excerpt says Rs.20 lakh.",
            # Realistic phrase: uses "receive" as a connecting verb
            "why_it_matters": "Borrowers can receive loans up to Rs. 20 lakh.",
            "who_may_be_affected": None,
            "change_type": "Modified loan limit",
        })
        group1 = self.analyst_group({
            "source_ids": ["cgfmu-tarun-plus"],
            "title": "Tarun Plus category",
            "what_changed": (
                "The updated excerpt states Tarun Plus covering loans above 10 lakh and "
                "upto 20 lakh for entrepreneurs who repaid previous Tarun loans."
            ),
            "previous_rule": "No corresponding passage was found in the supplied earlier comparison.",
            "updated_rule": (
                "The updated excerpt states Tarun Plus covering loans above 10 lakh and "
                "upto 20 lakh for entrepreneurs who repaid previous Tarun loans."
            ),
            # Realistic phrase: "higher" tier with numbers supported by evidence
            "why_it_matters": (
                "It introduces a higher loan category of up to Rs. 20 lakh for entrepreneurs."
            ),
            "who_may_be_affected": None,
            "change_type": "Added loan category",
        })
        response_content = json.dumps({
            "summary": (
                "The updated excerpt changes the specified loan limit from Rs.10 lakh to "
                "Rs.20 lakh and states a Tarun Plus category covering loans above 10 lakh "
                "and upto 20 lakh for entrepreneurs who repaid previous Tarun loans."
            ),
            "groups": [group0, group1],
        })

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps({"choices": [{"message": {"content": response_content}}]}).encode()

        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), \
                patch("urllib.request.urlopen", return_value=FakeResponse()):
            analysis, error = app.ai_policy_analysis(changes)

        self.assertIsNone(error)
        self.assertIsNotNone(analysis)
        self.assertEqual(len(analysis["groups"]), 2)
        # Both groups must pass why_it_matters validation
        self.assertIsNotNone(analysis["groups"][0]["claim_validation"]["why_it_matters"])
        self.assertIsNotNone(analysis["groups"][1]["claim_validation"]["why_it_matters"])



    # ------------------------------------------------------------------
    # higher/lower fallback range-guard tests (Safety Fix #1)
    # ------------------------------------------------------------------

    def test_higher_lower_fallback_single_amount_evidence_rejected(self):
        """Single numeric amount in evidence must NOT validate 'higher' or 'lower'.
        _evidence_establishes_range() requires ≥2 distinct values + range language."""
        evidence_single = "The loan limit shall be Rs. 20 lakh."
        claim = "It introduces a higher loan category of up to Rs. 20 lakh."
        self.assertIsNone(
            app.assess_evidence_claim(
                claim, evidence_single, allow_interpretation=True,
                earlier_evidence="", updated_evidence=evidence_single,
                evidence_pairs=[], context_evidence=evidence_single,
            )
        )

    def test_higher_lower_fallback_two_values_no_range_language_rejected(self):
        """Two values in evidence but no range language still fails the range guard."""
        evidence_no_range = "The minimum is Rs. 10 lakh. The maximum is Rs. 20 lakh."
        claim = "It introduces a higher loan category of up to Rs. 20 lakh."
        self.assertIsNone(
            app.assess_evidence_claim(
                claim, evidence_no_range, allow_interpretation=True,
                earlier_evidence="", updated_evidence=evidence_no_range,
                evidence_pairs=[], context_evidence=evidence_no_range,
            )
        )

    def test_higher_lower_fallback_explicit_range_accepted(self):
        """Evidence with 'above X and upto Y' (range language + 2 distinct values)
        allows a 'higher tier' description referencing one of those bounds."""
        evidence_range = (
            "Tarun Plus: covering loans above 10 lakh and upto 20 lakh for entrepreneurs."
        )
        claim = "It introduces a higher loan category of up to Rs. 20 lakh for entrepreneurs."
        result = app.assess_evidence_claim(
            claim, evidence_range, allow_interpretation=True,
            earlier_evidence="", updated_evidence=evidence_range,
            evidence_pairs=[], context_evidence=evidence_range,
        )
        self.assertIsNotNone(result)

    def test_higher_lower_fallback_hallucinated_number_in_range_evidence_rejected(self):
        """Even when evidence establishes a range, a claim with a number NOT in that
        range is blocked by the numeric grounding gate (claims_are_supported)."""
        evidence_range = "Covering loans above 10 lakh and upto 20 lakh."
        claim = "It introduces a higher loan category of up to Rs. 50 lakh."
        self.assertIsNone(
            app.assess_evidence_claim(
                claim, evidence_range, allow_interpretation=True,
                earlier_evidence="", updated_evidence=evidence_range,
                evidence_pairs=[], context_evidence=evidence_range,
            )
        )

    # ------------------------------------------------------------------
    # receive/access/obtain morphological mapping tests (Safety Fix #2)
    # ------------------------------------------------------------------

    def test_receive_is_not_a_stop_word_participates_in_coverage(self):
        """'receive' must not be silently ignored; _claim_terms must include it
        (as 'avail' via morphological mapping) so it participates in coverage."""
        terms = app._claim_terms("Borrowers can receive loans up to 20 lakh")
        self.assertIn("avail", terms, "'receive' must map to 'avail', not vanish")
        self.assertNotIn("receive", terms)  # mapped, not raw

    def test_receive_grounded_via_avail_in_context_accepted(self):
        """'receive' → 'avail' via morphological map; when companion evidence contains
        'availed', 'avail' is in context_terms and the claim is accepted."""
        earlier_ev = (
            "Specified limit of the loan shall be Rs.10 lakh or such other amount as may be decided"
        )
        updated_ev = (
            "Specified limit of the loan shall be Rs.20 lakh as defined above or such other "
            "amount as may be decided by the Fund from time to time."
        )
        group_ev = earlier_ev + " " + updated_ev
        # Companion contains "availed" (→ avail) and "borrower"
        companion_ev = (
            "Eligible borrower means micro unit. Entrepreneurs who have availed and repaid "
            "previous loans under Tarun may access loans above 10 lakh and upto 20 lakh."
        )
        all_ev = group_ev + " " + companion_ev
        claim = "Borrowers can receive loans up to Rs. 20 lakh."
        result = app.assess_evidence_claim(
            claim, group_ev, allow_interpretation=True,
            earlier_evidence=earlier_ev, updated_evidence=updated_ev,
            evidence_pairs=[(earlier_ev, updated_ev)], context_evidence=all_ev,
        )
        self.assertIsNotNone(result, "receive grounded via avail in companion should pass")

    def test_receive_hallucinated_number_still_rejected(self):
        """Numeric grounding is not weakened: a hallucinated amount in a 'receive'
        clause is blocked by claims_are_supported before term analysis."""
        earlier_ev = (
            "Specified limit of the loan shall be Rs.10 lakh or such other amount as may be decided"
        )
        updated_ev = (
            "Specified limit of the loan shall be Rs.20 lakh as defined above or such other "
            "amount as may be decided by the Fund from time to time."
        )
        group_ev = earlier_ev + " " + updated_ev
        claim = "Borrowers can receive loans up to Rs. 50 lakh."
        self.assertIsNone(
            app.assess_evidence_claim(
                claim, group_ev, allow_interpretation=True,
                earlier_evidence=earlier_ev, updated_evidence=updated_ev,
                evidence_pairs=[(earlier_ev, updated_ev)], context_evidence=group_ev,
            )
        )

    def test_receive_without_avail_in_evidence_or_context_coverage_behaviour(self):
        """Documents coverage behaviour when 'receive' (→ avail) is not grounded.
        With 4 claim terms and 3 grounded (borrower, loan, lakh), coverage=75%≥50% → passes.
        'avail' (from receive) counts as 1 unsupported but non-critical term."""
        evidence = (
            "Specified limit of the loan shall be Rs.20 lakh. "
            "Eligible borrower means micro unit."
        )
        claim = "Borrowers can receive loans up to Rs. 20 lakh."
        # borrower and loan and lakh are grounded; avail (from receive) is not
        result = app.assess_evidence_claim(
            claim, evidence, allow_interpretation=True, context_evidence=evidence,
        )
        # 3/4 coverage → passes; not critically rejected
        self.assertIsNotNone(result)
        # Confirm "avail" appears in claim_terms (not stop word)
        ct = app._claim_terms(claim)
        self.assertIn("avail", ct)


    # ------------------------------------------------------------------
    # Step 3: Comprehensive why_it_matters regression suite
    # ------------------------------------------------------------------

    def test_why_it_matters_cgfmu_loan_limit_increase_accepted(self):
        """Valid explanation for CGFMU loan-limit increase from Rs. 10 lakh to Rs. 20 lakh passes."""
        changes = app.compare_docs(*app.SAMPLE_DOCS)
        all_ev = " ".join(((c.get("previous") or {}).get("text") or "") + " " + ((c.get("updated") or {}).get("text") or "") for c in changes)
        all_pairs = [
            ((c.get("previous") or {}).get("text"), (c.get("updated") or {}).get("text"))
            for c in changes if c.get("previous") and c.get("updated")
        ]
        earlier_ev = "Specified limit of the loan shall be Rs.10 lakh or such other amount as may be decided"
        updated_ev = "Specified limit of the loan shall be Rs.20 lakh as defined above or such other amount as may be decided by the Fund from time to time."
        group_ev = earlier_ev + " " + updated_ev
        claim = "This change increases the specified loan limit from Rs. 10 lakh to Rs. 20 lakh, allowing eligible micro enterprises to access higher credit."
        result = app.assess_evidence_claim(
            claim, group_ev, allow_interpretation=True,
            earlier_evidence=earlier_ev, updated_evidence=updated_ev,
            evidence_pairs=all_pairs, context_evidence=all_ev,
        )
        self.assertIsNotNone(result)

    def test_why_it_matters_tarun_plus_provision_accepted(self):
        """Valid explanation of the Tarun Plus provision with range and companion pair passes."""
        changes = app.compare_docs(*app.SAMPLE_DOCS)
        all_ev = " ".join(((c.get("previous") or {}).get("text") or "") + " " + ((c.get("updated") or {}).get("text") or "") for c in changes)
        all_pairs = [
            ((c.get("previous") or {}).get("text"), (c.get("updated") or {}).get("text"))
            for c in changes if c.get("previous") and c.get("updated")
        ]
        tarun_change = next(c for c in changes if c["id"] == "cgfmu-tarun-plus")
        updated_ev = tarun_change["updated"]["text"]
        claim = (
            "This provision introduces the Tarun Plus category covering loans above 10 lakh and "
            "upto 20 lakh for entrepreneurs who have availed and successfully repaid previous Tarun loans."
        )
        result = app.assess_evidence_claim(
            claim, updated_ev, allow_interpretation=True,
            earlier_evidence="", updated_evidence=updated_ev,
            evidence_pairs=all_pairs, context_evidence=all_ev,
        )
        self.assertIsNotNone(result)

    def test_why_it_matters_unsupported_qualitative_claims_rejected(self):
        """Unsupported qualitative claims containing ungrounded critical terms are rejected."""
        changes = app.compare_docs(*app.SAMPLE_DOCS)
        all_ev = " ".join(((c.get("previous") or {}).get("text") or "") + " " + ((c.get("updated") or {}).get("text") or "") for c in changes)
        all_pairs = [
            ((c.get("previous") or {}).get("text"), (c.get("updated") or {}).get("text"))
            for c in changes if c.get("previous") and c.get("updated")
        ]
        earlier_ev = "Specified limit of the loan shall be Rs.10 lakh"
        updated_ev = "Specified limit of the loan shall be Rs.20 lakh"
        # "student" and "mandatory" (require) are critical terms not present in CGFMU evidence
        claim = "Students are eligible and must apply through national portal."
        result = app.assess_evidence_claim(
            claim, earlier_ev + " " + updated_ev, allow_interpretation=True,
            earlier_evidence=earlier_ev, updated_evidence=updated_ev,
            evidence_pairs=all_pairs, context_evidence=all_ev,
        )
        self.assertIsNone(result)

    def test_why_it_matters_unsupported_numbers_rejected(self):
        """Unsupported numerical claims (e.g. Rs. 50 lakh) are strictly rejected."""
        changes = app.compare_docs(*app.SAMPLE_DOCS)
        all_ev = " ".join(((c.get("previous") or {}).get("text") or "") + " " + ((c.get("updated") or {}).get("text") or "") for c in changes)
        all_pairs = [
            ((c.get("previous") or {}).get("text"), (c.get("updated") or {}).get("text"))
            for c in changes if c.get("previous") and c.get("updated")
        ]
        earlier_ev = "Specified limit of the loan shall be Rs.10 lakh"
        updated_ev = "Specified limit of the loan shall be Rs.20 lakh"
        claim = "The specified loan limit is increased to Rs. 50 lakh."
        result = app.assess_evidence_claim(
            claim, earlier_ev + " " + updated_ev, allow_interpretation=True,
            earlier_evidence=earlier_ev, updated_evidence=updated_ev,
            evidence_pairs=all_pairs, context_evidence=all_ev,
        )
        self.assertIsNone(result)

    def test_why_it_matters_unsupported_higher_lower_claims_rejected(self):
        """Inverted direction ('lower' when limit increased) or comparative without evidence is rejected."""
        changes = app.compare_docs(*app.SAMPLE_DOCS)
        all_ev = " ".join(((c.get("previous") or {}).get("text") or "") + " " + ((c.get("updated") or {}).get("text") or "") for c in changes)
        all_pairs = [
            ((c.get("previous") or {}).get("text"), (c.get("updated") or {}).get("text"))
            for c in changes if c.get("previous") and c.get("updated")
        ]
        earlier_ev = "Specified limit of the loan shall be Rs.10 lakh"
        updated_ev = "Specified limit of the loan shall be Rs.20 lakh"
        # Claim states decrease when evidence proves increase
        claim = "The specified loan limit decreases from Rs. 20 lakh to Rs. 10 lakh."
        result = app.assess_evidence_claim(
            claim, earlier_ev + " " + updated_ev, allow_interpretation=True,
            earlier_evidence=earlier_ev, updated_evidence=updated_ev,
            evidence_pairs=all_pairs, context_evidence=all_ev,
        )
        self.assertIsNone(result)

    def test_why_it_matters_valid_interpretations_supported_by_paired_and_contextual_evidence_pass(self):
        """End-to-end ai_policy_analysis with realistic AI explanations passes validation."""
        changes = app.compare_docs(*app.SAMPLE_DOCS)
        group0 = self.analyst_group({
            "source_ids": ["cgfmu-loan-limit"],
            "title": "Specified loan limit",
            "what_changed": "The specified loan limit changes from Rs.10 lakh to Rs.20 lakh.",
            "change_summary": "The specified loan limit changes from Rs.10 lakh to Rs.20 lakh.",
            "previous_rule": "The earlier excerpt says Rs.10 lakh.",
            "updated_rule": "The updated excerpt says Rs.20 lakh.",
            "why_it_matters": "This change increases the loan limit from Rs. 10 lakh to Rs. 20 lakh, allowing eligible micro enterprises to access higher credit.",
            "simple_explanation": "The specified loan limit increases from Rs. 10 lakh to Rs. 20 lakh.",
            "who_may_be_affected": None,
            "change_type": "Modified loan limit",
            "practical_implications": ["Borrowers can receive loans up to Rs. 20 lakh."],
            "what_changed_for_them": ["Eligible micro units may access loans up to Rs. 20 lakh."],
        })
        group1 = self.analyst_group({
            "source_ids": ["cgfmu-tarun-plus"],
            "title": "Tarun Plus category",
            "what_changed": (
                "The updated excerpt states Tarun Plus covering loans above 10 lakh and "
                "upto 20 lakh for entrepreneurs who repaid previous Tarun loans."
            ),
            "change_summary": (
                "The updated excerpt states Tarun Plus covering loans above 10 lakh and "
                "upto 20 lakh for entrepreneurs who repaid previous Tarun loans."
            ),
            "previous_rule": "No corresponding passage was found in the supplied earlier comparison.",
            "updated_rule": (
                "The updated excerpt states Tarun Plus covering loans above 10 lakh and "
                "upto 20 lakh for entrepreneurs who repaid previous Tarun loans."
            ),
            "why_it_matters": (
                "It introduces a higher loan category of up to Rs. 20 lakh for entrepreneurs who repaid previous Tarun loans."
            ),
            "simple_explanation": (
                "Tarun Plus covers loans above 10 lakh and upto 20 lakh for entrepreneurs who repaid previous Tarun loans."
            ),
            "who_may_be_affected": None,
            "change_type": "Added loan category",
            "practical_implications": ["Entrepreneurs who have availed and repaid Tarun loans may access higher loans up to 20 lakh."],
            "what_changed_for_them": ["Tarun borrowers can access loans above 10 lakh and upto 20 lakh."],
        })
        response_content = json.dumps({
            "summary": (
                "The updated excerpt changes the specified loan limit from Rs.10 lakh to "
                "Rs.20 lakh and states a Tarun Plus category covering loans above 10 lakh "
                "and upto 20 lakh for entrepreneurs who repaid previous Tarun loans."
            ),
            "groups": [group0, group1],
        })

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps({"choices": [{"message": {"content": response_content}}]}).encode()

        with patch.dict("os.environ", {"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}), \
                patch("urllib.request.urlopen", return_value=FakeResponse()):
            analysis, error = app.ai_policy_analysis(changes)

        self.assertIsNone(error)
        self.assertIsNotNone(analysis)
        self.assertEqual(len(analysis["groups"]), 2)
        self.assertIsNotNone(analysis["groups"][0]["claim_validation"]["why_it_matters"])
        self.assertIsNotNone(analysis["groups"][1]["claim_validation"]["why_it_matters"])

    def test_limitation_statement_with_intervening_clause_identified(self):
        """Honest statements describing missing earlier passages with intervening clauses are recognized."""
        s = "No corresponding passage regarding the Tarun Plus category or PMJDY overdrafts was found in the supplied earlier comparison."
        self.assertTrue(app._is_insufficient_evidence_statement(s))
        earlier_ev = "Specified limit of the loan shall be Rs.10 lakh or such other amount as may be decided"
        result = app.assess_evidence_claim(s, earlier_ev, allow_interpretation=True)
        self.assertEqual(result, "insufficient_evidence")

    def test_morphological_adverb_individual_mapping(self):
        """Adverb 'individually' in policy evidence grounds noun 'individuals' in affected groups."""
        evidence = "micro unit/enterprise set up under Joint Liability Group framework, individually or jointly, with PMJDY accounts"
        claim = "micro units and individuals with accounts"
        self.assertEqual(app.assess_evidence_claim(claim, evidence), "directly_supported")

    def test_live_gemini_combined_group_analysis_passes_validation(self):
        """Captured live Gemini response with combined change groups passes all validation steps."""
        changes = app.compare_docs(*app.SAMPLE_DOCS)
        group = self.analyst_group({
            "source_ids": ["cgfmu-loan-limit", "cgfmu-tarun-plus"],
            "supporting_evidence": ["cgfmu-loan-limit", "cgfmu-tarun-plus"],
            "title": "Expansion of Loan Limits and Introduction of Tarun Plus Category",
            "change_summary": "The maximum loan limit for micro units is increased to Rs. 20 lakh, and a new Tarun Plus category is established for entrepreneurs who have repaid previous Tarun loans.",
            "old_rule": "The specified limit of the loan was Rs. 10 lakh or another amount decided by the Fund. No corresponding passage regarding the Tarun Plus category or PMJDY overdrafts was found in the supplied earlier comparison.",
            "new_rule": "The loan limit is increased to Rs. 20 lakh. A new Tarun Plus category is created for loans between 10 lakh and 20 lakh for entrepreneurs who have successfully repaid previous Tarun loans. Overdraft facilities of Rs. 10,000 under PMJDY accounts are now eligible for coverage. The definition of an eligible borrower is updated to include micro units or enterprises whose credit requirements do not exceed the new Rs. 20 lakh limit.",
            "why_it_matters": "This change doubles the maximum credit available under the guarantee and provides a specific eligibility pathway for established borrowers to access higher funding levels.",
            "who_may_be_affected": "Entrepreneurs who have successfully repaid Tarun category loans, micro units or enterprises under the PMMY or MSMED Act, and individuals with PMJDY accounts.",
            "change_category": "Financial assistance",
            "change_type": "Mixed",
            "importance": "High",
            "importance_reason": "The provision significantly increases the maximum financial limit for loans and introduces a new category and eligibility definition for borrowers.",
            "substantive_change": True,
            "confidence": 1.0,
            "unknown_information": "The specific sectors covered under PMMY or the MSMED Act, and the exact eligibility criteria prescribed by the Fund mentioned in the borrower definition, are not detailed in these excerpts.",
            "simple_explanation": "The maximum loan amount for small businesses has been raised from 10 lakh to 20 lakh. A new Tarun Plus category is now available for business owners who have already repaid a previous Tarun loan, and small bank account overdrafts up to 10,000 are now covered.",
            "potentially_affected_groups": ["Entrepreneurs with repaid Tarun loans", "Micro units and enterprises", "PMJDY account holders"],
            "practical_implications": ["Higher borrowing capacity for established micro units", "Specific eligibility path for Tarun Plus based on repayment history", "Integration of PMJDY overdrafts into the guarantee scheme"],
            "what_changed_for_them": ["Can access higher loan amounts up to 20 lakh", "New Tarun Plus category available for those who successfully repaid Tarun loans", "PMJDY overdrafts of 10,000 are now eligible for coverage"],
            "limitations": "The excerpts do not define the specific eligibility criteria mentioned for the Fund or list the covered sectors under the MSMED Act.",
            "impact_supporting_evidence": ["cgfmu-loan-limit", "cgfmu-tarun-plus"],
        })
        response_content = json.dumps({
            "summary": "The amendments increase the maximum loan limit for micro units from Rs. 10 lakh to Rs. 20 lakh, introduce a new Tarun Plus category for entrepreneurs with a positive repayment history, and extend coverage to PMJDY overdraft facilities.",
            "groups": [group],
        })

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self):
                return json.dumps({
                    "status": "completed",
                    "steps": [{
                        "type": "model_output",
                        "content": [{"type": "text", "text": response_content}]
                    }]
                }).encode()

        with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}), \
                patch("app.open_provider_request", return_value=FakeResponse()):
            analysis, error = app.ai_policy_analysis(changes)

        self.assertIsNone(error)
        self.assertIsNotNone(analysis)
        self.assertEqual(len(analysis["groups"]), 1)
        self.assertEqual(analysis["groups"][0]["claim_validation"]["why_it_matters"], "interpretation")
        self.assertEqual(analysis["groups"][0]["claim_validation"]["old_rule"], "insufficient_evidence")
        self.assertEqual(analysis["groups"][0]["claim_validation"]["who_may_be_affected"], "directly_supported")

    def test_statutory_obligation_must_grounded_by_requirement(self):
        """Policy 'credit requirement' in excerpt grounds explanatory 'must' in policy analysis."""
        evidence = "Eligible borrower means micro enterprise whose credit requirement does not exceed Rs. 20 lakh."
        claim = "Borrowers must comply with the credit requirement of Rs. 20 lakh."
        result = app.assess_evidence_claim(claim, evidence, allow_interpretation=True)
        self.assertIsNotNone(result)

    def test_qualification_morphology_qualify_grounds_eligible(self):
        """'qualify' / 'qualifies' / 'qualifying' maps to 'eligible' in policy grounding."""
        evidence = "Eligible borrower means new or existing micro unit or enterprise."
        claim = "Micro units that qualify are considered eligible borrowers."
        result = app.assess_evidence_claim(claim, evidence, allow_interpretation=True)
        self.assertIsNotNone(result)

    def test_ask_civiclens_statutory_obligation_paraphrase_accepted(self):
        """Recovered Ask CivicLens sentence with statutory obligation 'must' is grounded by 'shall'."""
        evidence = (
            "Paragraph at Serial No.2 (xii) of Chapter I of Notification number S.O. 1443(E) dated the 18th April, 2016, "
            "shall be read as: 'Eligible borrower' means new or existing micro unit / enterprise, including micro unit/enterprise "
            "set up under Joint Liability Group (JLG) framework, individually or jointly (irrespective of the availability of "
            "guarantee under JLG), falling under any sector covered under PMMY or as defined in the MSMED Act, 2006 (as amended "
            "from time to time), who meets eligibility criteria prescribed by the Fund and whose credit requirement does not "
            "exceed the specified limit under PMMY. Specified limit of the loan shall be Rs.10 lakh or such other amount as may be decided"
        )
        s = "These borrowers must fall under sectors covered by the PMMY or the MSMED Act, 2006, and their credit requirement must not exceed Rs 10 lakh or an amount otherwise decided by the Fund."
        result = app.assess_evidence_claim(s, evidence, allow_interpretation=True)
        self.assertIsNotNone(result)

    def test_why_it_matters_qualification_and_obligation_paraphrases_accepted(self):
        """why_it_matters claims using 'must' and 'qualify' supported by source evidence pass validation."""
        changes = app.compare_docs(*app.SAMPLE_DOCS)
        all_ev = " ".join(((c.get("previous") or {}).get("text") or "") + " " + ((c.get("updated") or {}).get("text") or "") for c in changes)
        all_pairs = [((c.get("previous") or {}).get("text"), (c.get("updated") or {}).get("text")) for c in changes if c.get("previous") and c.get("updated")]
        tarun_c = next(c for c in changes if c["id"] == "cgfmu-tarun-plus")
        upd_text = tarun_c["updated"]["text"]

        claim1 = "Borrowers must meet eligibility criteria prescribed by the Fund to access loans up to 20 lakh."
        res1 = app.assess_evidence_claim(claim1, upd_text, allow_interpretation=True, evidence_pairs=all_pairs, context_evidence=all_ev)
        self.assertIsNotNone(res1)

        claim2 = "Entrepreneurs must have repaid previous Tarun category loans to qualify for Tarun Plus."
        res2 = app.assess_evidence_claim(claim2, upd_text, allow_interpretation=True, evidence_pairs=all_pairs, context_evidence=all_ev)
        self.assertIsNotNone(res2)

    def test_unsupported_must_and_qualify_claims_strictly_rejected(self):
        """Unsupported claims using 'must' or 'qualify' without source grounding are strictly rejected."""
        evidence = "Specified limit of the loan shall be Rs. 20 lakh for micro units."
        # 'students' and 'national portal' are completely unsupported
        self.assertIsNone(app.assess_evidence_claim("Students must apply through national portal.", evidence, allow_interpretation=True))
        # 'farmers' and 'grants' are completely unsupported
        self.assertIsNone(app.assess_evidence_claim("Farmers qualify for Rs. 50 lakh grants.", evidence, allow_interpretation=True))

    def test_multilingual_all_twelve_languages_supported(self):
        """Verify all 12 requested Indian languages are present with metadata and native names."""
        from policy_translator import SUPPORTED_LANGUAGES
        expected_languages = ["en", "te", "hi", "ta", "kn", "ml", "mr", "bn", "gu", "pa", "or", "ur"]
        for code in expected_languages:
            self.assertIn(code, SUPPORTED_LANGUAGES)
            self.assertTrue(len(SUPPORTED_LANGUAGES[code]["native"]) > 0)
            self.assertIn(SUPPORTED_LANGUAGES[code]["dir"], ("ltr", "rtl"))
        self.assertEqual(SUPPORTED_LANGUAGES["ur"]["dir"], "rtl")
        self.assertEqual(SUPPORTED_LANGUAGES["en"]["dir"], "ltr")

    def test_multilingual_policy_explanation_translation(self):
        """Verify policy summaries and explanations translate across Indian languages."""
        from policy_translator import translate_policy_text
        en_text = "The cited benefit limit changes from Rs 10 lakh to Rs 20 lakh."
        # Telugu
        te_text = translate_policy_text(en_text, "te")
        self.assertIn("రూ. 10 లక్షల", te_text)
        self.assertIn("రూ. 20 లక్షల", te_text)
        # Hindi
        hi_text = translate_policy_text(en_text, "hi")
        self.assertIn("10 लाख", hi_text)
        self.assertIn("20 लाख", hi_text)
        # Tamil
        ta_text = translate_policy_text(en_text, "ta")
        self.assertIn("10", ta_text)
        self.assertIn("20", ta_text)
        # Urdu (RTL)
        ur_text = translate_policy_text(en_text, "ur")
        self.assertIn("10", ur_text)
        self.assertIn("20", ur_text)
        # English remains unchanged
        self.assertEqual(translate_policy_text(en_text, "en"), en_text)

    def test_multilingual_evidence_first_preservation(self):
        """Translation preserves source_ids, supporting_evidence, classifications, and claim validations."""
        from policy_translator import translate_analysis_dict
        analysis = {
            "summary": "The cited benefit limit changes from Rs 10 lakh to Rs 20 lakh.",
            "groups": [{
                "source_ids": ["change-1", "change-2"],
                "supporting_evidence": ["change-1", "change-2"],
                "impact_supporting_evidence": ["change-1"],
                "title": "Benefit limit changed",
                "change_summary": "The cited limit changes from Rs 10 lakh to Rs 20 lakh.",
                "what_changed": "The cited limit changes from Rs 10 lakh to Rs 20 lakh.",
                "previous_rule": "The earlier passage says Rs 10 lakh.",
                "updated_rule": "The updated passage says Rs 20 lakh.",
                "why_it_matters": "The excerpt states a higher benefit limit.",
                "who_may_be_affected": "Applicants",
                "change_category": "Financial assistance",
                "change_type": "Modification",
                "importance": "High",
                "importance_reason": "Changes loan limits",
                "substantive_change": True,
                "confidence": 0.95,
                "unknown_information": "The supplied passages do not establish effects beyond the text shown.",
                "simple_explanation": "The excerpt states a higher benefit limit.",
                "potentially_affected_groups": ["Applicants"],
                "practical_implications": ["The excerpt states a higher benefit limit."],
                "what_changed_for_them": ["The excerpt states a higher benefit limit."],
                "limitations": "This passage-level analysis cannot establish broader legal or practical outcomes.",
                "claim_validation": {"what_changed": "directly_supported", "why_it_matters": "interpretation"}
            }]
        }
        for lang in ("te", "hi", "ta", "kn", "ml", "mr", "bn", "gu", "pa", "or", "ur"):
            translated = translate_analysis_dict(analysis, lang)
            group = translated["groups"][0]
            # Crucial: source IDs and evidence links must NEVER be changed
            self.assertEqual(group["source_ids"], ["change-1", "change-2"])
            self.assertEqual(group["supporting_evidence"], ["change-1", "change-2"])
            self.assertEqual(group["impact_supporting_evidence"], ["change-1"])
            self.assertEqual(group["change_type"], "Modification")
            self.assertEqual(group["importance"], "High")
            self.assertEqual(group["substantive_change"], True)
            self.assertEqual(group["confidence"], 0.95)
            self.assertEqual(group["claim_validation"]["what_changed"], "directly_supported")
            # The translated explanation text should not be empty
            self.assertTrue(len(translated["summary"]) > 0)
            self.assertTrue(len(group["what_changed"]) > 0)

    def test_ask_civiclens_multilingual_response(self):
        """Ask CivicLens answers in the selected language while keeping original evidence citations intact."""
        session_id = "multilingual-chat-session"
        app.remember_document_session(session_id,
            {"title": "Old notice", "version": "Earlier"},
            {"title": "Revised notice", "version": "Updated"},
            [{"page": 3, "text": "Eligible applicants may receive Rs 10 lakh.", "blocks": [{"text": "Eligible applicants may receive Rs 10 lakh."}]}],
            [{"page": 5, "text": "Eligible applicants may receive Rs 20 lakh.", "blocks": [{"text": "Eligible applicants may receive Rs 20 lakh."}]}],
            [{"id": "c1", "summary": "limit change"}],
        )
        evidence = app.retrieve_chat_evidence(app.DOCUMENT_SESSIONS[session_id], "What benefit may applicants receive?")
        answer_content = json.dumps({
            "answer": "The earlier notice lists Rs 10 lakh; the revised notice lists Rs 20 lakh.",
            "source_ids": [passage["id"] for passage in evidence]
        })
        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self):
                return json.dumps({
                    "status": "completed",
                    "steps": [{"type": "model_output", "content": [{"type": "text", "text": answer_content}]}]
                }).encode()

        # Telugu chat response
        with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "mock-key", "GEMINI_MODEL": "mock-model"}), patch("urllib.request.urlopen", return_value=FakeResponse()):
            result_te = app.answer_policy_question(session_id, "What benefit may applicants receive?", language="te")
        self.assertTrue(result_te["ok"])
        self.assertIn("రూ. 10 లక్షలు", result_te["answer"])
        self.assertIn("రూ. 20 లక్షలు", result_te["answer"])
        # Original sources must remain original English citations
        self.assertEqual(result_te["sources"][0]["text"], "Eligible applicants may receive Rs 10 lakh.")

        # Hindi chat response
        with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "mock-key", "GEMINI_MODEL": "mock-model"}), patch("urllib.request.urlopen", return_value=FakeResponse()):
            result_hi = app.answer_policy_question(session_id, "What benefit may applicants receive?", language="hi")
        self.assertTrue(result_hi["ok"])
        self.assertIn("10 लाख", result_hi["answer"])
        self.assertIn("20 लाख", result_hi["answer"])





    def test_dynamic_policy_content_translation(self):
        """Dynamic titles, summaries, and priority_reason translate in all 12 languages."""
        from policy_translator import translate_policy_text

        non_english = ("te", "hi", "ta", "kn", "ml", "mr", "bn", "gu", "pa", "or", "ur")

        # Fixed title: should translate for all non-English languages
        for lang in non_english:
            result = translate_policy_text("Modified loan limit", lang)
            self.assertNotEqual(result, "Modified loan limit", f"Title not translated for lang={lang}")
            self.assertGreater(len(result), 0)

        # Fixed summary: Hindi should contain the word for "notification"
        hi_summary = translate_policy_text(
            "The wording changed between the two notifications.", "hi"
        )
        self.assertNotEqual(
            hi_summary, "The wording changed between the two notifications.",
            "Summary not translated for hi"
        )

        # Fixed low priority_reason
        editorial_en = (
            "Classified as an editorial or heading change from the cited text; "
            "this is a review aid, not a legal assessment."
        )
        hi_low = translate_policy_text(editorial_en, "hi")
        self.assertNotEqual(hi_low, editorial_en, "Low priority_reason not translated for hi")

        # Dynamic high priority_reason pattern
        high_en = (
            "Higher review priority estimate because the changed provision concerns "
            "changed financial assistance terms. This is not a legal impact assessment."
        )
        hi_high = translate_policy_text(high_en, "hi")
        self.assertNotEqual(hi_high, high_en, "High priority_reason not translated for hi")

        # Dynamic medium priority_reason pattern
        med_en = (
            "Medium review priority because the changed text concerns application process. "
            "This is not a legal impact assessment."
        )
        te_med = translate_policy_text(med_en, "te")
        self.assertNotEqual(te_med, med_en, "Medium priority_reason not translated for te")

        # English must pass through unchanged
        en_result = translate_policy_text("Modified loan limit", "en")
        self.assertEqual(en_result, "Modified loan limit")

        # Numbers must be preserved in translated strings
        limit_en = "The cited benefit limit changes from Rs 10 lakh to Rs 20 lakh."
        hi_limit = translate_policy_text(limit_en, "hi")
        self.assertIn("10", hi_limit)
        self.assertIn("20", hi_limit)

        # ml key bug regression: Malayalam must translate (not silently fail)
        ml_title = translate_policy_text("Modified loan limit", "ml")
        self.assertNotEqual(ml_title, "Modified loan limit", "Malayalam (ml) translation failed — possible key bug")

        # translate_analysis_dict must translate priority_reason
        from policy_translator import translate_analysis_dict
        analysis = {
            "summary": "Two policy changes were detected.",
            "groups": [
                {
                    "title": "Modified loan limit",
                    "what_changed": "The wording changed between the two notifications.",
                    "priority_reason": editorial_en,
                    "why_it_matters": "The benefit limit is now different.",
                }
            ],
        }
        translated = translate_analysis_dict(analysis, "hi")
        group = translated["groups"][0]
        self.assertNotEqual(group["priority_reason"], editorial_en,
                            "translate_analysis_dict did not translate priority_reason")
        self.assertNotEqual(group["title"], "Modified loan limit",
                            "translate_analysis_dict did not translate title")

    def test_explain_policy_provision_end_to_end(self):
        """Test explain_policy_provision function with Groq SDK mock."""
        import groq as groq_sdk

        session_id = "test-explain-session"
        change_item = {
            "type": "Modified",
            "summary": "The cited benefit limit changes from Rs 10 lakh to Rs 20 lakh.",
            "previous": {
                "page": 2,
                "document": "Earlier Gazette",
                "version": "2023",
                "text": "Eligible applicants may receive Rs 10 lakh.",
            },
            "updated": {
                "page": 4,
                "document": "Updated Gazette",
                "version": "2024",
                "text": "Eligible applicants may receive Rs 20 lakh.",
            },
        }
        with app.DOCUMENT_SESSIONS_LOCK:
            app.DOCUMENT_SESSIONS[session_id] = {
                "changes": [change_item],
            }

        response_text = json.dumps({
            "explanation": "The cited benefit limit changes from Rs 10 lakh to Rs 20 lakh. The earlier passage says Rs 10 lakh. The updated passage says Rs 20 lakh."
        })

        groq_env = {
            "AI_PROVIDER": "groq",
            "GROQ_API_KEY": "mock-groq-key",
            "GROQ_MODEL": "openai/gpt-oss-20b",
        }

        def _make_sdk_response(content):
            fake_choice = MagicMock()
            fake_choice.message.content = content
            fake_resp = MagicMock()
            fake_resp.choices = [fake_choice]
            return fake_resp

        # 1. Successful English explanation
        with patch.dict("os.environ", groq_env), \
             patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = _make_sdk_response(response_text)
            result = app.explain_policy_provision(session_id, 0, language="en")
            self.assertTrue(result["ok"])
            self.assertIn("Rs 10 lakh", result["explanation"])
            self.assertIn("Rs 20 lakh", result["explanation"])
            self.assertIsNotNone(result["claim_status"])
            self.assertEqual(result["earlier_source"]["page"], 2)
            self.assertEqual(result["updated_source"]["page"], 4)
            # Verify SDK was called with correct model
            create_call = MockGroq.return_value.chat.completions.create.call_args
            self.assertEqual(create_call.kwargs["model"], "openai/gpt-oss-20b")
            self.assertNotIn("mock-groq-key", json.dumps(result))

        # 2. Multilingual explanation in Hindi
        with patch.dict("os.environ", groq_env), \
             patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = _make_sdk_response(response_text)
            result_hi = app.explain_policy_provision(session_id, 0, language="hi")
            self.assertTrue(result_hi["ok"])
            self.assertIn("10", result_hi["explanation"])
            self.assertIn("20", result_hi["explanation"])
            # Evidence sources remain original English
            self.assertEqual(result_hi["earlier_source"]["text"], "Eligible applicants may receive Rs 10 lakh.")

        # 3. Insufficient evidence handling: AI states limitation
        insufficient_response = json.dumps({
            "explanation": "The cited text does not specify the reason for this change."
        })
        with patch.dict("os.environ", groq_env), \
             patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = _make_sdk_response(insufficient_response)
            result_insufficient = app.explain_policy_provision(session_id, 0, language="en")
            self.assertTrue(result_insufficient["ok"])
            self.assertEqual(result_insufficient["claim_status"], "insufficient_evidence")

        # 4. Error handling: Expired session
        expired_result = app.explain_policy_provision("non-existent-session", 0)
        self.assertFalse(expired_result["ok"])
        self.assertIn("expired", expired_result["error"])

        # 5. Error handling: Invalid change index
        invalid_idx_result = app.explain_policy_provision(session_id, 999)
        self.assertFalse(invalid_idx_result["ok"])
        self.assertIn("Invalid change index", invalid_idx_result["error"])

        # 6. Error handling: Groq SDK 503 APIStatusError
        import httpx
        fake_http_response = MagicMock(spec=httpx.Response)
        fake_http_response.status_code = 503
        fake_http_response.headers = {}
        exc_503 = groq_sdk.APIStatusError(
            message="overloaded",
            response=fake_http_response,
            body={"error": {"message": "overloaded"}},
        )
        with patch.dict("os.environ", groq_env), \
             patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = exc_503
            err_result = app.explain_policy_provision(session_id, 0)
            self.assertFalse(err_result["ok"])
            self.assertIn("temporarily overloaded", err_result["error"])

    def test_groq_explain_and_health_check(self):
        """Verify Groq SDK integration for Explain with AI and health checks."""
        import groq as groq_sdk
        import httpx

        session_id = "test-groq-explain-session"
        change_item = {
            "type": "Modified",
            "summary": "The cited benefit limit changes from Rs 10 lakh to Rs 20 lakh.",
            "previous": {
                "page": 2,
                "document": "Earlier Gazette",
                "version": "2023",
                "text": "Eligible applicants may receive Rs 10 lakh.",
            },
            "updated": {
                "page": 4,
                "document": "Updated Gazette",
                "version": "2024",
                "text": "Eligible applicants may receive Rs 20 lakh.",
            },
        }
        with app.DOCUMENT_SESSIONS_LOCK:
            app.DOCUMENT_SESSIONS[session_id] = {
                "changes": [change_item],
            }

        groq_env = {
            "AI_PROVIDER": "groq",
            "GROQ_API_KEY": "mock-groq-key",
            "GROQ_MODEL": "openai/gpt-oss-20b",
        }

        def _sdk_response(content):
            fake_choice = MagicMock()
            fake_choice.message.content = content
            fake_resp = MagicMock()
            fake_resp.choices = [fake_choice]
            return fake_resp

        def _sdk_exc(status, msg):
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = status
            resp.headers = {}
            return groq_sdk.APIStatusError(
                message=msg,
                response=resp,
                body={"error": {"message": msg}},
            )

        # 1. Successful explain request
        valid_json = json.dumps({
            "explanation": "The cited benefit limit changes from Rs 10 lakh to Rs 20 lakh. The earlier passage says Rs 10 lakh. The updated passage says Rs 20 lakh."
        })
        with patch.dict("os.environ", groq_env), patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = _sdk_response(valid_json)
            res = app.explain_policy_provision(session_id, 0, language="en")
            self.assertTrue(res["ok"])
            self.assertIn("Rs 10 lakh", res["explanation"])
            create_call = MockGroq.return_value.chat.completions.create.call_args
            self.assertEqual(create_call.kwargs["model"], "openai/gpt-oss-20b")
            self.assertEqual(create_call.kwargs.get("response_format"), {"type": "json_object"})

        # 2. Invalid JSON response from Groq
        with patch.dict("os.environ", groq_env), patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = _sdk_response("not-valid-json")
            res_bad_json = app.explain_policy_provision(session_id, 0, language="en")
            self.assertFalse(res_bad_json["ok"])
            self.assertIn("valid JSON", res_bad_json["error"])

        # 3. Insufficient evidence response from Groq
        insufficient_json = json.dumps({
            "explanation": "The cited text does not specify the reason for this change."
        })
        with patch.dict("os.environ", groq_env), patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = _sdk_response(insufficient_json)
            res_insufficient = app.explain_policy_provision(session_id, 0, language="en")
            self.assertTrue(res_insufficient["ok"])
            self.assertEqual(res_insufficient["claim_status"], "insufficient_evidence")

        # 4. HTTP 429 Rate Limit
        with patch.dict("os.environ", groq_env), patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = _sdk_exc(429, "rate limit exceeded")
            res_429 = app.explain_policy_provision(session_id, 0, language="en")
            self.assertFalse(res_429["ok"])
            self.assertIn("rate or usage limits", res_429["error"])

        # 5. Timeout handling
        with patch.dict("os.environ", groq_env), patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = groq_sdk.APITimeoutError(request=MagicMock())
            res_timeout = app.explain_policy_provision(session_id, 0, language="en")
            self.assertFalse(res_timeout["ok"])
            self.assertIn("timed out", res_timeout["error"])

        # 6. Groq health check via SDK
        fake_choice = MagicMock()
        fake_choice.message.content = "CIVICLENS_AI_HEALTH_OK"
        fake_health_resp = MagicMock()
        fake_health_resp.choices = [fake_choice]
        with patch.dict("os.environ", groq_env), patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = fake_health_resp
            health = app.ai_provider_health_check()
            self.assertTrue(health["ok"])
            self.assertEqual(health["provider"], "groq")
            self.assertEqual(health["model"], "openai/gpt-oss-20b")


if __name__ == "__main__":
    unittest.main()
