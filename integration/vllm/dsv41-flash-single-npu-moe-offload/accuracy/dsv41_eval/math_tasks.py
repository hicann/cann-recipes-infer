# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MATH-500 and GSM8K pieces for the DSv4.1 suite.

MATH-500: HuggingFaceH4/MATH-500 test.jsonl (problem, solution, answer, subject, level, unique_id).
  Scoring: last \\boxed{...} of the response (fallback: math-verify's own extraction over the whole
  response), compared with the gold `answer` by math_verify.verify. AISBench's MATHEvaluator is not
  used because it extracts the *first* boxed match and silently skips (counts wrong, drops the detail
  row of) any item whose gold does not parse.
GSM8K: AISBench GSM8KDataset + 0-shot CoT chat prompt ("answer:$ANSWER"). Scoring: the number after the
  last "answer:" (fallback: last \\boxed{}, then last number in the text), commas/$/units stripped,
  numeric equality with the "#### N" gold. AISBench's gsm8k_postprocess takes the last regex number,
  which breaks on "1,000" (-> 000) and trailing text.
"""
import json
import re

from datasets import Dataset

from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.openicl.icl_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import ICL_EVALUATORS, LOAD_DATASET, TEXT_POSTPROCESSORS


# ----------------------------------------------------------------------------- MATH-500
@LOAD_DATASET.register_module()
class Math500Dataset(BaseDataset):

    @staticmethod
    def load(path: str, **kwargs):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                rows.append(dict(problem=t["problem"], answer=t["answer"], subject=t.get("subject", ""),
                                 level=t.get("level", 0), unique_id=t.get("unique_id", "")))
        return Dataset.from_list(rows)


def last_boxed(text: str):
    """Content of the last \\boxed{...} / \\fbox{...} with balanced braces, or None."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        m = re.match(r"\\boxed\s+([^\s$]+)", text[idx:])  # "\boxed 5"
        return m.group(1) if m else None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


@ICL_EVALUATORS.register_module()
class Math500VerifyEvaluator(BaseEvaluator):

    def score(self, predictions, references, test_set=None):
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

        norm = NormalizationConfig(nits=False, malformed_operators=False, basic_latex=True,
                                   equations=True, boxed="all", units=True)
        latex_cfg = [LatexExtractionConfig(normalization_config=norm, boxed_match_priority=0), ExprExtractionConfig()]

        def safe_parse(s, **kw):
            try:
                return parse(s, **kw)
            except Exception:  # noqa: BLE001
                return []

        details, correct = [], 0
        for pred, gold in zip(predictions, references):
            gold_p = safe_parse(f"${gold}$", extraction_config=latex_cfg) or safe_parse(gold)
            box = last_boxed(pred)
            if box is not None:
                pred_p = safe_parse(f"$\\boxed{{{box}}}$", extraction_config=latex_cfg) or safe_parse(box)
            else:
                pred_p = safe_parse(pred, extraction_config=latex_cfg)
            try:
                ok = bool(gold_p) and bool(pred_p) and bool(verify(gold_p, pred_p))
            except Exception:  # noqa: BLE001
                ok = False
            # exact string match on the boxed content as a last resort (e.g. text answers)
            if not ok and box is not None:
                ok = re.sub(r"\s|\\[,!;:]|\\text\{|\}|\{", "", box) == re.sub(r"\s|\\[,!;:]|\\text\{|\}|\{", "", gold)
            correct += ok
            details.append(dict(extracted=box, pred_parsed=str(pred_p), answer=gold, gold_parsed=str(gold_p),
                                no_boxed=box is None, correct=ok))
        n = max(len(predictions), 1)
        return {"accuracy": 100.0 * correct / n,
                "no_boxed_rate": 100.0 * sum(d["no_boxed"] for d in details) / n,
                "details": details}


# ----------------------------------------------------------------------------- GSM8K
_NUM = r"-?\$?\s*\d[\d,]*(?:\.\d+)?"


def _clean_num(s):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return v


@TEXT_POSTPROCESSORS.register_module()
def gsm8k_answer(text: str) -> str:
    t = text.replace("**", "")
    # "answer: N" (0-shot prompt) or "The answer is N" (8-shot exemplars); last occurrence wins
    for pat in [rf"(?i)answer\s*(?:[:：]|is\b[:：]?)\s*\$?\\?\(?\s*({_NUM})", None, rf"({_NUM})"]:
        if pat is None:
            box = last_boxed(t)
            if box is not None:
                nums = re.findall(_NUM, box)
                if nums:
                    v = _clean_num(nums[-1])
                    if v is not None:
                        return f"{v:g}" if v != int(v) else str(int(v))
            continue
        matches = re.findall(pat, t)
        if matches:
            v = _clean_num(matches[-1])
            if v is not None:
                return f"{v:g}" if v != int(v) else str(int(v))
    return "NULL"


@ICL_EVALUATORS.register_module()
class Gsm8kNumericEvaluator(BaseEvaluator):

    def score(self, predictions, references, test_set=None):
        details, correct = [], 0
        for pred, gold in zip(predictions, references):
            g = _clean_num(str(gold))
            p = _clean_num(str(pred)) if pred != "NULL" else None
            ok = p is not None and g is not None and abs(p - g) < 1e-6
            correct += ok
            details.append(dict(pred=pred, answer=gold, correct=ok))
        return {"accuracy": 100.0 * correct / max(len(predictions), 1), "details": details}
