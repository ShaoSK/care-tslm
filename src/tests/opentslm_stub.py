"""Minimal stand-ins for the OpenTSLM classes the dataset uses, copied in behaviour from
OpenTSLM main (QADataset._format_sample, PromptWithAnswer.to_dict, TextTimeSeriesPrompt checks,
and the collate padding), so the dataset can be tested without torch. On the GPU machine the real
OpenTSLM is used instead."""

from __future__ import annotations

import sys
import types

import numpy as np


class TextPrompt:
    def __init__(self, text: str) -> None:
        assert isinstance(text, str)
        self._text = text

    def get_text(self) -> str:
        return self._text


class TextTimeSeriesPrompt:
    def __init__(self, text, time_series) -> None:
        assert isinstance(text, str), "Text must be a string!"
        arr = np.array(time_series)
        assert arr.ndim == 1 and arr.size > 0 and np.issubdtype(arr.dtype, np.number)
        self._text, self._series = text, arr

    def get_text(self):
        return self._text

    def get_time_series(self):
        return self._series


class QADataset:
    def __init__(self, split, EOS_TOKEN, format_sample_str=False, time_series_format_function=None):
        self.EOS_TOKEN = EOS_TOKEN
        if not hasattr(self.__class__, "loaded"):
            train, val, test = self._load_splits()
            self.__class__._train_dataset = list(map(self._format_sample, train))
            self.__class__._validation_dataset = list(map(self._format_sample, val))
            self.__class__._test_dataset = list(map(self._format_sample, test))
            self.__class__.loaded = True
        self.dataset = {
            "train": self.__class__._train_dataset,
            "validation": self.__class__._validation_dataset,
            "test": self.__class__._test_dataset,
        }[split]

    def _format_sample(self, row):
        answer = self._get_answer(row)
        if not answer.endswith(self.EOS_TOKEN):
            answer += self.EOS_TOKEN
        prompts = self._get_text_time_series_prompt_list(row)
        return {
            "answer": answer.strip(),
            "post_prompt": TextPrompt(self._get_post_prompt(row).strip()).get_text(),
            "pre_prompt": TextPrompt(self._get_pre_prompt(row).strip()).get_text(),
            "time_series": [p.get_time_series() for p in prompts],
            "time_series_text": [p.get_text() for p in prompts],
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]


def collate_pad(batch, patch_size=4):
    """Numpy version of extend_time_series_to_match_patch_size_and_aggregate."""
    for element in batch:
        ts = [np.asarray(s, dtype=np.float32) for s in element["time_series"]]
        n = -(-max(len(s) for s in ts) // patch_size) * patch_size
        element["time_series"] = np.stack([np.pad(s, (0, n - len(s)))[:n] for s in ts])
    return batch


def install() -> None:
    modules = {
        "opentslm": types.ModuleType("opentslm"),
        "opentslm.prompt": types.ModuleType("opentslm.prompt"),
        "opentslm.prompt.text_time_series_prompt": types.ModuleType("opentslm.prompt.text_time_series_prompt"),
        "opentslm.time_series_datasets": types.ModuleType("opentslm.time_series_datasets"),
        "opentslm.time_series_datasets.QADataset": types.ModuleType("opentslm.time_series_datasets.QADataset"),
    }
    modules["opentslm.prompt.text_time_series_prompt"].TextTimeSeriesPrompt = TextTimeSeriesPrompt
    modules["opentslm.time_series_datasets.QADataset"].QADataset = QADataset
    sys.modules.update(modules)
