import re
import json
import hashlib
from loguru import logger
from typing import List, Tuple, Optional
import os

from lean_dojo import LeanGitRepo, Theorem, Pos, is_available_in_cache


# ======================================================================================================================================
def extract_tac(chat_res):
    lst = re.findall('`.+`:', chat_res)
    new_lst = []
    for i in lst:
        new_lst.append(i.rstrip(':').strip('`'))

    return new_lst


# ======================================================================================================================================
def _get_theorems(
    data_path: str,
    split: str,
    file_path: str,
    full_name: str,
    name_filter: str,
    num_theorems: int,
) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    repo, theorems, positions = _get_theorems_from_files(
        data_path,
        split,
        file_path,
        full_name,
        name_filter,
        num_theorems,
    )

    all_repos = {thm.repo for thm in theorems}
    for r in all_repos:
        assert is_available_in_cache(
            r
        ), f"{r} has not been traced yet. Please use LeanDojo to trace it so that it's available in the cache."

    return repo, theorems, positions


def _get_theorems_from_files(
    data_path: str,
    split: str,
    file_path: Optional[str],
    full_name: Optional[str],
    name_filter: Optional[str],
    num_theorems: Optional[int],
) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    data = json.load(open(os.path.join(data_path, f"{split}.json")))
    theorems = []
    positions = []

    for t in data:
        if file_path is not None and t["file_path"] != file_path:
            continue
        if full_name is not None and t["full_name"] != full_name:
            continue
        if name_filter is not None and not hashlib.md5(
            t["full_name"].encode()
        ).hexdigest().startswith(name_filter):
            continue
        repo = LeanGitRepo(t["url"], t["commit"])
        theorems.append(Theorem(repo, t["file_path"], t["full_name"]))
        positions.append(Pos(*t["start"]))
    theorems = sorted(
        theorems,
        key=lambda t: hashlib.md5(
            (str(t.file_path) + ":" + t.full_name).encode()
        ).hexdigest(),
    )
    if num_theorems is not None:
        theorems = theorems[:num_theorems]
        positions = positions[:num_theorems]
    logger.info(f"{len(theorems)} theorems loaded from {data_path}")

    metadata = json.load(open(os.path.join(data_path, "../metadata.json")))
    repo = LeanGitRepo(metadata["from_repo"]["url"], metadata["from_repo"]["commit"])

    return repo, theorems, positions
