# Thin wrapper around the Open ASR Leaderboard text normalizer.
#
# The implementation lives in the upstream repo, pinned as a git submodule at
# third_party/open_asr_leaderboard, so we track their changes instead of keeping a
# copy in sync by hand:
#
#     git submodule update --init third_party/open_asr_leaderboard
#
# To move to a newer upstream normalizer:
#
#     git -C third_party/open_asr_leaderboard fetch origin
#     git -C third_party/open_asr_leaderboard checkout <commit>
#     git add third_party/open_asr_leaderboard   # records the new pin
#
# Bumping the pin can change WER, since the normalizer defines what counts as a
# match, so treat it as a change to the eval itself and re-run the baselines.

import importlib.util
import sys
from pathlib import Path

SUBMODULE_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "open_asr_leaderboard"
_NORMALIZER_PKG = SUBMODULE_ROOT / "normalizer"
# Loaded under a private name so the leaderboard's generically named top-level
# `normalizer` package never has to go on sys.path.
_MODULE_NAME = "asrq.evaluation._open_asr_leaderboard_normalizer"


def _load_upstream_normalizer():
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]

    init_file = _NORMALIZER_PKG / "__init__.py"
    if not init_file.is_file():
        raise ImportError(
            f"The Open ASR Leaderboard submodule is missing from {SUBMODULE_ROOT}. "
            "Run: git submodule update --init third_party/open_asr_leaderboard"
        )

    spec = importlib.util.spec_from_file_location(
        _MODULE_NAME, init_file, submodule_search_locations=[str(_NORMALIZER_PKG)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the Open ASR Leaderboard normalizer from {init_file}")

    module = importlib.util.module_from_spec(spec)
    # Registered before exec_module so the package's own relative imports resolve.
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[_MODULE_NAME]
        raise
    return module


def load_upstream_data_utils():
    """Load the leaderboard's normalizer/data_utils.py, without putting `normalizer` on sys.path.

    data_utils imports its package by the absolute name (`from normalizer import ...`) and its
    siblings relatively, so it is loaded as a submodule of the privately named package above,
    with `normalizer` aliased to that package only while the import runs. Evaluation then works
    without adding the submodule to PYTHONPATH.
    """
    name = f"{_MODULE_NAME}.data_utils"
    if name in sys.modules:
        return sys.modules[name]
    package = _load_upstream_normalizer()
    spec = importlib.util.spec_from_file_location(name, _NORMALIZER_PKG / "data_utils.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load data_utils.py from {_NORMALIZER_PKG}")
    module = importlib.util.module_from_spec(spec)
    aliased = "normalizer" not in sys.modules
    if aliased:
        sys.modules["normalizer"] = package
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    finally:
        if aliased:
            del sys.modules["normalizer"]
    return module


_upstream = _load_upstream_normalizer()
_impl = sys.modules[f"{_MODULE_NAME}.normalizer"]

BasicTextNormalizer = _impl.BasicTextNormalizer
BasicMultilingualTextNormalizer = _impl.BasicMultilingualTextNormalizer
EnglishNumberNormalizer = _impl.EnglishNumberNormalizer
EnglishSpellingNormalizer = _impl.EnglishSpellingNormalizer
EnglishTextNormalizer = _impl.EnglishTextNormalizer
remove_symbols = _impl.remove_symbols
remove_symbols_and_diacritics = _impl.remove_symbols_and_diacritics

normalizer = EnglishTextNormalizer()

__all__ = [
    "BasicTextNormalizer",
    "BasicMultilingualTextNormalizer",
    "EnglishNumberNormalizer",
    "EnglishSpellingNormalizer",
    "EnglishTextNormalizer",
    "load_upstream_data_utils",
    "normalizer",
    "remove_symbols",
    "remove_symbols_and_diacritics",
]
