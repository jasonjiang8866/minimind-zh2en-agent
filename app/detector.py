import regex as re

# CJK Unified + Extension A + Compatibility Ideographs cover the dataset well
_CHN_RX = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")

def has_chinese(s: str) -> bool:
    return bool(_CHN_RX.search(s or ""))
