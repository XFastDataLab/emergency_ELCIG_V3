"""职责词汇容错：忽略非汉字，至少三个不同汉字相同（无需连续）。"""
import re


def three_character_match(left: str, right: str) -> bool:
    return len(set(re.findall(r"[\u4e00-\u9fff]", left or "")) &
               set(re.findall(r"[\u4e00-\u9fff]", right or ""))) >= 3
