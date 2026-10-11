#!/usr/bin/env python3
"""Stop hook: 拦截对话里会在 GitHub 风格 markdown 中变成乱码的写法。

检查最后一条 assistant 消息（去掉代码块和行内代码后）：
  1. LaTeX 定界符 $...$ / $$...$$ 或 \\命令{ ；
  2. 裸写的特殊 token <s> <e> <pad> <eos> <bos>；
  3. 表格单元格里的 $。
命中则返回 decision=block，让模型按 AGENTS.md「对话里的公式与特殊符号」重写。
知识卡文件不受影响，本脚本只看对话文本。
"""
import json
import re
import sys


def last_assistant_text(path: str) -> str:
    text = ""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") != "assistant":
                    continue
                content = rec.get("message", {}).get("content", "")
                if isinstance(content, str):
                    parts = [content]
                else:
                    parts = [b.get("text", "") for b in content if b.get("type") == "text"]
                if any(p.strip() for p in parts):
                    text = "\n".join(parts)
    except OSError:
        return ""
    return text


def strip_code(md: str) -> str:
    md = re.sub(r"```.*?```", "", md, flags=re.S)
    md = re.sub(r"~~~.*?~~~", "", md, flags=re.S)
    md = re.sub(r"`[^`\n]*`", "", md)
    return md


CHECKS = [
    (r"\$\$", "出现 $$（LaTeX 独立公式定界符）"),
    (r"\$[^$\n]{1,200}\$", "出现 $...$（LaTeX 行内公式）"),
    (r"\\(mathcal|mathbb|text|operatorname|frac|sum|mid|ldots|cdot|theta|lambda|pi|sigma|log|exp)\b", "出现 LaTeX 命令"),
    (r"(?<!`)<(s|e|pad|eos|bos|unk|sep|cls)>(?!`)", "裸写特殊 token（<s>、<e> 等会被当成 HTML）"),
    (r"^\|.*\$.*\|$", "表格单元格里出现 $"),
]


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    # 已经因为本 hook 续写过一次，不再拦，避免死循环
    if data.get("stop_hook_active"):
        return
    path = data.get("transcript_path")
    if not path:
        return
    text = strip_code(last_assistant_text(path))
    hits = []
    for pat, why in CHECKS:
        m = re.search(pat, text, flags=re.M)
        if m:
            snippet = m.group(0).strip()[:60]
            hits.append(f"{why}：`{snippet}`")
    if not hits:
        return
    reason = (
        "对话里有会渲染成乱码的写法，请按 AGENTS.md「对话里的公式与特殊符号」重写刚才的回复："
        "公式用 Unicode 或 ```text 代码块，特殊 token 放反引号，表格里不放 $。命中：\n- "
        + "\n- ".join(hits)
    )
    print(json.dumps({"decision": "block", "reason": reason}, ensure_ascii=False))


if __name__ == "__main__":
    main()
