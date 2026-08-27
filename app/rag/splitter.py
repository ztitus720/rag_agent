import re


SECTION_PATTERN = (
    r"(?m)^(教育经历|项目经历|专业技能|工作经历|实习经历|"
    r"科研经历|获奖经历|证书|技能)\s*$"
)

PROJECT_PATTERN = (
    r"(?m)^(基于 Web 的在线调查问卷系统|"
    r"基于 Web 的校园二手交易系统|"
    r"多六足机器人的研制)"
)


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fallback_split(text, chunk_size=1200, overlap=150):
    text = _normalize_text(text)

    chunks = []
    start = 0

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def _split_project_section(section):
    """
    将“项目经历”按照具体项目切分。
    不把“项目经历”这个标题单独作为 chunk。
    """

    matches = list(re.finditer(PROJECT_PATTERN, section))

    if not matches:
        return _fallback_split(section)

    chunks = []

    for i, match in enumerate(matches):
        start = match.start()
        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(section)
        )

        chunk = section[start:end].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def split_text(text, chunk_size=1200, overlap=150):
    """
    Structure-aware chunking.

    对简历类文档：
    - 按 section 识别
    - 项目经历进一步按项目切分
    - section 标题不单独形成 chunk
    """

    text = _normalize_text(text)

    matches = list(re.finditer(SECTION_PATTERN, text))

    if not matches:
        return _fallback_split(text, chunk_size, overlap)

    chunks = []

    # section 之前的内容，例如姓名、联系方式
    prefix = text[:matches[0].start()].strip()

    if prefix:
        chunks.append(prefix)

    for i, match in enumerate(matches):
        section_name = match.group(1)

        start = match.end()
        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        content = text[start:end].strip()

        if not content:
            continue

        # 项目经历特殊处理
        if section_name == "项目经历":
            project_chunks = _split_project_section(content)

            for chunk in project_chunks:
                if len(chunk) <= chunk_size:
                    chunks.append(chunk)
                else:
                    chunks.extend(
                        _fallback_split(
                            chunk,
                            chunk_size,
                            overlap,
                        )
                    )

        else:
            # 普通 section
            if len(content) <= chunk_size:
                chunks.append(content)
            else:
                chunks.extend(
                    _fallback_split(
                        content,
                        chunk_size,
                        overlap,
                    )
                )

    return chunks