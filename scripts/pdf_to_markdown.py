#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from pypdf import PdfReader


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _join_wrapped_lines(lines: list[str]) -> str:
    """Join lines from a PDF that are typically hard-wrapped.

    Heuristics:
    - Fix common hyphenation at line-end: "trans-" + "former" -> "transformer".
    - Otherwise join with a single space.

    This is intentionally conservative; we don't attempt to infer bullets/indentation.
    """

    out: str = ""
    for line in lines:
        s = line.strip()
        if not s:
            continue

        if not out:
            out = s
            continue

        # If the previous chunk ends with a hyphen, assume soft hyphenation.
        if out.endswith("-") and s and s[0].isalnum():
            out = out[:-1] + s
        else:
            out += " " + s

    # Normalize whitespace.
    out = _normalize_whitespace(out)
    return out


_TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

_TITLE_CONTINUATION_WORDS = {
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "to",
    "with",
}


def _smart_titlecase_if_allcaps(s: str) -> str:
    # If it's not mostly upper-case, preserve original casing.
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return s

    upper = sum(1 for c in letters if c.isupper())
    if upper / max(1, len(letters)) < 0.7:
        return s

    parts = s.split(" ")
    out_parts: list[str] = []
    prev_ended_sentence = True

    for raw in parts:
        if raw == "":
            continue

        # Keep surrounding punctuation while casing the core token.
        m = re.match(r"^(\W*)([A-Za-z0-9&]+)(\W*)$", raw)
        if not m:
            out_parts.append(raw)
            prev_ended_sentence = raw.endswith(":")
            continue

        prefix, core, suffix = m.group(1), m.group(2), m.group(3)

        core_letters = re.sub(r"[^A-Za-z]", "", core)
        lower = core.lower()

        if core_letters.isupper() and len(core_letters) <= 4:
            # Preserve acronyms like IO, but lowercase short stopwords like "for".
            if (not prev_ended_sentence) and (lower in _TITLE_STOPWORDS):
                cased = lower
            else:
                cased = core_letters
        else:
            if (not prev_ended_sentence) and lower in _TITLE_STOPWORDS:
                cased = lower
            else:
                cased = lower[:1].upper() + lower[1:]

        # Re-insert non-letters (numbers, &, hyphens, etc.) preserving original.
        # If we can't confidently case it, keep it as-is.
        if core != core_letters:
            cased = core

        out_parts.append(prefix + cased + suffix)
        prev_ended_sentence = suffix.endswith(":")

    return " ".join(out_parts)


def _fix_allcaps_spaced_letters(s: str) -> str:
    # Some PDFs extract as "G ENERAL"; merge single-letter splits.
    prev = None
    cur = s
    for _ in range(5):
        if cur == prev:
            break
        prev = cur
        # Avoid merging standalone articles like "A" / "I".
        cur = re.sub(r"\b([B-HJ-Z])\s+([A-Z]{2,})\b", r"\1\2", cur)
    return cur


def _looks_like_author_line(s: str) -> bool:
    s2 = s.strip()
    if not s2:
        return False

    if re.match(r"^https?://", s2):
        return True

    if any(ch in s2 for ch in ("@", "{", "}", "*", "∗", "†", "‡")):
        return True

    if "arxiv" in s2.lower():
        return True

    # Comma-separated author lists are common.
    if s2.count(",") >= 1:
        nameish = re.findall(r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?\b", s2)
        if len(nameish) >= 3 and len(s2) <= 140:
            return True

    # Affiliation-ish lines.
    if re.search(
        r"\b(University|Department|School of|Institute|Laboratory|Lab|DeepMind|Google|Microsoft|Meta|Facebook|Amazon|Apple)\b",
        s2,
    ):
        return True

    return False


def extract_pdf_title(reader: PdfReader, fallback: str) -> str:
    arxiv_like = bool(re.fullmatch(r"\d{4}\.\d{5}v\d+", fallback))

    # If the filename already looks like a real title, prefer it.
    # (Avoids cases where first-page extraction contains headers/duplicated lines.)
    if (
        (not arxiv_like)
        and (not re.search(r"\d", fallback))
        and ("(" not in fallback)
        and (")" not in fallback)
    ):
        if 4 <= len(fallback) <= 120:
            return fallback

    # 1) Prefer embedded metadata.
    meta = reader.metadata
    meta_title = getattr(meta, "title", None) if meta else None
    if meta_title:
        t = _normalize_whitespace(str(meta_title))
        if t and t.lower() not in {"untitled", "title"}:
            return t

    # 2) Heuristic: infer from first page.
    try:
        first_page_text = reader.pages[0].extract_text() or ""
    except Exception:
        first_page_text = ""

    lines = [ln.strip() for ln in first_page_text.splitlines()]
    lines = [ln for ln in lines if ln]

    # Skip leading boilerplate.
    def _is_boilerplate_line(s: str) -> bool:
        sl = s.lower()
        return (
            sl.startswith("provided proper attribution")
            or "grants permission" in sl
            or "permission" in sl
            or "journalistic" in sl
            or "scholarly works" in sl
            or sl.startswith("published as")
            or "conference paper" in sl
            or sl.startswith("journal of")
            or "latex class files" in sl
            or sl.startswith("proceedings of")
            or sl.startswith("preprint")
        )

    while lines and _is_boilerplate_line(lines[0]):
        lines.pop(0)

    # Stop at abstract if present.
    stop_idx = None
    for i, ln in enumerate(lines[:80]):
        if ln.lower().startswith("abstract"):
            stop_idx = i
            break
    if stop_idx is not None:
        lines = lines[:stop_idx]

    title_lines: list[str] = []
    max_lines = 3 if arxiv_like else 1

    for ln in lines[:12]:
        if not ln:
            break

        if re.match(r"^https?://", ln):
            # Often a project page; not part of the title.
            break

        if _looks_like_author_line(ln):
            break

        if ln.lower().startswith("arxiv:"):
            break

        # After the first line, treat long Title-Case lines as likely author lists.
        if title_lines:
            words = re.findall(r"[A-Za-z][A-Za-z'\-]*", ln)
            if len(words) >= 8:
                lowers = {w.lower() for w in words}
                if not (lowers & _TITLE_CONTINUATION_WORDS):
                    break

        title_lines.append(ln)
        if len(title_lines) >= max_lines:
            break

        if len(" ".join(title_lines)) > 180:
            break

    if not title_lines:
        return fallback

    candidate = _normalize_whitespace(" ".join(title_lines))
    candidate = _fix_allcaps_spaced_letters(candidate)
    candidate = _smart_titlecase_if_allcaps(candidate)
    candidate = candidate.strip(" -–—:\t")

    return candidate or fallback


def _safe_filename(s: str) -> str:
    # Keep spaces for readability; remove path separators and other problematic chars.
    s2 = _normalize_whitespace(s)
    s2 = s2.replace("/", "-").replace(":", " - ")
    s2 = re.sub(r"[\\\0]", "-", s2)
    s2 = re.sub(r"[<>\|\*\?\"]", "", s2)
    s2 = _normalize_whitespace(s2)
    return s2[:180].strip(" .")


def _slugify(s: str) -> str:
    s2 = _normalize_whitespace(s).lower()
    s2 = re.sub(r"[^a-z0-9]+", "-", s2).strip("-")
    return s2[:80] or "doc"


def _index_existing_outputs(out_dir: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for p in out_dir.glob("*.md"):
        try:
            head = p.read_text(encoding="utf-8", errors="ignore").splitlines()[:60]
        except Exception:
            continue
        for ln in head:
            m = re.search(r"<!--\s*Source:\s*(.+?)\s*-->", ln)
            if m:
                src = m.group(1)
                out.setdefault(src, []).append(p)
                break
    return out


def _extract_arxiv_id_from_text(text: str) -> str | None:
    # New-style IDs: arXiv:2103.03206v2 or 2103.03206
    m = re.search(r"arXiv:\s*(\d{4}\.\d{5})(v\d+)?", text, flags=re.IGNORECASE)
    if m:
        return m.group(1) + (m.group(2) or "")

    # Sometimes the versionless id appears without the prefix.
    m = re.search(r"\b(\d{4}\.\d{5})(v\d+)?\b", text)
    if m:
        return m.group(1) + (m.group(2) or "")

    return None


_URL_RE = re.compile(r"https?://[^\s\)\]<>\"']+")


def _extract_urls_from_text(text: str) -> list[str]:
    urls = []
    for m in _URL_RE.finditer(text or ""):
        u = m.group(0).rstrip(".,;:")
        urls.append(u)
    return urls


def _extract_urls_from_pdf_annotations(
    reader: PdfReader, max_pages: int = 3
) -> list[str]:
    urls: list[str] = []

    for i in range(min(max_pages, len(reader.pages))):
        page = reader.pages[i]
        annots = page.get("/Annots") if hasattr(page, "get") else None
        if not annots:
            continue

        for annot_ref in annots:
            try:
                annot = annot_ref.get_object()
            except Exception:
                continue

            try:
                action = annot.get("/A")
                if not action:
                    continue
                uri = action.get("/URI")
                if not uri:
                    continue
                urls.append(str(uri))
            except Exception:
                continue

    return urls


_KNOWN_TITLE_URLS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bannotated\s+transformer\b", flags=re.IGNORECASE),
        "https://nlp.seas.harvard.edu/annotated-transformer/",
    ),
]


def extract_web_source_urls(title: str, reader: PdfReader) -> list[str]:
    # 1) Known title mappings (best-effort; avoids relying on PDF having embedded links).
    for pat, url in _KNOWN_TITLE_URLS:
        if pat.search(title):
            return [url]

    # 2) Try PDF link annotations.
    urls = _extract_urls_from_pdf_annotations(reader)

    # 3) Try URLs visible in extracted text.
    for i in range(min(3, len(reader.pages))):
        try:
            t = reader.pages[i].extract_text() or ""
        except Exception:
            t = ""
        urls.extend(_extract_urls_from_text(t))

    # Filter + de-dup.
    seen: set[str] = set()
    cleaned: list[str] = []
    for u in urls:
        u2 = u.strip()
        if not u2:
            continue
        if u2 in seen:
            continue
        seen.add(u2)

        pr = urlparse(u2)
        if pr.scheme not in {"http", "https"}:
            continue
        cleaned.append(u2)

    if not cleaned:
        return []

    title_words = {
        w
        for w in re.findall(r"[A-Za-z]{4,}", title.lower())
        if w not in {"guide", "ultimate", "deep", "learning", "towards"}
    }

    # Prefer URLs that look related to the title.
    def score(u: str) -> int:
        ul = u.lower()
        s = 0
        if "arxiv.org" in ul:
            s -= 50
        if "github.com" in ul:
            s += 10
        if ul.startswith("https://"):
            s += 2
        if "doi.org" in ul:
            s += 3

        # Reward overlap with title keywords.
        for w in title_words:
            if w in ul:
                s += 6

        return s

    cleaned.sort(key=score, reverse=True)
    return cleaned


def extract_arxiv_id(pdf: Path, reader: PdfReader) -> str | None:
    # Prefer from filename if it looks like an arXiv id.
    m = re.fullmatch(r"(\d{4}\.\d{5})(v\d+)?", pdf.stem)
    if m:
        return m.group(1) + (m.group(2) or "")

    # Otherwise, look for arXiv id on the first 2 pages.
    for page_i in range(min(2, len(reader.pages))):
        try:
            t = reader.pages[page_i].extract_text() or ""
        except Exception:
            continue
        arxiv_id = _extract_arxiv_id_from_text(t)
        if arxiv_id:
            return arxiv_id

    return None


def _safe_extract_tar(tar: tarfile.TarFile, dest: Path) -> None:
    dest_abs = dest.resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest_abs)):
            raise RuntimeError(f"Unsafe path in tar: {member.name}")
    tar.extractall(dest)


def download_arxiv_source(arxiv_id: str, cache_dir: Path, refresh: bool) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache by versionless id, but keep the requested id for download.
    base_id = re.sub(r"v\d+$", "", arxiv_id)
    out_dir = cache_dir / base_id

    if out_dir.exists() and any(out_dir.rglob("*.tex")) and not refresh:
        return out_dir

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://arxiv.org/e-print/{arxiv_id}"
    tar_path = out_dir / "source.tar"

    req = urllib.request.Request(url, headers={"User-Agent": "oz-pdf-to-markdown/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        tar_path.write_bytes(resp.read())

    with tarfile.open(tar_path, mode="r:*") as tf:
        _safe_extract_tar(tf, out_dir)

    return out_dir


def _score_tex_file(path: Path) -> int:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return -1

    score = 0
    if "\\begin{document}" in txt:
        score += 100
    if re.search(r"\\documentclass\b", txt):
        score += 40
    if re.search(r"\\title\s*\{", txt):
        score += 30
    # Prefer larger files (often main).
    score += min(30, int(path.stat().st_size / 20_000))
    return score


def choose_main_tex(arxiv_src_dir: Path) -> Path | None:
    candidates = [p for p in arxiv_src_dir.rglob("*.tex") if p.is_file()]

    # Exclude common junk directories (relative to the arXiv source dir).
    cleaned: list[Path] = []
    for p in candidates:
        rel_parts = p.relative_to(arxiv_src_dir).parts
        if "__MACOSX" in rel_parts:
            continue
        if any(part.startswith(".") for part in rel_parts):
            continue
        cleaned.append(p)

    candidates = cleaned

    if not candidates:
        return None

    scored = sorted(
        ((_score_tex_file(p), p) for p in candidates), reverse=True, key=lambda x: x[0]
    )
    best_score, best_path = scored[0]
    if best_score < 50:
        return None

    return best_path


def choose_bib(arxiv_src_dir: Path, main_tex: Path) -> Path | None:
    bibs = [p for p in arxiv_src_dir.rglob("*.bib") if p.is_file()]
    if not bibs:
        return None

    # Try to match \bibliography{foo,bar}.
    try:
        txt = main_tex.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        txt = ""

    m = re.search(r"\\bibliography\{([^}]+)\}", txt)
    if m:
        for name in [x.strip() for x in m.group(1).split(",") if x.strip()]:
            for b in bibs:
                if b.stem == name:
                    return b

    return sorted(bibs, key=lambda p: p.stat().st_size, reverse=True)[0]


def _pandoc_available() -> bool:
    return shutil.which("pandoc") is not None


def _convert_refs_html_to_markdown(html: str) -> str:
    if not html.strip():
        return ""

    if not _pandoc_available():
        # Leave as-is.
        return html

    with tempfile.TemporaryDirectory(prefix="oz-refs-") as td:
        in_path = Path(td) / "refs.html"
        in_path.write_text(html, encoding="utf-8")
        res = subprocess.run(
            ["pandoc", str(in_path), "-f", "html", "-t", "gfm", "--wrap=preserve"],
            check=False,
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            return html
        return res.stdout.strip() + "\n"


_LATEX_INCLUDE_RE = re.compile(r"\\(input|include)\{([^}]+)\}")


_LATEX_DOC_BOUNDARY_RE = re.compile(
    r"\\(documentclass|begin\{document\}|end\{document\})\b"
)


def _rewrite_tex_skip_primitives(line: str) -> str:
    # pandoc's LaTeX reader can choke on raw TeX primitives like:
    #   \vskip - 0.55 in
    # Rewrite to more standard LaTeX where possible.
    def _repl_skip(m: re.Match[str]) -> str:
        cmd = m.group(1)
        sign = (m.group(2) or "").replace(" ", "")
        num = m.group(3)
        unit = m.group(4)
        val = f"{sign}{num}{unit}"
        if cmd == "vskip":
            return f"\\vspace{{{val}}}"
        return f"\\hspace{{{val}}}"

    line = re.sub(
        r"\\(vskip|hskip)\s*([-+]?\s*)?([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)",
        _repl_skip,
        line,
    )
    return line


def _sanitize_latex_line(line: str, *, is_root: bool) -> str:
    # Drop document boundaries in included fragments; keep only for the root file.
    if (not is_root) and _LATEX_DOC_BOUNDARY_RE.search(line):
        return ""

    line = _rewrite_tex_skip_primitives(line)
    return line


def _ensure_single_end_document(tex: str) -> str:
    # Keep only the last \end{document}.
    matches = list(re.finditer(r"\\end\{document\}", tex))
    if len(matches) <= 1:
        return tex

    last = matches[-1].start()
    before = tex[:last]
    after = tex[last:]
    before = re.sub(r"\\end\{document\}\s*", "", before)
    return before + after


def _strip_environments(tex: str, envs: set[str]) -> str:
    # Line-based removal of \begin{env}...\end{env} blocks (including nested).
    begin_re = re.compile(r"\\begin\{([A-Za-z*]+)\}")
    end_re = re.compile(r"\\end\{([A-Za-z*]+)\}")

    out: list[str] = []
    stack: list[str] = []

    for line in tex.splitlines(True):
        mb = begin_re.search(line)
        if mb and mb.group(1) in envs:
            stack.append(mb.group(1))
            continue

        me = end_re.search(line)
        if me and stack and me.group(1) == stack[-1]:
            stack.pop()
            continue

        if stack:
            continue

        out.append(line)

    return "".join(out)


# Convert common LaTeX citation macros to pandoc citations so they survive even
# when we can't run citeproc due to broken .bib files.
_LATEX_CITE_RE = re.compile(r"\\cite\w*\*?\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]+)\}")


def _rewrite_latex_citations(s: str) -> str:
    def repl(m: re.Match[str]) -> str:
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        if not keys:
            return ""
        if len(keys) == 1:
            return f"[@{keys[0]}]"
        return "[" + "; ".join([f"@{k}" for k in keys]) + "]"

    return _LATEX_CITE_RE.sub(repl, s)


def _strip_latex_comment(line: str) -> str:
    out: list[str] = []
    prev = ""
    for ch in line:
        if ch == "%" and prev != "\\":
            break
        out.append(ch)
        prev = ch
    return "".join(out)


def _expand_latex_includes(path: Path, seen: set[Path], *, is_root: bool) -> str:
    path = path.resolve()
    if path in seen:
        return f"% [oz] skipped recursive include: {path.name}\n"
    seen.add(path)

    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return f"% [oz] missing include: {path.name}\n"

    out_lines: list[str] = []
    for line in raw.splitlines(True):
        # Keep a newline even if we strip the comment.
        stripped = _strip_latex_comment(line)
        stripped = _sanitize_latex_line(stripped, is_root=is_root)

        m = _LATEX_INCLUDE_RE.search(stripped)
        if not m:
            out_lines.append(_rewrite_latex_citations(stripped))
            continue

        cmd, rel = m.group(1), m.group(2).strip()
        # Support \\input{foo} and \\input{foo.tex}
        inc = Path(rel)
        if inc.suffix == "":
            inc = inc.with_suffix(".tex")
        inc_path = path.parent / inc

        # Replace the whole line with expanded content.
        out_lines.append(f"% [oz] begin {cmd}: {inc}\n")
        out_lines.append(_expand_latex_includes(inc_path, seen, is_root=False))
        out_lines.append(f"% [oz] end {cmd}: {inc}\n")

    tex = "".join(out_lines)
    if is_root:
        tex = _ensure_single_end_document(tex)
    return tex


def _pandoc_latex_to_md(
    expanded_tex: Path,
    *,
    out_dir: Path,
    main_tex_dir: Path,
    media_rel: Path,
    bibliography: Path | None,
    use_bibliography: bool,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "pandoc",
        str(expanded_tex),
        "-f",
        "latex",
        "-t",
        "gfm",
        "--wrap=preserve",
        "--shift-heading-level-by=1",
        "--resource-path",
        str(main_tex_dir),
        "--extract-media",
        str(media_rel),
    ]

    if use_bibliography and bibliography is not None:
        cmd += ["--citeproc", "--bibliography", str(bibliography)]

    return subprocess.run(
        cmd, check=False, capture_output=True, text=True, cwd=str(out_dir)
    )


def _postprocess_pandoc_markdown(md: str) -> str:
    # Strip some common HTML wrappers pandoc emits (safe, conservative).
    lines = md.splitlines()
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        if re.fullmatch(r"</?div[^>]*>", s):
            continue
        if re.fullmatch(r"</?span[^>]*>", s):
            continue
        if re.fullmatch(r"</?figure[^>]*>", s):
            continue

        m = re.match(r"^<embed\s+[^>]*src=\"([^\"]+)\"[^>]*/?>\s*$", s)
        if m:
            out.append(f"![]({m.group(1)})")
            continue

        out.append(ln)

    return "\n".join(out).strip() + "\n"


def latex_source_to_markdown(
    main_tex: Path,
    out_dir: Path,
    title: str,
    pdf_name: str,
    bibliography: Path | None,
) -> str:
    if not _pandoc_available():
        raise RuntimeError("pandoc not found")

    # Expand \\input / \\include so pandoc sees the whole document.
    expanded_tex = main_tex.parent / "__oz_expanded_main.tex"
    expanded_content = _expand_latex_includes(main_tex, seen=set(), is_root=True)
    expanded_tex.write_text(expanded_content, encoding="utf-8")

    slug = _slugify(title)

    bib_out: str | None = None
    if bibliography is not None:
        bib_dir = out_dir / "bib"
        bib_dir.mkdir(parents=True, exist_ok=True)
        bib_path = bib_dir / f"{slug}.bib"
        try:
            bib_path.write_bytes(bibliography.read_bytes())
            bib_out = str(bib_path.relative_to(out_dir))
        except Exception:
            bib_out = None

    media_rel = Path("media") / slug

    # 1) Try with bibliography (if present)
    res = _pandoc_latex_to_md(
        expanded_tex,
        out_dir=out_dir,
        main_tex_dir=main_tex.parent,
        media_rel=media_rel,
        bibliography=bibliography,
        use_bibliography=True,
    )

    # 2) If that fails, retry without bibliography/citeproc.
    if res.returncode != 0 and bibliography is not None:
        res2 = _pandoc_latex_to_md(
            expanded_tex,
            out_dir=out_dir,
            main_tex_dir=main_tex.parent,
            media_rel=media_rel,
            bibliography=bibliography,
            use_bibliography=False,
        )
        if res2.returncode == 0:
            res = res2

    # 3) If still failing, sanitize + retry.
    if res.returncode != 0:
        err = (res.stderr or "").strip()

        sanitized = expanded_content

        # If pandoc complains about table/tabular, strip them rather than failing the whole paper.
        if re.search(r"end\{(table|tabular)\}\b", err):
            sanitized = _strip_environments(
                sanitized,
                {"table", "table*", "tabular", "tabular*", "longtable"},
            )

        # If pandoc complains about unexpected end{document}, ensure we only keep the last one.
        if "end{document}" in err:
            sanitized = _ensure_single_end_document(sanitized)

        # Strip stray bare \end tokens.
        sanitized = re.sub(r"^\s*\\end\s*$", "", sanitized, flags=re.MULTILINE)

        sanitized_tex = main_tex.parent / "__oz_expanded_main.sanitized.tex"
        sanitized_tex.write_text(sanitized, encoding="utf-8")

        res3 = _pandoc_latex_to_md(
            sanitized_tex,
            out_dir=out_dir,
            main_tex_dir=main_tex.parent,
            media_rel=media_rel,
            bibliography=bibliography,
            use_bibliography=False,
        )
        if res3.returncode == 0:
            res = res3

        # 4) Last resort: strip a wider set of environments and retry.
        if res.returncode != 0:
            sanitized2 = _strip_environments(
                sanitized,
                {
                    "table",
                    "table*",
                    "tabular",
                    "tabular*",
                    "longtable",
                    "algorithm",
                    "algorithm*",
                    "lstlisting",
                },
            )
            sanitized2_tex = main_tex.parent / "__oz_expanded_main.sanitized2.tex"
            sanitized2_tex.write_text(sanitized2, encoding="utf-8")

            res4 = _pandoc_latex_to_md(
                sanitized2_tex,
                out_dir=out_dir,
                main_tex_dir=main_tex.parent,
                media_rel=media_rel,
                bibliography=bibliography,
                use_bibliography=False,
            )
            if res4.returncode == 0:
                res = res4

    # 5) If still failing, try converting the original main .tex without include expansion.
    if res.returncode != 0:
        res5 = _pandoc_latex_to_md(
            main_tex,
            out_dir=out_dir,
            main_tex_dir=main_tex.parent,
            media_rel=media_rel,
            bibliography=bibliography,
            use_bibliography=False,
        )
        if res5.returncode == 0:
            res = res5

    if res.returncode != 0:
        raise RuntimeError((res.stderr or "").strip() or "pandoc failed")

    body = _postprocess_pandoc_markdown(res.stdout)

    # If citeproc emitted HTML refs, convert that block to markdown.
    m = re.search(r"(<div id=\"refs\"[\s\S]*?</div>\s*</div>)", body)
    if m:
        refs_html = m.group(1)
        refs_md = _convert_refs_html_to_markdown(refs_html)
        replacement = "\n\n## References\n\n" + refs_md
        body = body.replace(refs_html, replacement)

    bib_comment = f"<!-- Bibliography: {bib_out} -->\n" if bib_out else ""

    return (
        f"# {title}\n\n"
        f"<!-- Source: {pdf_name} -->\n"
        f"<!-- Generated from: arXiv source (LaTeX) when available -->\n"
        f"{bib_comment}\n"
        f"{body.strip()}\n"
    )


_HEADING_WORDS = {
    "abstract",
    "introduction",
    "background",
    "conclusion",
    "acknowledgements",
    "acknowledgments",
    "references",
    "appendix",
    "contents",
}


def _is_page_number_line(s: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}", s))


def _inject_html_base_tag(html: str, base_url: str) -> str:
    hl = html.lower()
    if "<base" in hl:
        return html

    base_tag = f'<base href="{base_url}">'

    m = re.search(r"<head[^>]*>", html, flags=re.IGNORECASE)
    if m:
        i = m.end()
        return html[:i] + base_tag + html[i:]

    # No <head>: prepend a minimal one.
    return f"<head>{base_tag}</head>\n" + html


def _looks_like_blocked_or_error_page(md: str) -> bool:
    ml = md.lower().strip()
    if not ml:
        return True
    if "incapsula incident" in ml or "request unsuccessful" in ml:
        return True
    if len(ml) < 200:
        return True
    return False


def web_source_to_markdown(
    url: str,
    out_dir: Path,
    title: str,
    pdf_name: str,
) -> str:
    if not _pandoc_available():
        raise RuntimeError("pandoc not found")

    slug = _slugify(title)
    media_rel = Path("media") / slug

    with tempfile.TemporaryDirectory(prefix="oz-websrc-") as td:
        td_path = Path(td)

        req = urllib.request.Request(
            url, headers={"User-Agent": "oz-pdf-to-markdown/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                content = resp.read()
                ctype = (resp.headers.get("Content-Type") or "").lower()
        except urllib.error.URLError as e:
            raise RuntimeError(f"failed to fetch web source: {e}")

        # Heuristic input format.
        is_ipynb = url.lower().endswith(".ipynb") or "application/x-ipynb+json" in ctype
        in_path = td_path / ("doc.ipynb" if is_ipynb else "doc.html")

        if is_ipynb:
            in_path.write_bytes(content)
        else:
            html = content.decode("utf-8", errors="ignore")
            html = _inject_html_base_tag(html, base_url=url)
            in_path.write_text(html, encoding="utf-8")

        def run(extract_media: bool) -> subprocess.CompletedProcess[str]:
            cmd = [
                "pandoc",
                str(in_path),
                "-f",
                "ipynb" if is_ipynb else "html",
                "-t",
                "gfm",
                "--wrap=preserve",
                "--shift-heading-level-by=1",
            ]
            if extract_media:
                cmd += ["--extract-media", str(media_rel)]
            return subprocess.run(
                cmd, check=False, capture_output=True, text=True, cwd=str(out_dir)
            )

        res = run(extract_media=True)
        if res.returncode != 0:
            # Retry without fetching/extracting media (more robust for ad-heavy pages).
            res2 = run(extract_media=False)
            if res2.returncode == 0:
                res = res2

        if res.returncode != 0:
            raise RuntimeError(
                (res.stderr or "").strip() or "pandoc html conversion failed"
            )

        body = _postprocess_pandoc_markdown(res.stdout)

    if _looks_like_blocked_or_error_page(body):
        raise RuntimeError("web source looks like an error/blocked page")

    return (
        f"# {title}\n\n"
        f"<!-- Source: {pdf_name} -->\n"
        f"<!-- Generated from: web source ({url}) -->\n\n"
        f"{body.strip()}\n"
    )


def _is_heading_line(s: str) -> bool:
    s2 = s.strip()
    if not s2:
        return False

    # Common single-word section headers.
    if s2.lower() in _HEADING_WORDS:
        return True

    # Numbered headings: "1 Introduction", "3.2.1 Scaled Dot-Product Attention", etc.
    # Try to avoid false positives from table rows and years (e.g. "2014 English-French dataset...").
    m = re.match(r"^(\d+)(?:\.\d+)*\s+(.+)$", s2)
    if m:
        major = int(m.group(1))
        rest = m.group(2).strip()

        # Section numbers in papers are almost always small.
        if major > 50:
            return False

        # Reject table-like rows where the next token is numeric.
        first_token = rest.split()[0] if rest else ""
        if first_token and re.fullmatch(r"[0-9.]+", first_token):
            return False

        # Heuristic: headings are usually relatively short.
        if len(s2) > 120:
            return False

        return True

    return False


def _format_heading(s: str) -> str:
    # Preserve original casing (PDFs often use Title Case already).
    return f"## {s.strip()}"


def _is_caption_line(s: str) -> bool:
    return bool(re.match(r"^(Figure|Table)\s+\d+\s*[:.]\s+", s.strip()))


def clean_pdf_text(text: str) -> str:
    # Remove soft hyphen characters if present.
    text = text.replace("\u00ad", "")

    lines = [ln.rstrip() for ln in text.splitlines()]

    blocks: list[str] = []
    cur: list[str] = []

    def flush_paragraph() -> None:
        nonlocal cur
        if cur:
            blocks.append(_join_wrapped_lines(cur))
            cur = []

    for ln in lines:
        s = ln.strip()

        if not s:
            flush_paragraph()
            continue

        if _is_page_number_line(s):
            # Common in PDF extraction; doesn't add value in Markdown.
            continue

        if _is_heading_line(s):
            flush_paragraph()
            blocks.append(_format_heading(s))
            continue

        if _is_caption_line(s):
            flush_paragraph()
            blocks.append(f"**{s}**")
            continue

        cur.append(s)

    flush_paragraph()

    # Double-newline between paragraphs for Markdown readability.
    cleaned = "\n\n".join([b for b in blocks if b])
    return cleaned.strip() + "\n"


def reader_to_markdown(reader: PdfReader) -> str:
    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            parts.append(page_text)

    raw_text = "\n\n".join(parts)
    return clean_pdf_text(raw_text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert PDFs to Markdown (prefer arXiv LaTeX source when available; "
            "otherwise try to fetch a web/HTML source if the PDF contains a URL; "
            "fallback to PDF text extraction)."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="Literature",
        help="Input directory containing PDFs (default: Literature)",
    )
    parser.add_argument(
        "--out-dir",
        default="Literature/markdown",
        help="Output directory for generated .md files (default: Literature/markdown)",
    )
    parser.add_argument(
        "--no-arxiv",
        action="store_true",
        help="Disable downloading/converting from arXiv sources.",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable fetching/converting from web sources (HTML/ipynb).",
    )
    parser.add_argument(
        "--arxiv-cache-dir",
        default="Literature/.cache/arxiv",
        help="Cache directory for downloaded arXiv sources (default: Literature/.cache/arxiv)",
    )
    parser.add_argument(
        "--refresh-arxiv",
        action="store_true",
        help="Re-download arXiv sources even if they are present in cache.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable writing a conversion report under .reports/.",
    )
    parser.add_argument(
        "--report-path",
        default=".reports/pdf_to_markdown_report.json",
        help="Path for conversion report (default: .reports/pdf_to_markdown_report.json)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    in_dir = (project_root / args.input).resolve()
    out_dir = (project_root / args.out_dir).resolve()

    if not in_dir.exists() or not in_dir.is_dir():
        print(f"Input directory not found: {in_dir}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    existing_by_source = _index_existing_outputs(out_dir)

    pdfs = sorted(in_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in: {in_dir}", file=sys.stderr)
        return 0

    arxiv_cache_dir = (project_root / args.arxiv_cache_dir).resolve()

    report: dict[str, object] = {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "count": len(pdfs),
        "items": [],
    }

    for pdf in pdfs:
        reader = PdfReader(str(pdf))

        title = extract_pdf_title(reader, fallback=pdf.stem)
        base = _safe_filename(title) or pdf.stem
        md_path = out_dir / f"{base}.md"

        item: dict[str, object] = {
            "pdf": pdf.name,
            "title": title,
            "md": str(md_path),
            "pipeline": None,
            "arxiv_id": None,
            "web_url": None,
            "error": None,
        }

        # If this filename collides with a different source, disambiguate.
        if md_path.exists():
            try:
                head = md_path.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()[:40]
            except Exception:
                head = []

            src_match = None
            for ln in head:
                m = re.search(r"<!--\s*Source:\s*(.+?)\s*-->", ln)
                if m:
                    src_match = m.group(1)
                    break

            if src_match and src_match != pdf.name:
                # Prefer version suffix if it looks like an arXiv versioned filename.
                mver = re.search(r"v(\d+)$", pdf.stem)
                if mver:
                    md_path = out_dir / f"{base} (v{mver.group(1)}).md"
                else:
                    md_path = out_dir / f"{base} ({pdf.stem}).md"

        # Rename/remove previous generated outputs for this same PDF to avoid duplicates.
        prev_paths = existing_by_source.get(pdf.name, [])
        for prev in prev_paths:
            if prev == md_path:
                continue
            if not md_path.exists():
                prev.rename(md_path)
            else:
                prev.unlink(missing_ok=True)

        print(
            f"Converting: {pdf.name} -> {md_path.relative_to(project_root)}",
            file=sys.stderr,
        )

        md_text: str | None = None

        if not args.no_arxiv:
            arxiv_id = extract_arxiv_id(pdf, reader)
            item["arxiv_id"] = arxiv_id
            if arxiv_id:
                try:
                    src_dir = download_arxiv_source(
                        arxiv_id=arxiv_id,
                        cache_dir=arxiv_cache_dir,
                        refresh=args.refresh_arxiv,
                    )
                    main_tex = choose_main_tex(src_dir)
                    if main_tex is not None:
                        bib = choose_bib(src_dir, main_tex)
                        md_text = latex_source_to_markdown(
                            main_tex=main_tex,
                            out_dir=out_dir,
                            title=title,
                            pdf_name=pdf.name,
                            bibliography=bib,
                        )
                        item["pipeline"] = "arxiv"
                except Exception as e:
                    item["error"] = f"arxiv: {e}"
                    print(
                        f"  [warn] arXiv source conversion failed for {pdf.name}: {e}",
                        file=sys.stderr,
                    )

        if md_text is None and not args.no_web:
            try:
                urls = extract_web_source_urls(title, reader)
                # Try a few candidates in order; keep the first that converts.
                for url in urls[:3]:
                    try:
                        md_text = web_source_to_markdown(
                            url=url,
                            out_dir=out_dir,
                            title=title,
                            pdf_name=pdf.name,
                        )
                        item["web_url"] = url
                        item["pipeline"] = "web"
                        break
                    except Exception:
                        continue
            except Exception as e:
                if not item.get("error"):
                    item["error"] = f"web: {e}"
                print(
                    f"  [warn] web source conversion failed for {pdf.name}: {e}",
                    file=sys.stderr,
                )

        if md_text is None:
            body = reader_to_markdown(reader)
            md_text = f"# {title}\n\n" f"<!-- Source: {pdf.name} -->\n\n" f"{body}"
            item["pipeline"] = "pdf"

        md_path.write_text(md_text, encoding="utf-8")
        report["items"].append(item)

    if not args.no_report:
        report_path = (project_root / args.report_path).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
