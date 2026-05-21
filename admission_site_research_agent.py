#!/usr/bin/env python3
"""Agent-style university admissions site research.

This is the queue-driven version of admission_site_research.py.

Pipeline version:
    navigator -> topic worker -> leaf archive

Agent version:
    queue.pop -> fetch/archive -> LLM decision -> enqueue next links

Every topic has a fixed directory under data/:
    undergraduate, master, phd, non_degree, scholarships, faq

Each visited topic-specific node is stored as:
    data/<topic>/node_0001_<node_type>_<name>_<hash>/
      raw.html              # for HTML nodes
      decision.json         # LLM decision + node metadata
      source.json           # parent/source metadata
      source_pages/         # copied parent HTML when available
      files/                # downloaded PDFs/docs/xlsx/etc.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.parse import parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import admission_site_research as core


Topic = core.Topic
TOPICS = core.TOPICS

YEAR_PATTERN = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


@dataclass
class QueueItem:
    url: str
    topic: str = ""
    depth: int = 0
    label: str = ""
    purpose: str = ""
    parent_id: str = ""
    parent_url: str = ""
    parent_local_html: str = ""


@dataclass
class NodeRecord:
    node_id: str
    url: str
    final_url: str = ""
    topic: str = ""
    node_type: str = "unknown"
    depth: int = 0
    title: str = ""
    local_html: str = ""
    node_dir: str = ""
    files: list[dict[str, Any]] = field(default_factory=list)
    document_links: list[dict[str, Any]] = field(default_factory=list)
    source: dict[str, str] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def to_jsonable(value: Any) -> Any:
    if hasattr(core, "to_jsonable"):
        return core.to_jsonable(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value


class AgentStore(core.EvidenceStore):
    def __init__(self, output_dir: Path, verbose_trace: bool = True) -> None:
        super().__init__(output_dir, verbose_trace=verbose_trace)
        self.global_nodes_path = self.data_dir / "global_nodes.jsonl"

    def node_dir(self, topic: str, node_index: int, node_type: str, label: str, url: str) -> Path:
        topic = topic if topic in TOPICS else infer_topic(label, url)
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
        name = f"node_{node_index:04d}_{core.safe_name(node_type, 30)}_{core.safe_name(label or url, 70)}_{digest}"
        path = self.data_dir / topic / name
        (path / "files").mkdir(parents=True, exist_ok=True)
        (path / "source_pages").mkdir(parents=True, exist_ok=True)
        return path

    def write_global_node(self, record: NodeRecord) -> None:
        with self.global_nodes_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")

    def copy_parent_html(self, node_dir: Path, item: QueueItem) -> str:
        if not item.parent_local_html:
            return ""
        source = Path(item.parent_local_html)
        if not source.exists():
            return ""
        digest = hashlib.sha256((item.parent_url or item.parent_id or str(source)).encode("utf-8")).hexdigest()[:8]
        target = node_dir / "source_pages" / f"parent_{digest}.html"
        try:
            shutil.copyfile(source, target)
            return str(target)
        except Exception:
            return ""


def infer_topic(text: str, url: str = "") -> str:
    topic, score = core.best_topic(text, url)
    return topic if score > 0 else "undergraduate"


def allowed_years_for(target_year: int | None) -> set[int]:
    if not target_year:
        return set()
    return {target_year - 1, target_year}


def years_in_text(*values: str) -> set[int]:
    found: set[int] = set()
    for value in values:
        for match in YEAR_PATTERN.findall(str(value or "")):
            found.add(int(match))
    return found


def is_year_relevant(url: str, text: str, target_year: int | None, allow_years: set[int]) -> bool:
    if not target_year:
        return True
    years = years_in_text(url, text)
    if not years:
        return True
    return bool(years & allow_years)


def args_year_label(year: int | None) -> str:
    return str(year) if year else "latest"


def link_for_decision(link: core.Link) -> dict[str, Any]:
    return {
        "url": link.url,
        "text": core.compact(link.text, 120),
        "nearby_text": core.compact(link.nearby_text, 220),
        "region": core.compact(link.region, 80),
        "is_document": core.is_document(link.url),
    }


def relevant_links_for_agent(links: list[core.Link], limit: int) -> list[dict[str, Any]]:
    scored: list[tuple[int, int, core.Link]] = []
    for index, link in enumerate(links):
        text = f"{link.text} {link.nearby_text} {link.url}"
        score = max(core.topic_score(topic, text, link.url) for topic in TOPICS)
        lower = text.lower()
        if any(token in lower for token in ("admission", "program", "degree", "scholar", "faq", "application", "catalog", "brochure", "download", "招生", "项目", "奖学金", "简章", "目录")):
            score += 8
        if core.is_document(link.url):
            score += 10
        scored.append((score, -index, link))
    scored.sort(key=lambda item: item[:2], reverse=True)
    return [link_for_decision(link) for score, _index, link in scored if score > -8][:limit]


def filter_links_by_year(links: list[core.Link], target_year: int | None, allow_years: set[int]) -> list[core.Link]:
    if not target_year:
        return links
    return [link for link in links if is_year_relevant(link.url, f"{link.text} {link.nearby_text}", target_year, allow_years)]


def is_pagination_link(link: core.Link, current_url: str) -> bool:
    text = (link.text or "").strip().lower()
    nearby = (link.nearby_text or "").strip().lower()
    parsed = urlparse(link.url)
    current = urlparse(current_url)
    if parsed.path != current.path:
        return False
    query = parse_qs(parsed.query)
    current_query = parse_qs(current.query)
    page_value = (query.get("page") or query.get("p") or query.get("pageNo") or query.get("PageNo") or [""])[0]
    current_page_value = (current_query.get("page") or current_query.get("p") or current_query.get("pageNo") or current_query.get("PageNo") or [""])[0]
    if "page" in query or "p" in query or "pageNo" in query or "PageNo" in query:
        if page_value == "1" and not current_page_value:
            return False
        if parsed.query != current.query:
            return True
    if text in {"next", ">", ">>", "下一页", "下页", "后一页"}:
        return True
    if text.isdigit() and text not in {"0"}:
        return True
    if "page" in nearby and text.isdigit():
        return True
    return False


def pagination_links(page: core.Page, limit: int) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    seen: set[str] = set()
    current_url = page.final_url or page.url
    for link in page.links:
        if link.url in seen:
            continue
        if is_pagination_link(link, current_url):
            seen.add(link.url)
            links.append(
                {
                    "url": link.url,
                    "text": link.text or "pagination",
                    "nearby_text": link.nearby_text,
                    "is_document": False,
                    "reason": "deterministic pagination link",
                }
            )
    return links[:limit]


class NodeAnalyzer:
    def __init__(self, llm: core.JSONLLM, max_next_links: int, target_year: int | None, allow_years: set[int]) -> None:
        self.llm = llm
        self.max_next_links = max_next_links
        self.target_year = target_year
        self.allow_years = allow_years

    def analyze(self, item: QueueItem, page: core.Page) -> dict[str, Any]:
        payload = {
            "current": {
                "url": page.final_url or page.url or item.url,
                "title": page.title or item.label,
                "known_topic": item.topic,
                "depth": item.depth,
                "purpose": item.purpose,
                "text_excerpt": core.compact(page.main_text or page.text, 4500),
            },
            "target_topics": TOPICS,
            "target_year": self.target_year,
            "allowed_years": sorted(self.allow_years),
            "links": relevant_links_for_agent(filter_links_by_year(page.links, self.target_year, self.allow_years), self.max_next_links * 4),
            "pagination_links": pagination_links(page, 20),
            "rules": {
                "focus": "international student admissions only",
                "include": [
                    "admission brochures",
                    "program catalogs",
                    "application eligibility",
                    "application process",
                    "application timeline",
                    "tuition or fees",
                    "scholarship details",
                    "FAQ details",
                    "downloadable official files",
                ],
                "exclude": ["student life", "campus news", "housing-only", "visa-only", "alumni", "career"],
                "year_filter": f"Prefer admissions information for {sorted(self.allow_years)}. Skip links explicitly about other years unless they are generic pages without year-specific alternatives.",
            },
        }
        system = f"""You are the decision engine for an admissions research crawler.
For the current page, decide whether to archive it and which links to visit next.

Return strict JSON:
{{
  "node_type": "hub|topic|listing|detail|document|off_topic|error",
  "primary_topic": "undergraduate|master|phd|non_degree|scholarships|faq|null",
  "archive": true,
  "summary": "Chinese summary of what this page contributes",
  "important_facts": [{{"field":"programs|eligibility|timeline|fees|scholarships|documents|faq|contact|other","value":"fact","evidence":"short quote"}}],
  "next_links": [
    {{
      "url": "input URL only",
      "topic": "undergraduate|master|phd|non_degree|scholarships|faq",
      "action": "expand|download|archive",
      "purpose": "admission_brochure|program_catalog|application_process|timeline|scholarship_detail|faq_detail|document|other",
      "priority": 0.0,
      "reason": "why this link should be visited"
    }}
  ],
  "pagination_links": [
    {{
      "url": "input pagination URL only",
      "topic": "undergraduate|master|phd|non_degree|scholarships|faq",
      "action": "expand",
      "purpose": "pagination",
      "priority": 1.0,
      "reason": "next/list page"
    }}
  ],
  "stop_reason": "why no more links are needed, if applicable"
}}

Rules:
- A detail page can still return next_links if the actual brochure/catalog/details are linked from it.
- Always include relevant pagination links from pagination_links for listing pages so later pages are visited.
- Prefer deeper official admission evidence over navigation-only pages.
- Target year is {self.target_year or 'current/latest'}; allowed publication/admission years are {sorted(self.allow_years) or 'not constrained'}.
- Skip links whose title or URL explicitly targets years outside the allowed years, unless the page is a generic hub/listing with no year in the title/URL.
- Do not return links about student life, news, housing-only, visa-only, alumni, or careers.
- Return at most {self.max_next_links} next_links.
- Preserve input URLs exactly. Do not invent URLs."""
        parsed = self.llm.invoke(system, payload, max_tokens=4096)
        return self._normalize(parsed, page)

    def _normalize(self, parsed: dict[str, Any], page: core.Page) -> dict[str, Any]:
        input_urls = {link.url for link in page.links}
        primary_topic = parsed.get("primary_topic") or ""
        if primary_topic not in TOPICS:
            primary_topic = ""
        links: list[dict[str, Any]] = []
        for raw in parsed.get("next_links") or []:
            url = core.normalize_url(raw.get("url", ""))
            topic = raw.get("topic") or primary_topic
            if url not in input_urls or topic not in TOPICS:
                continue
            text = next((link.text for link in page.links if link.url == url), "")
            if not is_year_relevant(url, f"{text} {raw.get('reason', '')} {raw.get('purpose', '')}", self.target_year, self.allow_years):
                continue
            links.append(
                {
                    "url": url,
                    "topic": topic,
                    "action": raw.get("action") or ("download" if core.is_document(url) else "expand"),
                    "purpose": raw.get("purpose") or "",
                    "priority": float(raw.get("priority") or 0),
                    "reason": raw.get("reason") or "",
                    "text": text,
                }
            )
        links.sort(key=lambda link: link["priority"], reverse=True)
        return {
            "node_type": parsed.get("node_type") or "unknown",
            "primary_topic": primary_topic,
            "archive": bool(parsed.get("archive", True)),
            "summary": parsed.get("summary") or "",
            "important_facts": parsed.get("important_facts") or [],
            "next_links": links[: self.max_next_links],
            "pagination_links": self._normalize_extra_links(parsed.get("pagination_links") or [], page, primary_topic, "pagination")[:20],
            "stop_reason": parsed.get("stop_reason") or "",
        }

    def _normalize_extra_links(self, raw_links: list[dict[str, Any]], page: core.Page, primary_topic: str, purpose: str) -> list[dict[str, Any]]:
        input_urls = {link.url for link in page.links}
        normalized: list[dict[str, Any]] = []
        for raw in raw_links:
            url = core.normalize_url(raw.get("url", ""))
            topic = raw.get("topic") or primary_topic
            if url not in input_urls or topic not in TOPICS:
                continue
            text = next((link.text for link in page.links if link.url == url), "")
            if not is_year_relevant(url, f"{text} {raw.get('reason', '')} {raw.get('purpose', '')}", self.target_year, self.allow_years):
                continue
            normalized.append(
                {
                    "url": url,
                    "topic": topic,
                    "action": raw.get("action") or "expand",
                    "purpose": raw.get("purpose") or purpose,
                    "priority": float(raw.get("priority") or 1),
                    "reason": raw.get("reason") or purpose,
                    "text": text,
                }
            )
        return normalized


class AgentCrawler:
    def __init__(self, args: argparse.Namespace, roots: list[str], store: AgentStore, fetcher: core.HTTPFetcher, analyzer: NodeAnalyzer) -> None:
        self.args = args
        self.roots = roots
        self.store = store
        self.fetcher = fetcher
        self.analyzer = analyzer
        self.queue: list[QueueItem] = []
        self.visited: set[str] = set()
        self.records: list[NodeRecord] = []
        self.node_counter = 0
        self.allow_years = allowed_years_for(args.year)

    def enqueue(self, item: QueueItem) -> None:
        url = core.normalize_url(item.url)
        if not url or url in self.visited:
            return
        if urlparse(url).scheme not in {"http", "https"}:
            return
        if not core.allowed(url, self.roots):
            return
        if not is_year_relevant(url, f"{item.label} {item.purpose}", self.args.year, self.allow_years):
            self.store.trace("agent_skip_year", url=url, label=item.label, year=args_year_label(self.args.year), allowed_years=sorted(self.allow_years))
            return
        item.url = url
        self.queue.append(item)
        self.store.trace("agent_enqueue", url=item.url, topic=item.topic, depth=item.depth, label=item.label, purpose=item.purpose)

    def run(self) -> list[NodeRecord]:
        self.enqueue(QueueItem(url=self.args.entrypoint, topic="", depth=0, label="entrypoint", purpose="seed"))
        while self.queue and len(self.records) < self.args.max_nodes:
            item = self.queue.pop(0)
            if item.url in self.visited:
                continue
            self.visited.add(item.url)
            if item.depth > self.args.max_depth:
                continue
            try:
                record = self.process(item)
                self.records.append(record)
            except Exception as exc:
                self.store.trace("agent_node_error", url=item.url, error=str(exc))
        return self.records

    def process(self, item: QueueItem) -> NodeRecord:
        self.node_counter += 1
        if core.is_document(item.url):
            return self.process_document(item)
        page = self.fetcher.fetch(item.url)
        if page.error.startswith("non_html_content_type"):
            return self.process_document(item)

        decision = self.analyzer.analyze(item, page)
        topic = decision.get("primary_topic") or item.topic or infer_topic(f"{page.title} {page.main_text[:800]}", page.final_url or page.url)
        node_dir = self.store.node_dir(topic, self.node_counter, decision.get("node_type") or "unknown", page.title or item.label, page.final_url or item.url)
        if page.local_html:
            target_html = node_dir / "raw.html"
            shutil.copyfile(page.local_html, target_html)
            page.local_html = str(target_html)
        source = self.write_source(node_dir, item)
        document_links = self.document_links_from_decision(decision)
        files = self.download_page_documents(page, node_dir, topic, document_links)
        record = NodeRecord(
            node_id=f"node_{self.node_counter:04d}",
            url=item.url,
            final_url=page.final_url or page.url,
            topic=topic,
            node_type=decision.get("node_type") or "unknown",
            depth=item.depth,
            title=page.title or item.label,
            local_html=page.local_html,
            node_dir=str(node_dir),
            files=files,
            document_links=document_links,
            source=source,
            decision=decision,
            error=page.error,
        )
        self.store.write_json(node_dir / "decision.json", {"queue_item": item, "record": record})
        self.store.trace("agent_node", node_id=record.node_id, topic=topic, node_type=record.node_type, url=record.final_url, next_links=len(decision.get("next_links") or []), files=len(files))
        self.enqueue_next(record, page)
        return record

    def process_document(self, item: QueueItem) -> NodeRecord:
        if item.parent_id:
            raise RuntimeError("Unexpected queued document with parent; documents should be saved under the parent node files/")
        topic = item.topic if item.topic in TOPICS else infer_topic(item.label, item.url)
        self.node_counter += 0
        node_dir = self.store.node_dir(topic, self.node_counter, "document", item.label or Path(urlparse(item.url).path).name, item.url)
        source = self.write_source(node_dir, item)
        file_item = self.fetcher.download(item.url, item.label, target_dir=node_dir / "files")
        record = NodeRecord(
            node_id=f"node_{self.node_counter:04d}",
            url=item.url,
            final_url=file_item.get("url") or item.url,
            topic=topic,
            node_type="document",
            depth=item.depth,
            title=item.label or Path(urlparse(item.url).path).name,
            node_dir=str(node_dir),
            files=[file_item],
            source=source,
            decision={"node_type": "document", "primary_topic": topic, "archive": True, "summary": "Downloaded official document.", "next_links": []},
            error=file_item.get("error", ""),
        )
        self.store.write_json(node_dir / "decision.json", {"queue_item": item, "record": record})
        self.store.trace("agent_node", node_id=record.node_id, topic=topic, node_type="document", url=record.final_url, next_links=0, files=1)
        return record

    def write_source(self, node_dir: Path, item: QueueItem) -> dict[str, str]:
        source = {
            "parent_id": item.parent_id,
            "parent_url": item.parent_url,
            "parent_local_html": item.parent_local_html,
            "label": item.label,
            "purpose": item.purpose,
        }
        if item.parent_local_html and Path(item.parent_local_html).exists():
            copied = self.store.copy_parent_html(node_dir, item)
            if copied:
                source["copied_parent_html"] = copied
        self.store.write_json(node_dir / "source.json", source)
        return source

    def document_links_from_decision(self, decision: dict[str, Any]) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for link in decision.get("next_links") or []:
            url = link.get("url", "")
            if not core.is_document(url) or url in seen:
                continue
            seen.add(url)
            docs.append(link)
        return docs

    def download_page_documents(self, page: core.Page, node_dir: Path, topic: str, decision_document_links: list[dict[str, Any]]) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        seen: set[str] = set()
        for doc in decision_document_links:
            if len(files) >= self.args.max_attachments_per_node:
                break
            url = doc.get("url", "")
            if not core.is_document(url) or url in seen:
                continue
            seen.add(url)
            files.append(self.fetcher.download(url, doc.get("text") or doc.get("purpose") or url, target_dir=node_dir / "files"))
        for link in page.links:
            if len(files) >= self.args.max_attachments_per_node:
                break
            if not core.is_document(link.url) or link.url in seen:
                continue
            seen.add(link.url)
            files.append(self.fetcher.download(link.url, link.text, target_dir=node_dir / "files"))
        return files

    def enqueue_next(self, record: NodeRecord, page: core.Page) -> None:
        if record.depth >= self.args.max_depth:
            return
        combined_links = []
        seen: set[str] = set()
        for link in (record.decision.get("pagination_links") or []) + (record.decision.get("next_links") or []):
            if link.get("url") in seen:
                continue
            if core.is_document(link.get("url", "")):
                continue
            seen.add(link.get("url"))
            combined_links.append(link)
        deterministic_pages = [
            item
            for item in pagination_links(page, self.args.max_pagination_links)
            if is_year_relevant(item["url"], item.get("text", ""), self.args.year, self.allow_years)
        ]
        for page_link in deterministic_pages:
            if page_link["url"] in seen:
                continue
            seen.add(page_link["url"])
            combined_links.insert(
                0,
                {
                    "url": page_link["url"],
                    "topic": record.topic,
                    "action": "expand",
                    "purpose": "pagination",
                    "priority": 1.0,
                    "reason": page_link.get("reason") or "pagination",
                    "text": page_link.get("text") or "pagination",
                },
            )
        if deterministic_pages:
            self.store.trace("agent_pagination_links", node_id=record.node_id, topic=record.topic, count=len(deterministic_pages), urls=[item["url"] for item in deterministic_pages])
        for link in combined_links:
            self.enqueue(
                QueueItem(
                    url=link["url"],
                    topic=link.get("topic") or record.topic,
                    depth=record.depth + 1,
                    label=link.get("text") or link.get("purpose") or link["url"],
                    purpose=link.get("purpose") or link.get("action") or "",
                    parent_id=record.node_id,
                    parent_url=record.final_url,
                    parent_local_html=record.local_html,
                )
            )


def render_report(output_dir: Path, records: list[NodeRecord]) -> str:
    lines = ["# Agent Admission Site Research", ""]
    by_topic: dict[str, list[NodeRecord]] = {topic: [] for topic in TOPICS}
    for record in records:
        if record.topic in by_topic:
            by_topic[record.topic].append(record)
    for topic, topic_records in by_topic.items():
        lines += [f"## {TOPICS[topic]}", ""]
        if not topic_records:
            lines.append("- No nodes archived.")
            lines.append("")
            continue
        for record in topic_records:
            lines += [
                f"### {record.title or record.final_url}",
                f"- Node: {record.node_id}",
                f"- Type: {record.node_type}",
                f"- URL: {record.final_url}",
                f"- Directory: {record.node_dir}",
            ]
            if record.local_html:
                lines.append(f"- HTML: {record.local_html}")
            if record.source.get("copied_parent_html"):
                lines.append(f"- Source page: {record.source['copied_parent_html']}")
            if record.files:
                lines.append("- Files:")
                for item in record.files:
                    lines.append(f"  - {item.get('label') or item.get('url')}: {item.get('error') or item.get('local_path')}")
            if record.document_links:
                lines.append(f"- Document links from decision: {len(record.document_links)}")
            summary = record.decision.get("summary")
            if summary:
                lines.append(f"- Summary: {summary}")
            lines.append("")
    path = output_dir / "report.md"
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return str(path)


async def run(args: argparse.Namespace) -> None:
    core.DEBUG_OUTPUT_CHARS = args.debug_output_chars
    roots = [core.root_domain(args.entrypoint), *[core.root_domain(item) for item in args.allowed_domain]]
    roots = list(dict.fromkeys(root for root in roots if root))
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/admission_site_research_agent") / core.safe_name(args.university) / str(args.year or "latest") / core.stamp()
    store = AgentStore(output_dir, verbose_trace=not args.quiet_trace)
    core.DEBUG_OUTPUT_DIR = store.data_dir / "debug_model_outputs"
    store.write_json("run_metadata.json", {"mode": "agent", "university": args.university, "year": args.year, "entrypoint": args.entrypoint, "roots": roots, "topics": TOPICS, "created_at": core.now()})
    fetcher = core.HTTPFetcher(roots, store, max_download_bytes=args.max_download_mb * 1024 * 1024)
    llm = core.JSONLLM(args.llm_model, args.llm_base_url, os.getenv(args.llm_api_key_env), args.llm_max_tokens)
    if not llm.enabled:
        raise SystemExit("LLM is required. Configure --llm-provider or --llm-base-url/--llm-api-key-env/--llm-model.")
    try:
        analyzer = NodeAnalyzer(llm, args.max_next_links, args.year, allowed_years_for(args.year))
        crawler = AgentCrawler(args, roots, store, fetcher, analyzer)
        records = crawler.run()
        final = {"records": records, "topics": TOPICS}
        store.write_json("final_result.json", final)
        report = render_report(store.output_dir, records)
        print(json.dumps({"output_dir": str(store.output_dir), "report": report, "final_result": str(store.data_dir / "final_result.json")}, ensure_ascii=False, indent=2))
    finally:
        fetcher.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-style admissions research crawler.")
    parser.add_argument("--university", required=True)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--allowed-domain", action="append", default=[])
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-nodes", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-next-links", type=int, default=12)
    parser.add_argument("--max-pagination-links", type=int, default=20)
    parser.add_argument("--max-attachments-per-node", type=int, default=20)
    parser.add_argument("--max-download-mb", type=int, default=50)
    parser.add_argument("--quiet-trace", action="store_true")
    parser.add_argument("--debug-output-chars", type=int, default=20000)
    parser.add_argument("--llm-provider", choices=["custom", "openai", "openrouter", "aihubmix"], default=os.getenv("ADMISSION_LLM_PROVIDER", "custom"))
    parser.add_argument("--openrouter", action="store_true", default=os.getenv("OPENROUTER", "").lower() in {"1", "true", "yes"})
    parser.add_argument("--aihubmix", action="store_true", default=os.getenv("AIHUBMIX", "").lower() in {"1", "true", "yes"})
    parser.add_argument("--openrouter-api-key-env", default=os.getenv("OPENROUTER_API_KEY_ENV", "OPENROUTER_API_KEY"))
    parser.add_argument("--aihubmix-api-key-env", default=os.getenv("AIHUBMIX_API_KEY_ENV", "AIHUBMIX_API_KEY"))
    parser.add_argument("--llm-model", default=os.getenv("ADMISSION_RESEARCH_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-mini")))
    parser.add_argument("--llm-base-url", default=os.getenv("ADMISSION_RESEARCH_BASE_URL", os.getenv("OPENAI_BASE_URL", "")))
    parser.add_argument("--llm-api-key-env", default=os.getenv("ADMISSION_RESEARCH_API_KEY_ENV", ""))
    parser.add_argument("--llm-max-tokens", type=int, default=int(os.getenv("ADMISSION_RESEARCH_MAX_TOKENS", "4096")))
    args = parser.parse_args(argv)
    if args.openrouter:
        args.llm_provider = "openrouter"
    if args.aihubmix:
        args.llm_provider = "aihubmix"
    if args.llm_provider in core.LLM_PROVIDER_PRESETS:
        preset = core.LLM_PROVIDER_PRESETS[args.llm_provider]
        args.llm_base_url = args.llm_base_url or preset["base_url"]
        if not args.llm_api_key_env:
            if args.llm_provider == "openrouter":
                args.llm_api_key_env = args.openrouter_api_key_env
            elif args.llm_provider == "aihubmix":
                args.llm_api_key_env = args.aihubmix_api_key_env
            else:
                args.llm_api_key_env = str(preset["api_key_env"])
    elif not args.llm_api_key_env:
        args.llm_api_key_env = "OPENAI_API_KEY"
    return args


def main() -> None:
    asyncio.run(run(parse_args(sys.argv[1:])))


if __name__ == "__main__":
    main()
