import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

import aiohttp
import requests

from astrbot.api import logger


DEFAULT_CONNECT_TIMEOUT = 10
DEFAULT_READ_TIMEOUT = 180
DEFAULT_REQUEST_TIMEOUT = (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)
DEFAULT_RETRY_TOTAL = 3
DEFAULT_RETRY_BACKOFF = 1.0
LORA_MANIFEST_SUFFIX = ".lora.json"
LORA_CONTEXT_CACHE_SECONDS = 60
LORA_FILE_EXTENSIONS = {
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".pt2",
    ".pkl",
    ".sft",
}
LORA_CLEAR_DEFAULTS_TOKEN = "!clear_defaults"
_HTTP_SESSION = None


def _normalize_server_address(server_address):
    server_address = (server_address or "").strip()
    if not server_address:
        return server_address
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", server_address):
        server_address = f"http://{server_address}"
    return server_address.rstrip("/")


def _coerce_timeout(value):
    if value is None or value == "":
        return DEFAULT_REQUEST_TIMEOUT
    if isinstance(value, (int, float)):
        value = max(float(value), 0.1)
        return (min(value, DEFAULT_CONNECT_TIMEOUT), value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        connect_timeout = max(float(value[0]), 0.1)
        read_timeout = max(float(value[1]), 0.1)
        return (connect_timeout, read_timeout)
    return DEFAULT_REQUEST_TIMEOUT


def _build_http_session():
    global _HTTP_SESSION
    if _HTTP_SESSION is not None:
        return _HTTP_SESSION
    session = requests.Session()
    try:
        from requests.adapters import HTTPAdapter
        try:
            from urllib3.util.retry import Retry
        except Exception:
            from requests.packages.urllib3.util.retry import Retry
        retry = Retry(
            total=DEFAULT_RETRY_TOTAL,
            connect=DEFAULT_RETRY_TOTAL,
            read=DEFAULT_RETRY_TOTAL,
            status=DEFAULT_RETRY_TOTAL,
            backoff_factor=DEFAULT_RETRY_BACKOFF,
            status_forcelist=(408, 409, 425, 429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
    except Exception:
        pass
    _HTTP_SESSION = session
    return session


def _http_request(method, url, **kwargs):
    session = _build_http_session()
    timeout = _coerce_timeout(kwargs.pop("timeout", None))
    return session.request(method=method, url=url, timeout=timeout, **kwargs)


def _http_get(url, **kwargs):
    return _http_request("GET", url, **kwargs)


def _http_post(url, **kwargs):
    return _http_request("POST", url, **kwargs)


class ComfyUI:
    def __init__(self, config: dict, data_dir: Path = None) -> None:
        """
        Initialize the ComfyUI API client.

        Args:
            config: plugin config dict
            data_dir: persistent plugin data directory
        """
        self.server_address = _normalize_server_address(config.get("server_address", "127.0.0.1:8188"))
        if self.server_address.startswith("http"):
            self.url = self.server_address
        elif "." in self.server_address and ":" not in self.server_address:
            self.url = f"https://{self.server_address}"
        else:
            self.url = f"http://{self.server_address}"

        sub_conf = config.get("sub_config", {})
        self.steps = sub_conf.get("steps", 20)
        self.width = sub_conf.get("width", 768)
        self.height = sub_conf.get("height", 1024)
        self.neg_prompt = sub_conf.get("negative_prompt", "")

        wf_conf = config.get("workflow_settings", {})
        self.wf_filename = wf_conf.get("json_file", "workflow_api.json")
        self.input_id = str(wf_conf.get("input_node_id", "6"))
        self.neg_node_id = str(wf_conf.get("neg_node_id", ""))
        self.output_id = str(wf_conf.get("output_node_id", "9"))
        self.seed_id = None
        try:
            self.gen_timeout = max(30, int(wf_conf.get("timeout_seconds", 1800) or 1800))
        except (TypeError, ValueError):
            self.gen_timeout = 1800

        llm_conf = config.get("llm_settings", {}) or {}
        lora_conf = llm_conf.get("lora_control", {}) or {}
        self.lora_control_enabled = bool(lora_conf.get("enabled", False))
        self.inject_lora_catalog = bool(lora_conf.get("inject_catalog_into_system_prompt", True))
        self.inject_selected_lora_hints = bool(lora_conf.get("inject_selected_prompt_hints", True))
        self.keep_workflow_defaults_when_selected = bool(
            lora_conf.get("keep_workflow_defaults_when_selected", True)
        )
        try:
            self.max_lora_count = max(1, int(lora_conf.get("max_lora_count", 4) or 4))
        except (TypeError, ValueError):
            self.max_lora_count = 4
        self.lora_scan_roots = self._parse_lora_scan_roots(lora_conf.get("scan_directories", ""))
        try:
            self.max_trigger_options_per_lora = max(
                1, int(lora_conf.get("max_trigger_options_per_lora", 3) or 3)
            )
        except (TypeError, ValueError):
            self.max_trigger_options_per_lora = 3
        try:
            self.trigger_option_preview_length = max(
                24, int(lora_conf.get("trigger_option_preview_length", 80) or 80)
            )
        except (TypeError, ValueError):
            self.trigger_option_preview_length = 80

        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            logger.warning("[ComfyUI API] data_dir missing, fallback to plugin dir")

        self.workflow_dir = self.data_dir / "workflow"
        self.workflow_path = self.workflow_dir / self.wf_filename
        self._lora_context_cache = None
        self._lora_context_cache_key = None
        self._lora_context_cache_at = 0.0
        self._lora_scan_cache = None
        self._lora_scan_cache_key = None
        self._lora_scan_cache_at = 0.0

        logger.info(
            f"[ComfyUI API] loaded | workflow_dir: {self.workflow_dir} | workflow: {self.wf_filename}"
        )
        if self.lora_scan_roots:
            logger.info(
                "[ComfyUI API] LoRA scan roots: %s",
                ", ".join(str(path) for path in self.lora_scan_roots),
            )

    def reload_config(self, filename: str, input_id: str = None, output_id: str = None, neg_node_id: str = None):
        """Hot-switch workflow without restarting the plugin."""
        self.wf_filename = filename
        self.workflow_path = self.workflow_dir / filename
        self._clear_lora_caches()

        if input_id:
            self.input_id = str(input_id)
        if output_id:
            self.output_id = str(output_id)
        if neg_node_id:
            self.neg_node_id = str(neg_node_id)

        exists = self.workflow_path.exists()
        status = "exists" if exists else "missing"

        logger.info(
            f"[ComfyUI] switch workflow -> {filename} [{status}] | "
            f"Input:{self.input_id} | Neg:{self.neg_node_id} | Output:{self.output_id or 'auto'}"
        )
        return exists, (
            f"已切换至 {filename}，文件状态：{status}\n"
            f"当前节点设置: Positive={self.input_id}, Negative={self.neg_node_id}, Output={self.output_id or '自动'}"
        )

    @staticmethod
    def _parse_lora_scan_roots(value) -> list:
        if isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        else:
            raw_items = re.split(r"[\r\n]+", str(value or ""))

        roots = []
        seen = set()
        for raw in raw_items:
            path_text = str(raw or "").strip().strip('"').strip("'")
            if not path_text:
                continue
            path = Path(path_text).expanduser()
            try:
                path = path.resolve()
            except Exception:
                path = Path(os.path.abspath(str(path)))
            key = os.path.normcase(str(path))
            if key in seen:
                continue
            seen.add(key)
            roots.append(path)
        return roots

    def _clear_lora_caches(self):
        self._lora_context_cache = None
        self._lora_context_cache_key = None
        self._lora_context_cache_at = 0.0
        self._lora_scan_cache = None
        self._lora_scan_cache_key = None
        self._lora_scan_cache_at = 0.0

    @staticmethod
    def _normalize_lora_key(value) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", "", str(value).replace("\\", "/")).casefold()

    @staticmethod
    def _coerce_lora_strength(value, default: float = 1.0) -> float:
        try:
            if value in ("", None):
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _format_lora_strength(value: float) -> str:
        number = round(float(value), 4)
        text = f"{number:.4f}".rstrip("0").rstrip(".")
        return text or "0"

    @staticmethod
    def _normalize_prompt_token(value) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value).strip()).casefold()

    @staticmethod
    def _read_json_file(path: Path):
        for encoding in ("utf-8", "utf-8-sig"):
            try:
                with open(path, "r", encoding=encoding) as f:
                    return json.load(f)
            except UnicodeError:
                continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return json.load(f)

    @staticmethod
    def _trim_preview_text(value: str, max_length: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip()).strip(",; ")
        if len(text) <= max_length:
            return text
        return text[: max(0, max_length - 1)].rstrip() + "…"

    @staticmethod
    def _extract_quoted_segments(value: str) -> list:
        text = str(value or "")
        segments = []
        for pattern in (r"“([^”]+)”", r'"([^"]+)"', r"'([^']+)'"):
            segments.extend(match.strip() for match in re.findall(pattern, text) if match.strip())
        return segments

    def _split_trigger_options(self, value) -> list:
        text = re.sub(r"\s+", " ", str(value or "").strip()).strip(",; ")
        if not text:
            return []

        quoted_segments = self._extract_quoted_segments(text)
        if quoted_segments:
            return quoted_segments
        return [text]

    def _expand_prompt_hint_tokens(self, values) -> list:
        expanded = []
        seen = set()
        for raw in values or []:
            pieces = re.split(r"[,;\n，；]+", str(raw or ""))
            if len(pieces) <= 1:
                pieces = [str(raw or "")]

            for piece in pieces:
                text = re.sub(r"\s+", " ", str(piece or "").strip()).strip(",; ")
                normalized = self._normalize_prompt_token(text)
                if not text or not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                expanded.append(text)
        return expanded

    def _iter_lora_lookup_keys(self, entry: dict):
        raw_values = [
            entry.get("display_name"),
            entry.get("name"),
            entry.get("workflow_name"),
            entry.get("loader_name"),
            entry.get("model_name"),
        ] + list(entry.get("aliases", []))

        seen = set()
        for raw in raw_values:
            text = str(raw or "").strip()
            if not text:
                continue

            variants = {text, text.replace("\\", "/")}
            try:
                path_like = Path(text.replace("\\", "/"))
                variants.add(path_like.name)
                variants.add(path_like.stem)
            except Exception:
                pass

            for variant in variants:
                normalized = self._normalize_lora_key(variant)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                yield normalized

    def _build_lora_lookup(self, catalog: dict) -> dict:
        lookup = {}
        for key, entry in catalog.items():
            for normalized in self._iter_lora_lookup_keys(entry):
                lookup.setdefault(normalized, key)
        return lookup

    @staticmethod
    def _build_lora_entry(
        display_name: str,
        *,
        workflow_name: str = "",
        loader_name: str = "",
        default_strength=None,
        default_clip_strength=None,
        default_active: bool = False,
    ) -> dict:
        return {
            "name": display_name,
            "display_name": display_name,
            "workflow_name": workflow_name or "",
            "loader_name": loader_name or "",
            "default_strength": default_strength,
            "default_clip_strength": default_clip_strength,
            "default_active": bool(default_active),
            "aliases": [],
            "description": "",
            "trigger_words": [],
            "prompt_hints": [],
            "trigger_options": [],
            "source_nodes": [],
            "model_name": "",
            "base_model": "",
            "tags": [],
            "is_style_lora": False,
        }

    def _merge_lora_entry(self, base: dict, incoming: dict):
        if not base.get("display_name") and incoming.get("display_name"):
            base["display_name"] = incoming["display_name"]
            base["name"] = incoming["display_name"]

        if incoming.get("workflow_name") and not base.get("workflow_name"):
            base["workflow_name"] = incoming["workflow_name"]
        if incoming.get("loader_name") and not base.get("loader_name"):
            base["loader_name"] = incoming["loader_name"]

        if incoming.get("default_active"):
            base["default_active"] = True
        if incoming.get("default_strength") not in (None, ""):
            base["default_strength"] = self._coerce_lora_strength(
                incoming.get("default_strength"), base.get("default_strength", 1.0)
            )
        if incoming.get("default_clip_strength") not in (None, ""):
            base["default_clip_strength"] = self._coerce_lora_strength(
                incoming.get("default_clip_strength"), base.get("default_clip_strength", 1.0)
            )

        for field in ("aliases", "trigger_words", "prompt_hints", "trigger_options", "source_nodes", "tags"):
            merged = [
                str(item).strip()
                for item in list(base.get(field, [])) + list(incoming.get(field, []))
                if str(item).strip()
            ]
            base[field] = list(dict.fromkeys(merged))

        if incoming.get("description") and not base.get("description"):
            base["description"] = str(incoming["description"]).strip()
        if incoming.get("model_name") and not base.get("model_name"):
            base["model_name"] = str(incoming["model_name"]).strip()
        if incoming.get("base_model") and not base.get("base_model"):
            base["base_model"] = str(incoming["base_model"]).strip()
        if incoming.get("is_style_lora"):
            base["is_style_lora"] = True

        return base

    def _get_lora_manifest_path(self) -> Path:
        return self.workflow_path.parent / f"{self.workflow_path.stem}{LORA_MANIFEST_SUFFIX}"

    def _find_lora_loader_nodes(self, workflow: dict):
        nodes = []
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            inputs = node_data.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            loras = inputs.get("loras")
            if not isinstance(loras, dict):
                continue
            entries = loras.get("__value__")
            if not isinstance(entries, list):
                continue
            if not any(isinstance(item, dict) and str(item.get("name", "")).strip() for item in entries):
                continue
            nodes.append((str(node_id), node_data))
        return nodes

    def _extract_lora_catalog(self, workflow: dict):
        catalog = {}
        loader_node_ids = []

        for node_id, node_data in self._find_lora_loader_nodes(workflow):
            loader_node_ids.append(node_id)
            entries = node_data.get("inputs", {}).get("loras", {}).get("__value__", [])
            for item in entries:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if not name:
                    continue

                entry = catalog.setdefault(
                    name,
                    self._build_lora_entry(
                        name,
                        workflow_name=name,
                        default_strength=self._coerce_lora_strength(item.get("strength"), 1.0),
                        default_clip_strength=self._coerce_lora_strength(
                            item.get("clipStrength"), item.get("strength") or 1.0
                        ),
                        default_active=bool(item.get("active", False)),
                    ),
                )

                if item.get("active"):
                    entry["default_active"] = True
                if node_id not in entry["source_nodes"]:
                    entry["source_nodes"].append(node_id)

        return {"catalog": catalog, "loader_node_ids": loader_node_ids}

    def _load_lora_manifest(self) -> dict:
        path = self._get_lora_manifest_path()
        if not path.exists():
            return {}

        try:
            data = self._read_json_file(path)
        except Exception as e:
            logger.warning(f"[ComfyUI] failed to load LoRA sidecar {path.name}: {e}")
            return {}

        if not isinstance(data, dict):
            return {}

        loras = data.get("loras", data)
        return loras if isinstance(loras, dict) else {}

    def _extract_metadata_trigger_options(self, metadata: dict) -> list:
        civitai = metadata.get("civitai", {}) or {}
        raw_values = civitai.get("trainedWords", []) or []

        options = []
        for raw in raw_values:
            options.extend(self._split_trigger_options(raw))

        cleaned = []
        seen = set()
        for item in options:
            text = re.sub(r"\s+", " ", str(item or "").strip()).strip(",; ")
            normalized = self._normalize_prompt_token(text)
            if not text or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(text)
        return cleaned

    def _scan_lora_catalog(self) -> list:
        cache_key = tuple(str(path) for path in self.lora_scan_roots)
        now = time.monotonic()
        if (
            self._lora_scan_cache_key == cache_key
            and self._lora_scan_cache is not None
            and now - self._lora_scan_cache_at < LORA_CONTEXT_CACHE_SECONDS
        ):
            return self._lora_scan_cache

        entries = []
        seen_files = set()

        for root in self.lora_scan_roots:
            if not root.exists() or not root.is_dir():
                logger.warning(f"[ComfyUI] LoRA scan root does not exist: {root}")
                continue

            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in LORA_FILE_EXTENSIONS:
                    continue

                resolved_path = os.path.normcase(str(path.resolve()))
                if resolved_path in seen_files:
                    continue
                seen_files.add(resolved_path)

                try:
                    loader_name = path.relative_to(root).as_posix()
                except ValueError:
                    loader_name = path.name

                metadata_path = path.with_suffix(".metadata.json")
                metadata = {}
                if metadata_path.exists():
                    try:
                        metadata = self._read_json_file(metadata_path)
                    except Exception as e:
                        logger.warning(f"[ComfyUI] failed to read metadata {metadata_path.name}: {e}")
                        metadata = {}

                file_name = str(metadata.get("file_name") or path.stem).strip() or path.stem
                model_name = str(metadata.get("model_name") or "").strip()
                notes = str(metadata.get("notes") or "").strip()
                base_model = str(metadata.get("base_model") or "").strip()
                civitai = metadata.get("civitai", {}) or {}
                civitai_model = civitai.get("model", {}) or {}
                tag_values = []
                for source in (metadata.get("tags", []), civitai_model.get("tags", [])):
                    tag_values.extend(str(item).strip() for item in source if str(item).strip())

                description = notes or model_name or str(civitai_model.get("name") or "").strip()
                trigger_options = self._extract_metadata_trigger_options(metadata)
                aliases = [path.stem]
                if model_name and model_name != file_name:
                    aliases.append(model_name)
                if loader_name and loader_name not in aliases:
                    aliases.append(loader_name)

                tag_text = {self._normalize_prompt_token(tag) for tag in tag_values}
                entry = self._build_lora_entry(file_name, loader_name=loader_name)
                entry["aliases"] = list(dict.fromkeys(alias for alias in aliases if alias))
                entry["description"] = description
                entry["model_name"] = model_name
                entry["base_model"] = base_model
                entry["tags"] = list(dict.fromkeys(tag_values))
                entry["trigger_options"] = trigger_options
                entry["is_style_lora"] = "style" in tag_text or self._normalize_prompt_token(model_name).endswith("/ style")
                entries.append(entry)

        self._lora_scan_cache = entries
        self._lora_scan_cache_key = cache_key
        self._lora_scan_cache_at = now
        return entries

    def _merge_scanned_lora_catalog(self, catalog: dict):
        if not self.lora_scan_roots:
            return catalog

        lookup = self._build_lora_lookup(catalog)

        for entry in self._scan_lora_catalog():
            matched_key = None
            for normalized in self._iter_lora_lookup_keys(entry):
                matched_key = lookup.get(normalized)
                if matched_key:
                    break

            if matched_key:
                self._merge_lora_entry(catalog[matched_key], entry)
            else:
                unique_key = entry.get("loader_name") or entry.get("display_name")
                suffix = 1
                while unique_key in catalog:
                    suffix += 1
                    unique_key = f"{entry.get('loader_name') or entry.get('display_name')}#{suffix}"
                catalog[unique_key] = entry

            lookup = self._build_lora_lookup(catalog)

        return catalog

    def _apply_manifest_metadata(self, catalog: dict):
        manifest = self._load_lora_manifest()
        if not manifest:
            return catalog

        lookup = self._build_lora_lookup(catalog)
        for raw_name, meta in manifest.items():
            if not isinstance(meta, dict):
                continue

            name = str(raw_name).strip()
            if not name:
                continue

            matched_key = lookup.get(self._normalize_lora_key(name))
            if matched_key:
                entry = catalog[matched_key]
            else:
                entry = self._build_lora_entry(name)
                catalog[name] = entry
                matched_key = name

            aliases = [
                str(alias).strip()
                for alias in meta.get("aliases", [])
                if str(alias).strip()
            ]
            trigger_words = [
                str(word).strip()
                for word in meta.get("trigger_words", [])
                if str(word).strip()
            ]
            prompt_hints = [
                str(word).strip()
                for word in meta.get("prompt_hints", [])
                if str(word).strip()
            ]

            incoming = self._build_lora_entry(
                entry.get("display_name") or name,
                workflow_name=entry.get("workflow_name", ""),
                loader_name=entry.get("loader_name", ""),
                default_strength=self._coerce_lora_strength(
                    meta.get("default_strength"), entry.get("default_strength", 1.0)
                ),
                default_clip_strength=self._coerce_lora_strength(
                    meta.get("default_clip_strength"), entry.get("default_clip_strength", 1.0)
                ),
                default_active=entry.get("default_active", False),
            )
            incoming["aliases"] = aliases
            incoming["description"] = str(meta.get("description") or "").strip()
            incoming["trigger_words"] = trigger_words
            incoming["prompt_hints"] = prompt_hints
            incoming["trigger_options"] = trigger_words
            self._merge_lora_entry(entry, incoming)
            lookup = self._build_lora_lookup(catalog)

        return catalog

    def get_lora_runtime_context(self) -> dict:
        cache_key = (
            str(self.workflow_path),
            tuple(str(path) for path in self.lora_scan_roots),
        )
        now = time.monotonic()
        if (
            self._lora_context_cache_key == cache_key
            and self._lora_context_cache is not None
            and now - self._lora_context_cache_at < LORA_CONTEXT_CACHE_SECONDS
        ):
            return self._lora_context_cache

        try:
            workflow = self._load_workflow()
        except Exception as e:
            logger.warning(f"[ComfyUI] failed to read workflow for LoRA context: {e}")
            return {
                "supported": False,
                "catalog": {},
                "loader_node_ids": [],
                "manifest_path": str(self._get_lora_manifest_path()),
            }

        extracted = self._extract_lora_catalog(workflow)
        catalog = extracted["catalog"]
        self._merge_scanned_lora_catalog(catalog)
        self._apply_manifest_metadata(catalog)

        for entry in catalog.values():
            entry["default_strength"] = self._coerce_lora_strength(entry.get("default_strength"), 1.0)
            entry["default_clip_strength"] = self._coerce_lora_strength(
                entry.get("default_clip_strength"), entry["default_strength"]
            )

        sorted_items = sorted(
            catalog.items(),
            key=lambda item: (
                not bool(item[1].get("default_active")),
                self._normalize_lora_key(item[1].get("display_name") or item[0]),
            ),
        )
        sorted_catalog = {key: value for key, value in sorted_items}
        default_active_loras = [entry for entry in sorted_catalog.values() if entry.get("default_active")]

        context = {
            "supported": bool(extracted["loader_node_ids"]),
            "catalog": sorted_catalog,
            "loader_node_ids": extracted["loader_node_ids"],
            "manifest_path": str(self._get_lora_manifest_path()),
            "scan_roots": [str(path) for path in self.lora_scan_roots],
            "default_active_loras": default_active_loras,
        }
        self._lora_context_cache = context
        self._lora_context_cache_key = cache_key
        self._lora_context_cache_at = now
        return context

    def get_lora_prompt_appendix(self) -> str:
        if not (self.lora_control_enabled and self.inject_lora_catalog):
            return ""

        context = self.get_lora_runtime_context()
        catalog = context.get("catalog", {})
        if not context.get("supported") or not catalog:
            return ""

        default_loras = context.get("default_active_loras", [])
        lines = [
            "--------------------------------------------------",
            "【五、可选 LoRA 堆控制】",
            "--------------------------------------------------",
            "",
            "当前工作流支持 LoRA 堆联动。",
            "如果当前画面确实需要特定 LoRA，请在对应的 `<pic prompt=\"...\">` 之前额外输出一个标签：",
            "`<lora picks=\"LoRA名:强度@触发词序号, 另一个LoRA名:强度:clip强度@1+2\">`",
            "",
            f"规则：最多选择 {self.max_lora_count} 个；只能从下面清单里挑；不需要时不要输出 `<lora picks>`。",
            f"如果想只在本次图片里禁用工作流默认 LoRA，请输出：`<lora picks=\"{LORA_CLEAR_DEFAULTS_TOKEN}\">`。",
            f"如果想禁用默认 LoRA 后改用别的 LoRA，请输出：`<lora picks=\"{LORA_CLEAR_DEFAULTS_TOKEN}, LoRA名:0.8@1\">`。",
            "这里的 `@1`、`@1+2` 表示使用该 LoRA 下方列出来的第 1 个或第 1+2 个触发词候选。",
            "如果某个 LoRA 没有触发词候选，就把它视为全局风格 LoRA，直接选 LoRA 本体即可。",
            "重点规则：只要你写了 `@序号`，就表示系统会自动把该编号对应的触发词注入最终 prompt。",
            "因此，`<pic prompt>` 里不要再手动复述同一组触发词，不要把角色名、发色、服装、固定特征整段再写一遍。",
            "选择了 `@序号` 后，`<pic prompt>` 只写额外画面需求，例如构图、动作、表情、镜头、环境、光照、氛围。",
            "只有在触发词没有覆盖到某个关键信息时，才额外补少量缺失 tags；不要原样照抄触发词全文。",
            "",
        ]

        if default_loras:
            lines.append("当前工作流默认启用的 LoRA：")
            for entry in default_loras:
                strength_text = self._format_lora_strength(entry.get("default_strength", 1.0))
                clip_text = self._format_lora_strength(
                    entry.get("default_clip_strength", entry.get("default_strength", 1.0))
                )
                lines.append(f"- {entry['display_name']} | default_on={strength_text}/{clip_text}")
        else:
            lines.append("当前工作流默认启用的 LoRA：无")

        lines.extend(["", "当前可用 LoRA 清单："])
        for entry in catalog.values():
            strength_text = self._format_lora_strength(entry.get("default_strength", 1.0))
            clip_strength = entry.get("default_clip_strength", entry.get("default_strength", 1.0))
            clip_text = self._format_lora_strength(clip_strength)
            default_flag = "on" if entry.get("default_active") else "off"
            alias_values = [
                alias
                for alias in entry.get("aliases", [])
                if self._normalize_lora_key(alias) != self._normalize_lora_key(entry.get("display_name"))
            ]
            alias_preview = ", ".join(alias_values[:2])
            trigger_options = entry.get("trigger_options", [])[: self.max_trigger_options_per_lora]
            trigger_preview = " ; ".join(
                f"[{idx}] {self._trim_preview_text(option, self.trigger_option_preview_length)}"
                for idx, option in enumerate(trigger_options, start=1)
            )

            details = [f"default={default_flag}:{strength_text}/{clip_text}"]
            if alias_preview:
                details.append(f"aliases={alias_preview}")
            if entry.get("model_name"):
                details.append(f"model={self._trim_preview_text(entry['model_name'], 48)}")
            if entry.get("base_model"):
                details.append(f"base={entry['base_model']}")
            if trigger_preview:
                details.append(f"triggers={trigger_preview}")
            elif entry.get("prompt_hints"):
                details.append(
                    "hints="
                    + " ; ".join(
                        self._trim_preview_text(text, self.trigger_option_preview_length)
                        for text in entry.get("prompt_hints", [])[: self.max_trigger_options_per_lora]
                    )
                )
            elif entry.get("is_style_lora"):
                details.append("triggers=无（更偏全局风格）")
            else:
                details.append("triggers=无")
            if entry.get("description"):
                details.append(f"note={self._trim_preview_text(entry['description'], 60)}")
            lines.append(f"- {entry['display_name']} | " + " | ".join(details))

        manifest_path = context.get("manifest_path")
        if manifest_path or context.get("scan_roots"):
            lines.extend(
                [
                    "",
                ]
            )
        if context.get("scan_roots"):
            lines.append("LoRA 扫描根目录：")
            for root in context.get("scan_roots", []):
                lines.append(f"- {root}")
        if manifest_path:
            lines.extend(
                [
                    f"工作流 sidecar 元数据文件：{manifest_path}",
                    "如果某个 LoRA 有额外 aliases、trigger_words、prompt_hints、description，请按这些信息理解。",
                ]
            )

        return "\n".join(lines).strip()

    def resolve_lora_selections(self, selections) -> list:
        if not selections:
            return []

        context = self.get_lora_runtime_context()
        catalog = context.get("catalog", {})
        if not context.get("supported") or not catalog:
            return []

        lookup = self._build_lora_lookup(catalog)

        resolved = []
        seen = set()
        clear_defaults = False
        for item in selections:
            if not isinstance(item, dict):
                continue

            raw_name = str(item.get("name", "")).strip()
            if not raw_name:
                continue

            normalized_name = self._normalize_lora_key(raw_name)
            if item.get("control") == "clear_defaults" or normalized_name == self._normalize_lora_key(LORA_CLEAR_DEFAULTS_TOKEN):
                clear_defaults = True
                continue

            matched_key = lookup.get(normalized_name)
            if not matched_key:
                logger.info(f"[ComfyUI] LoRA not found in current workflow: {raw_name}")
                continue
            if matched_key in seen:
                continue

            entry = catalog[matched_key]
            strength = self._coerce_lora_strength(item.get("strength"), entry.get("default_strength", 1.0))
            clip_strength = self._coerce_lora_strength(
                item.get("clip_strength"), entry.get("default_clip_strength", strength)
            )
            valid_trigger_indexes = []
            selected_prompt_hints = []
            for index in item.get("trigger_indexes", []) or []:
                try:
                    trigger_index = int(index)
                except (TypeError, ValueError):
                    continue
                if trigger_index < 1 or trigger_index > len(entry.get("trigger_options", [])):
                    continue
                valid_trigger_indexes.append(trigger_index)
                selected_prompt_hints.append(entry["trigger_options"][trigger_index - 1])

            if not selected_prompt_hints:
                if entry.get("prompt_hints"):
                    selected_prompt_hints = list(entry.get("prompt_hints", []))
                elif len(entry.get("trigger_options", [])) == 1:
                    selected_prompt_hints = [entry["trigger_options"][0]]

            resolved.append(
                {
                    "name": entry.get("workflow_name") or entry.get("loader_name") or entry.get("display_name"),
                    "display_name": entry.get("display_name") or raw_name,
                    "workflow_name": entry.get("workflow_name") or "",
                    "loader_name": entry.get("loader_name") or "",
                    "requested_name": raw_name,
                    "strength": strength,
                    "clip_strength": clip_strength,
                    "aliases": entry.get("aliases", []),
                    "trigger_words": entry.get("trigger_words", []),
                    "prompt_hints": entry.get("prompt_hints", []),
                    "trigger_options": entry.get("trigger_options", []),
                    "trigger_indexes": valid_trigger_indexes,
                    "selected_prompt_hints": selected_prompt_hints,
                    "description": entry.get("description", ""),
                }
            )
            seen.add(matched_key)

            if len(resolved) >= self.max_lora_count:
                break

        if clear_defaults:
            resolved.insert(0, {"name": LORA_CLEAR_DEFAULTS_TOKEN, "control": "clear_defaults"})
        return resolved

    def _inject_selected_lora_prompt_hints(self, prompt: str, selections: list):
        if not (self.inject_selected_lora_hints and selections):
            return prompt, []

        existing_tokens = {
            self._normalize_prompt_token(token)
            for token in str(prompt or "").split(",")
            if str(token).strip()
        }

        injected = []
        for item in selections:
            if item.get("control"):
                continue
            selected_tokens = self._expand_prompt_hint_tokens(
                item.get("selected_prompt_hints", []) or item.get("prompt_hints", [])
            )
            for token in selected_tokens:
                cleaned = str(token).strip()
                normalized = self._normalize_prompt_token(cleaned)
                if not cleaned or not normalized or normalized in existing_tokens:
                    continue
                injected.append(cleaned)
                existing_tokens.add(normalized)

        if not injected:
            return prompt, []

        prefix = ", ".join(injected)
        final_prompt = f"{prefix}, {str(prompt or '').strip()}".strip(", ").strip()
        return final_prompt, injected

    def _rebuild_lora_text(self, items: list, active_only: bool = True) -> str:
        tokens = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if active_only and not item.get("active", False):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue

            strength = self._format_lora_strength(self._coerce_lora_strength(item.get("strength"), 1.0))
            clip_strength = self._format_lora_strength(
                self._coerce_lora_strength(item.get("clipStrength"), item.get("strength") or 1.0)
            )
            if strength == clip_strength:
                tokens.append(f"<lora:{name}:{strength}>")
            else:
                tokens.append(f"<lora:{name}:{strength}:{clip_strength}>")

        return ", ".join(tokens)

    def _apply_lora_selections(self, workflow: dict, selections: list):
        if not selections:
            return []

        loader_nodes = self._find_lora_loader_nodes(workflow)
        if not loader_nodes:
            return []

        clear_defaults = any(item.get("control") == "clear_defaults" for item in selections if isinstance(item, dict))
        effective_selections = [item for item in selections if isinstance(item, dict) and not item.get("control")]
        selected_lookup = {}
        for selection in effective_selections:
            for candidate in (
                selection.get("name"),
                selection.get("workflow_name"),
                selection.get("loader_name"),
                selection.get("display_name"),
                selection.get("requested_name"),
            ):
                normalized = self._normalize_lora_key(candidate)
                if normalized:
                    selected_lookup.setdefault(normalized, selection)

        applied = []

        for _, node_data in loader_nodes:
            inputs = node_data.get("inputs", {})
            lora_container = inputs.get("loras", {})
            items = lora_container.get("__value__", [])
            if not isinstance(items, list):
                continue

            node_applied_keys = set()
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if not name:
                    continue

                selection = selected_lookup.get(self._normalize_lora_key(name))
                if selection:
                    item["active"] = True
                    item["strength"] = self._format_lora_strength(selection["strength"])
                    item["clipStrength"] = self._format_lora_strength(selection["clip_strength"])
                    node_applied_keys.add(self._normalize_lora_key(selection.get("name")))
                    label = selection.get("display_name") or name
                    if label not in applied:
                        applied.append(label)
                elif clear_defaults or not self.keep_workflow_defaults_when_selected:
                    item["active"] = False

            for selection in effective_selections:
                selection_key = self._normalize_lora_key(selection.get("name"))
                if selection_key in node_applied_keys:
                    continue
                runtime_name = (
                    selection.get("loader_name")
                    or selection.get("workflow_name")
                    or selection.get("name")
                )
                if not runtime_name:
                    continue
                items.append(
                    {
                        "name": runtime_name,
                        "strength": self._format_lora_strength(selection["strength"]),
                        "active": True,
                        "expanded": False,
                        "clipStrength": self._format_lora_strength(selection["clip_strength"]),
                    }
                )
                node_applied_keys.add(selection_key)
                label = selection.get("display_name") or runtime_name
                if label not in applied:
                    applied.append(label)

            inputs["text"] = self._rebuild_lora_text(items, active_only=True)

        return applied

    def _load_workflow(self):
        if not self.workflow_path.exists():
            raise FileNotFoundError(f"工作流文件不存在: {self.workflow_path}")
        with open(self.workflow_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _ref_node_id(value):
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
            return value[0]
        return None

    @staticmethod
    def _sampler_link_pairs(workflow):
        """Collect (positive_id, negative_id) link pairs from every sampler-like node."""
        pairs = []
        for _, node in workflow.items():
            if not isinstance(node, dict):
                continue
            if "KSampler" not in str(node.get("class_type", "")):
                continue
            inputs = node.get("inputs", {}) if isinstance(node.get("inputs"), dict) else {}
            pairs.append(
                (
                    ComfyUI._ref_node_id(inputs.get("positive")),
                    ComfyUI._ref_node_id(inputs.get("negative")),
                )
            )
        return pairs

    # --- editor-style workflow support ---
    _TEXT_INPUT_KEYS = ("text", "opt_text", "string", "text_positive", "positive", "prompt", "wildcard_text")

    def _has_writable_text(self, workflow, node_id):
        """放宽版文本输入判断：编辑节点（如 Easy_QwenEdit2509）的 prompt 输入也算。"""
        node = workflow.get(str(node_id)) if node_id else None
        if not isinstance(node, dict):
            return False
        ct = str(node.get("class_type", ""))
        if ct in ("KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"):
            return False
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            return False
        return any(key in inputs for key in self._TEXT_INPUT_KEYS)

    def _is_safe_prompt_node(self, workflow, node_id):
        """放宽版正向候选校验：可写文本且未被任何采样器当作 negative 使用。"""
        if not self._has_writable_text(workflow, node_id):
            return False
        pairs = self._sampler_link_pairs(workflow)
        negative_ids = {neg for _, neg in pairs if neg}
        return str(node_id) not in negative_ids

    def _detect_node_ids(self, workflow):
        """Pick prompt/negative/output nodes from the graph when configured ids are stale.

        Positive 只认采样器 positive 连线指向的文本节点，且绝不采用任何采样器的
        negative 目标，避免把用户提示词写进负面提示词节点；negative 仅作为参考值
        返回，是否采用由调用方决定。
        """
        found = {}
        pairs = self._sampler_link_pairs(workflow)
        negative_ids = {neg for _, neg in pairs if neg}
        positive_ids = [pos for pos, _ in pairs if pos]

        for pos in positive_ids:
            if pos not in negative_ids and self._has_text_input(workflow, pos) and "input" not in found:
                found["input"] = pos
        if "input" not in found:
            for nid, node in workflow.items():
                if (
                    isinstance(node, dict)
                    and str(nid) not in negative_ids
                    and self._has_text_input(workflow, nid)
                ):
                    found["input"] = str(nid)
                    break
        if "input" not in found:
            # 编辑类工作流：正向文本可能来自 Easy_QwenEdit / Kontext 等带 prompt 输入的
            # 编辑节点，采样器 positive 连线中间隔着 conditioning 包装节点，strict 识别
            # 找不到时放宽为“有可写文本输入且未被当 negative 使用”的节点。
            for nid, node in workflow.items():
                if (
                    isinstance(node, dict)
                    and str(nid) not in negative_ids
                    and self._has_writable_text(workflow, nid)
                ):
                    found["input"] = str(nid)
                    found["input_mode"] = "editor"
                    break

        for _, neg in pairs:
            if neg and self._has_text_input(workflow, neg):
                found["neg"] = neg
                break
        if "neg" not in found:
            for nid, node in workflow.items():
                if str(nid) != found.get("input") and self._has_text_input(workflow, nid):
                    found["neg"] = str(nid)
                    break

        for nid, node in workflow.items():
            if isinstance(node, dict) and node.get("class_type") in ("SaveImage", "PreviewImage"):
                found["output"] = str(nid)
                break
        return found

    @staticmethod
    def _has_text_input(workflow, node_id):
        node = workflow.get(str(node_id)) if node_id else None
        if not isinstance(node, dict):
            return False
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            return False
        return "TextEncode" in str(node.get("class_type", "")) and "clip" in inputs

    def _is_positive_text_node(self, workflow, node_id):
        """True 仅当该节点是文本节点且没有被任何采样器当作 negative 使用。

        采样器存在 positive 连线时，还要求节点确实出现在 positive 目标里，
        这样即便配置误填成负面提示词节点也不会被当成正向节点写入。
        """
        if not self._has_text_input(workflow, node_id):
            return False
        pairs = self._sampler_link_pairs(workflow)
        negative_ids = {neg for _, neg in pairs if neg}
        positive_ids = {pos for pos, _ in pairs if pos}
        if str(node_id) in negative_ids:
            return False
        if positive_ids and str(node_id) not in positive_ids:
            return False
        return True

    @staticmethod
    def _is_output_node(workflow, node_id):
        node = workflow.get(str(node_id)) if node_id else None
        return isinstance(node, dict) and node.get("class_type") in ("SaveImage", "PreviewImage")

    def _ensure_node_ids(self, workflow):
        """Fix a stale or wrong-kind positive/output id.

        负面节点 ID 尊重用户配置：非空时永不自动改写，只有配置为空时才补一个参考值。
        """
        detected = self._detect_node_ids(workflow)
        changes = []

        if not self._is_positive_text_node(workflow, self.input_id):
            candidate = detected.get("input")
            if candidate and (
                self._is_positive_text_node(workflow, candidate)
                or (detected.get("input_mode") == "editor" and self._is_safe_prompt_node(workflow, candidate))
            ):
                if self.input_id and self._has_text_input(workflow, self.input_id):
                    logger.warning(
                        f"[ComfyUI] 配置的正向节点 {self.input_id} 实际是负面提示词节点，"
                        f"已改用 {candidate}，避免提示词写进负面条件"
                    )
                self.input_id = candidate
                changes.append(f"input={self.input_id}")

        if not self.neg_node_id and self._has_text_input(workflow, detected.get("neg")):
            self.neg_node_id = detected["neg"]
            logger.info(
                f"[ComfyUI] 未配置负面节点，工作流 {self.wf_filename} 自动识别为 {self.neg_node_id}"
            )

        if not self._is_output_node(workflow, self.output_id) and self._is_output_node(workflow, detected.get("output")):
            self.output_id = detected["output"]
            changes.append(f"output={self.output_id}")

        if changes:
            logger.warning(
                f"[ComfyUI] 配置的节点 ID 不适用于工作流 {self.wf_filename}，已自动识别: "
                + ", ".join(changes)
            )
        return bool(changes)

    @staticmethod
    def _normalize_tag_text(tag):
        """把 tag 归一化到可比较的形式：去括号/权重、下划线转空格、小写、压缩空白。"""
        if not isinstance(tag, str):
            return ""
        text = tag.strip().strip("()[]{} ").strip()
        text = re.sub(r":\s*[\d.]+\s*$", "", text).strip()
        text = text.replace("_", " ").lower()
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _split_tag_text(cls, text):
        """按未被括号包裹的逗号切分 tag 列表（保留 (tag:1.2) 这类权重语法）。"""
        if not isinstance(text, str) or not text.strip():
            return []
        return [part.strip() for part in re.split(r",(?![^()]*\))", text) if part.strip()]

    def _negative_tag_set(self, workflow):
        """工作流负面节点与配置 negative_prompt 里出现的 tag 集合。

        配置里的 neg_node_id 可能指向一个已不存在于当前工作流的节点（换工作流后很常见），
        这种情况下按采样器的 negative 连线回退取值，避免负面词过滤失效。
        """
        tags = set()
        candidate_ids = []
        if self.neg_node_id:
            candidate_ids.append(str(self.neg_node_id))
        detected_neg = self._detect_node_ids(workflow).get("neg")
        if detected_neg and str(detected_neg) not in candidate_ids:
            candidate_ids.append(str(detected_neg))

        for node_id in candidate_ids:
            if not node_id or str(node_id) == str(self.input_id):
                continue
            node = workflow.get(node_id)
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs") or {}
            for key in ("text", "string", "negative", "text_negative", "prompt"):
                value = inputs.get(key)
                if isinstance(value, str) and value.strip():
                    tags.update(
                        self._normalize_tag_text(item)
                        for item in self._split_tag_text(value)
                    )
                    break
        tags.update(
            self._normalize_tag_text(item)
            for item in self._split_tag_text(self.neg_prompt or "")
        )
        tags.discard("")
        return tags

    def _strip_negative_tags(self, workflow, prompt):
        """把误写进正向提示词的负面 tag 去掉，保证负面词只出现在负面节点里。"""
        if not isinstance(prompt, str) or not prompt.strip():
            return prompt, []
        negatives = self._negative_tag_set(workflow)
        if not negatives:
            return prompt, []
        kept = []
        removed = []
        for tag in self._split_tag_text(prompt):
            if self._normalize_tag_text(tag) in negatives:
                removed.append(tag)
            else:
                kept.append(tag)
        if not kept:
            return prompt, []
        return ", ".join(kept), removed

    def _inject_params(self, workflow, prompt, lora_selections=None, write_prompt=True):
        """Inject prompt, optional LoRA selections, steps overrides, and seeds into workflow."""
        if not write_prompt:
            # 原生图生图工作流：只刷新种子与步数覆盖，绝不改写工作流自带的提示词。
            prompt = None
            lora_selections = None
        self._ensure_node_ids(workflow)
        prompt, removed_negative = self._strip_negative_tags(workflow, prompt)
        if removed_negative:
            logger.info(
                f"[ComfyUI] 已从正向提示词移除 {len(removed_negative)} 个与负面提示词重复的 tag: "
                + ", ".join(removed_negative)
            )
        if prompt is not None:
            node = workflow.get(self.input_id)
            if not node:
                message = (
                    f"工作流 {self.wf_filename} 中没有输入节点 {self.input_id}；"
                    "请检查插件配置里的 input_node_id（正向提示词节点），或把工作流改成 API 格式"
                )
                logger.error(f"critical: {message}")
                raise ValueError(message)

            inputs = node.get("inputs", {})
            target_keys = [
                "text",
                "opt_text",
                "string",
                "text_positive",
                "positive",
                "prompt",
                "wildcard_text",
            ]
            for key in target_keys:
                if key in inputs:
                    inputs[key] = prompt
                    break

        if prompt is not None and self.neg_node_id and self.neg_prompt:
            neg_node = workflow.get(self.neg_node_id)
            if neg_node:
                n_inputs = neg_node.get("inputs", {})
                n_keys = ["text", "string", "negative", "text_negative", "prompt"]
                for n_key in n_keys:
                    if n_key in n_inputs:
                        existing_neg = str(n_inputs.get(n_key, "")).strip()
                        config_neg = self.neg_prompt.strip()

                        if existing_neg and config_neg:
                            n_inputs[n_key] = f"{existing_neg}, {config_neg}"
                        elif config_neg:
                            n_inputs[n_key] = config_neg
                        break

        if self.lora_control_enabled and lora_selections:
            applied = self._apply_lora_selections(workflow, lora_selections)
            if applied:
                logger.info(f"[ComfyUI] applied LoRA selections: {', '.join(applied)}")
            else:
                logger.info("[ComfyUI] LoRA selections detected, but no compatible LoRA stack node was found")

        overrides = self._load_steps_override()
        if overrides:
            count = self._apply_steps_override(workflow, overrides)
            if count > 0:
                override_info = ", ".join([f"{k}:{v}步" for k, v in overrides.items()])
                logger.info(f"[ComfyUI] steps override applied: {override_info} (updated {count} targets)")
            else:
                logger.info("[ComfyUI] steps override configured but no matching reference was found")

        base_seed = random.randint(1, 999999999999999)
        ks_count = 0
        offset = 0

        for _, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            n_inputs = node_data.get("inputs", {})
            if not isinstance(n_inputs, dict):
                continue

            changed = False

            if "seed" in n_inputs:
                n_inputs["seed"] = base_seed + offset
                offset += 1
                changed = True

            if "noise_seed" in n_inputs:
                n_inputs["noise_seed"] = base_seed + offset
                offset += 1
                changed = True

            if changed:
                ks_count += 1

        logger.info(
            f"[ComfyUI] base seed: {base_seed}, updated {ks_count} seed/noise_seed inputs"
        )

    def _load_steps_override(self) -> dict:
        """Load steps override sidecar for the current workflow."""
        try:
            stem = self.workflow_path.stem
            sidecar = self.workflow_path.parent / f"{stem}.steps.json"

            if not sidecar.exists():
                return {}

            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return {}

            result = {}
            for key, value in data.items():
                if isinstance(value, dict) and "steps" in value:
                    steps = value.get("steps")
                    if isinstance(steps, (int, float)) and steps > 0:
                        result[str(key)] = int(steps)
                elif isinstance(value, (int, float)) and value > 0:
                    result[str(key)] = int(value)

            return result

        except Exception as e:
            logger.warning(f"[ComfyUI] failed to read steps override file: {e}")
            return {}

    def _apply_steps_override(self, workflow: dict, overrides: dict):
        """
        Override steps by node id.

        Only targets steps/steps_total fields that reference the requested ParameterBreak nodes.
        """
        if not overrides:
            return 0

        pb_nodes = {}
        for nid, node_data in workflow.items():
            if isinstance(node_data, dict) and node_data.get("class_type") == "ParameterBreak":
                pb_nodes[str(nid)] = node_data

        if not pb_nodes:
            logger.debug("[ComfyUI] no ParameterBreak nodes detected")
            return 0

        valid_overrides = {}
        for pb_id, steps in overrides.items():
            if pb_id in pb_nodes:
                valid_overrides[pb_id] = steps
            else:
                logger.warning(f"[ComfyUI] override target node {pb_id} not found in current workflow")

        if not valid_overrides:
            return 0

        override_count = 0
        steps_keys = ("steps", "steps_total")

        for nid, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue

            n_inputs = node_data.get("inputs", {})
            if not isinstance(n_inputs, dict):
                continue

            for key in steps_keys:
                if key not in n_inputs:
                    continue

                value = n_inputs[key]
                if isinstance(value, list) and len(value) == 2:
                    ref_node_id = str(value[0])
                    if ref_node_id in valid_overrides:
                        new_steps = valid_overrides[ref_node_id]
                        n_inputs[key] = new_steps
                        override_count += 1
                        logger.debug(f"[ComfyUI] node {nid}.{key}: [{ref_node_id}] -> {new_steps}")

        return override_count

    def _find_native_image_node(self, workflow):
        """工作流自带且唯一的 LoadImage 节点 id；没有或多个则返回 None。"""
        load_ids = [
            str(nid)
            for nid, node in workflow.items()
            if isinstance(node, dict) and str(node.get("class_type", "")) == "LoadImage"
        ]
        if len(load_ids) == 1:
            return load_ids[0]
        if len(load_ids) > 1:
            logger.info(f"[ComfyUI] 工作流含 {len(load_ids)} 个 LoadImage 节点，退回通用 img2img 改造")
        return None

    def _find_vae_node_id(self, workflow):
        """Locate the node that provides the VAE for image encoding.

        Preference order: the node referenced by a VAEDecode-family "vae" input,
        otherwise the first VAELoader. Returns a node id string or None."""
        decode_vae = None
        vae_loaders = []
        for nid, node in workflow.items():
            if not isinstance(node, dict):
                continue
            ct = str(node.get("class_type", ""))
            if ct == "VAELoader":
                vae_loaders.append(str(nid))
            if "Decode" not in ct:
                continue
            v = (node.get("inputs") or {}).get("vae")
            if isinstance(v, list) and len(v) == 2:
                decode_vae = str(v[0])
        if decode_vae:
            return decode_vae
        if vae_loaders:
            return vae_loaders[0]
        return None

    def _apply_img2img(self, workflow, image_name, denoise):
        """Convert a loaded text-to-image workflow into image-to-image in place.

        Adds a LoadImage + VAEEncode pair and repoints every KSampler/
        KSamplerAdvanced latent_image input at the encoded latent, lowering
        denoise so the original image survives. Returns
        (sampler_count, error_message); sampler_count 0 means not eligible."""
        sampler_ids = []
        for nid, node in workflow.items():
            if not isinstance(node, dict):
                continue
            ct = str(node.get("class_type", ""))
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            if ct in ("KSampler", "KSamplerAdvanced") and "latent_image" in inputs:
                sampler_ids.append(str(nid))
        if not sampler_ids:
            return 0, "当前工作流没有 KSampler/KSamplerAdvanced 采样节点，无法改造成图生图"
        vae_id = self._find_vae_node_id(workflow)
        if not vae_id:
            return 0, "当前工作流找不到 VAE 来源节点（VAELoader 或解码节点的 vae 输入），无法改造成图生图"
        load_id = "comfybot_img_load"
        encode_id = "comfybot_vae_encode"
        while load_id in workflow:
            load_id += "x"
        while encode_id in workflow:
            encode_id += "x"
        workflow[load_id] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        workflow[encode_id] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": [load_id, 0], "vae": [vae_id, 0]},
        }
        try:
            denoise_val = max(0.05, min(0.95, float(denoise)))
        except (TypeError, ValueError):
            denoise_val = 0.55
        for sid in sampler_ids:
            node = workflow[sid]
            node["inputs"]["latent_image"] = [encode_id, 0]
            if "denoise" in node["inputs"]:
                node["inputs"]["denoise"] = denoise_val
        return len(sampler_ids), None

    async def upload_image(self, image_path):
        """Upload a local image into ComfyUI input directory.

        Returns (server_filename, error_message)."""
        try:
            with open(image_path, "rb") as fp:
                data = fp.read()
        except Exception as e:
            return None, f"读取图片失败: {e}"
        suffix = Path(image_path).suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
            suffix = ".png"
        name = "comfybot_" + "".join(random.choice("0123456789abcdef") for _ in range(8)) + suffix
        try:
            async with aiohttp.ClientSession() as session:
                form = aiohttp.FormData()
                form.add_field("image", data, filename=name, content_type="application/octet-stream")
                form.add_field("overwrite", "true")
                async with session.post(f"{self.url}/upload/image", data=form) as resp:
                    if resp.status != 200:
                        return None, f"上传图片失败: HTTP {resp.status}"
                    res = await resp.json()
                    return res.get("name") or name, None
        except Exception as e:
            return None, f"上传图片报错: {e}"

    async def generate_img2img(self, prompt, image_path, denoise=0.55, lora_selections=None):
        """Generate with an input image: upload it, convert the current

        workflow into image-to-image, then submit and wait like generate()."""
        up_name, err = await self.upload_image(image_path)
        if not up_name:
            return None, err

        try:
            workflow = self._load_workflow()
        except Exception as e:
            return None, str(e)

        native_image_id = self._find_native_image_node(workflow)
        if native_image_id:
            # 原生图生图工作流（自带 LoadImage，如 K2 图生三次元）：只把图片填进
            # 工作流自带的输入节点，保留其自带提示词与潜空间链路。
            node_inputs = workflow[native_image_id].setdefault("inputs", {})
            node_inputs["image"] = up_name
            if str(prompt or "").strip():
                logger.info(f"[ComfyUI] 原生图生图工作流，忽略用户描述: {(prompt or '')[:60]}")
            logger.info(f"[ComfyUI] img2img: 原生模式，图片填入 LoadImage 节点 {native_image_id}")
            self._inject_params(workflow, None, None, write_prompt=False)
        else:
            resolved_loras = self.resolve_lora_selections(lora_selections) if self.lora_control_enabled else []
            prompt, injected_hints = self._inject_selected_lora_prompt_hints(str(prompt or ""), resolved_loras)
            if injected_hints:
                logger.info(f"[ComfyUI] injected LoRA prompt hints: {', '.join(injected_hints)}")
            try:
                self._inject_params(workflow, prompt, resolved_loras)
            except ValueError as exc:
                return None, str(exc)

            sampler_count, conv_err = self._apply_img2img(workflow, up_name, denoise)
            if conv_err:
                return None, conv_err
            logger.info(f"[ComfyUI] img2img: uploaded={up_name} denoise={denoise} samplers={sampler_count}")

        client_id = str(random.randint(100000, 999999))
        async with aiohttp.ClientSession() as session:
            payload = {"prompt": workflow, "client_id": client_id}
            try:
                async with session.post(f"{self.url}/prompt", json=payload) as resp:
                    if resp.status != 200:
                        return None, f"连接 ComfyUI 失败: {resp.status}"
                    res_json = await resp.json()
                    prompt_id = res_json.get("prompt_id")
            except Exception as e:
                return None, f"请求报错: {str(e)}"

            deadline = time.monotonic() + self.gen_timeout
            while time.monotonic() < deadline:
                await asyncio.sleep(1)
                try:
                    async with session.get(f"{self.url}/history/{prompt_id}") as h_resp:
                        if h_resp.status != 200:
                            continue
                        history = await h_resp.json()
                except Exception:
                    continue

                if prompt_id in history:
                    entry = history.get(prompt_id) or {}
                    status = entry.get("status") or {}
                    if status.get("status_str") == "error":
                        return None, f"工作流执行失败: {status.get('messages') or '未知错误'}"
                    outputs = entry.get("outputs", {})
                    img_info = None

                    if self.output_id and self.output_id in outputs:
                        imgs = outputs[self.output_id].get("images", [])
                        if imgs:
                            img_info = imgs[0]

                    if not img_info:
                        for node_out in outputs.values():
                            if "images" in node_out and node_out["images"]:
                                img_info = node_out["images"][0]
                                break

                    if img_info:
                        fname = img_info["filename"]
                        sfolder = img_info["subfolder"]
                        itype = img_info["type"]
                        img_url = f"{self.url}/view?filename={fname}&subfolder={sfolder}&type={itype}"

                        async with session.get(img_url) as img_res:
                            if img_res.status == 200:
                                return await img_res.read(), None
                            return None, "下载图片失败"

                    return None, "工作流执行完成，但未找到输出图片"

            return None, f"生成超时（已等待 {self.gen_timeout} 秒仍未完成，可在插件配置 workflow_settings.timeout_seconds 调大）"

    async def generate(self, prompt, lora_selections=None):
        """Generate an image with the current workflow."""
        client_id = str(random.randint(100000, 999999))
        try:
            workflow = self._load_workflow()
        except Exception as e:
            return None, str(e)

        resolved_loras = self.resolve_lora_selections(lora_selections) if self.lora_control_enabled else []
        prompt, injected_hints = self._inject_selected_lora_prompt_hints(str(prompt or ""), resolved_loras)

        if injected_hints:
            logger.info(f"[ComfyUI] injected LoRA prompt hints: {', '.join(injected_hints)}")

        try:
            self._inject_params(workflow, prompt, resolved_loras)
        except ValueError as exc:
            return None, str(exc)

        async with aiohttp.ClientSession() as session:
            payload = {"prompt": workflow, "client_id": client_id}
            try:
                async with session.post(f"{self.url}/prompt", json=payload) as resp:
                    if resp.status != 200:
                        return None, f"连接 ComfyUI 失败: {resp.status}"
                    res_json = await resp.json()
                    prompt_id = res_json.get("prompt_id")
            except Exception as e:
                return None, f"请求报错: {str(e)}"

            deadline = time.monotonic() + self.gen_timeout
            while time.monotonic() < deadline:
                await asyncio.sleep(1)
                try:
                    async with session.get(f"{self.url}/history/{prompt_id}") as h_resp:
                        if h_resp.status != 200:
                            continue
                        history = await h_resp.json()
                except Exception:
                    continue

                if prompt_id in history:
                    entry = history[prompt_id] or {}
                    status = entry.get("status") or {}
                    if status.get("status_str") == "error":
                        return None, f"工作流执行失败: {status.get('messages') or '未知错误'}"
                    outputs = entry.get("outputs", {})
                    img_info = None

                    if self.output_id and self.output_id in outputs:
                        imgs = outputs[self.output_id].get("images", [])
                        if imgs:
                            img_info = imgs[0]

                    if not img_info:
                        for node_out in outputs.values():
                            if "images" in node_out and node_out["images"]:
                                img_info = node_out["images"][0]
                                break

                    if img_info:
                        fname = img_info["filename"]
                        sfolder = img_info["subfolder"]
                        itype = img_info["type"]
                        img_url = f"{self.url}/view?filename={fname}&subfolder={sfolder}&type={itype}"

                        async with session.get(img_url) as img_res:
                            if img_res.status == 200:
                                return await img_res.read(), None
                            return None, "下载图片失败"

                    return None, "工作流执行完成，但未找到输出图片"

            return None, f"生成超时（已等待 {self.gen_timeout} 秒仍未完成，可在插件配置 workflow_settings.timeout_seconds 调大）"
