from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser  # i np. JsonOutputParser, jeśli używasz
from langchain_core.runnables import (
    RunnableLambda,
    RunnableMap,
    RunnableParallel,       # jeśli gdzieś używasz
    RunnablePassthrough,    # jeśli gdzieś używasz
)
from langchain_openai import ChatOpenAI, OpenAI



from bs4 import BeautifulSoup  
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, field
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, SystemMessage
from typing import List, Dict, Optional, Any, Set, Literal, get_origin, get_args, Union
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
#from langchain.prompts import PromptTemplate
#from langchain.schema import HumanMessage, SystemMessage
#from langchain_core.messages import 
#from langchain.agents import tool
from langchain.tools import tool
from langgraph.types import Interrupt
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from datetime import datetime
from rich import print as rprint
from datetime import datetime, timezone
from langgraph.graph import StateGraph
from IPython.display import Image, display      
from langgraph.types import interrupt, Command
from typing import Callable, Optional
from DAPartner.Configs import config as conf
from DAPartner.State import state as st
from typing import Iterable


import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # folder DAPartner_repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
import asyncio
import re
import os
os.environ.setdefault("USER_AGENT", "DAPartner/1.0")
import urllib.request
import csv
import io

# ---------------------------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------------------------
load_dotenv()  # wymaga OPENAI_API_KEY w .env

session_id = "terminal-user"
thread_id = "auto-fill-pydantic"
BASE_DIR = Path(__file__).parent
BASE_PROMPTS = BASE_DIR / ".." / "Prompts" / "Reusable"
GLOBAL_FILES = BASE_DIR / ".." / "Configs" 
os.environ.setdefault("USER_AGENT", "DAPartner/1.0 (+https://leanx.eu)")

# ---------------------------------------------------------------------------
# 2. Variables 
# ---------------------------------------------------------------------------
memory_store = {}


# ---------------------------------------------------------------------------
# 3. Helpers functions
# ---------------------------------------------------------------------------

NOTIFY_HOOK: Optional[Callable[[str], None]] = None
def set_notify_hook(fn: Callable[[str], None]) -> None:
    """Frontend może podpiąć callback do komunikatów postępu."""
    global NOTIFY_HOOK
    NOTIFY_HOOK = fn

def update_state_with_whitelist(state, response, allowed_state_changes: set[str]):
    """
    Aktualizuje `state` TYLKO o klucze z `allowed_state_changes`.

    Parametry:
      - state: DesignState (DAPartner)
      - response: dict LUB string z JSON-em (podzbiór głównego stanu)
      - allowed_state_changes: set[str] z dozwolonymi polami

    Zachowanie:
      - string z JSON → parsowany (np. "[...]", "{...}", "true"/"false", liczby)
      - "a,b,c" → ["a","b","c"] gdy pole oczekuje List[str]
      - nie nadpisuje pustymi (None, "", [], {}) – wyjątek: typ bool
      - dopasowuje do adnotacji typów z klasy stanu (Pydantic v1/v2)
      - listy obiektów Pydantic składa z list słowników (np. dimensions_to_create)
    """

    # --- wejście może być stringiem z JSON-em ---
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except Exception:
            # jeśli to nie jest poprawny JSON, nic nie robimy
            return state
    updates = dict(response or {})

    StateCls = state.__class__
    # Pobierz adnotacje typów pól klasy (działa dla Pydantic v1 i v2)
    try:
        type_hints = StateCls.model_fields  # v2
        def _get_ann(name): 
            fld = type_hints.get(name)
            return getattr(fld, "annotation", Any) if fld else Any
    except Exception:
        # v1
        try:
            fields_v1 = StateCls.__fields__
            def _get_ann(name):
                fld = fields_v1.get(name)
                return getattr(fld, "type_", Any) if fld else Any
        except Exception:
            def _get_ann(_): return Any

    def _is_nonempty(val: Any) -> bool:
        if isinstance(val, bool):
            return True
        if val is None:
            return False
        if isinstance(val, str) and val.strip() == "":
            return False
        if isinstance(val, (list, tuple, set, dict)) and len(val) == 0:
            return False
        return True

    def _looks_like_json_array_or_obj(s: str) -> bool:
        s2 = s.strip()
        return (s2.startswith("[") and s2.endswith("]")) or (s2.startswith("{") and s2.endswith("}"))

    def _coerce_primitive(t, v):
        origin = get_origin(t)
        if origin is Union:
            args = [a for a in get_args(t) if a is not type(None)]  # noqa: E721
            if args:
                return _coerce_primitive(args[0], v)

        if t is bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("true", "t", "yes", "y", "1"):  return True
                if s in ("false", "f", "no", "n", "0"):  return False
            return bool(v)

        if t is int:
            if isinstance(v, int): return v
            if isinstance(v, float): return int(v)
            if isinstance(v, str):
                s = v.strip()
                if re.fullmatch(r"[-+]?\d+", s):
                    return int(s)
            return v

        if t is float:
            if isinstance(v, (int, float)): return float(v)
            if isinstance(v, str):
                s = v.strip().replace(",", ".")
                try:
                    return float(s)
                except Exception:
                    return v
            return v

        if t is str:
            if isinstance(v, str):
                if _looks_like_json_array_or_obj(v):
                    try:
                        return json.loads(v)
                    except Exception:
                        return v
                return v
            return str(v)

        if t is dict or get_origin(t) is dict:
            if isinstance(v, dict):
                return v
            if isinstance(v, str) and _looks_like_json_array_or_obj(v):
                try:
                    obj = json.loads(v)
                    return obj if isinstance(obj, dict) else v
                except Exception:
                    return v
            return v

        return v

    def _coerce_to_type(t, v):
        origin = get_origin(t)

        if origin is Union:
            args = [a for a in get_args(t) if a is not type(None)]  # noqa: E721
            if not args:
                return v
            return _coerce_to_type(args[0], v)

        # List[T] / Tuple[T] / Set[T]
        if origin in (list, tuple, set):
            elem_type = get_args(t)[0] if get_args(t) else Any

            if isinstance(v, str):
                s = v.strip()
                if _looks_like_json_array_or_obj(s):
                    try:
                        arr = json.loads(s)
                        if isinstance(arr, list):
                            return [ _coerce_to_type(elem_type, x) for x in arr ]
                    except Exception:
                        pass
                if elem_type in (str, Any) and "," in s:
                    items = [it.strip() for it in s.split(",") if it.strip() != ""]
                    return items
                return [ _coerce_to_type(elem_type, s) ]

            if isinstance(v, (list, tuple, set)):
                return [ _coerce_to_type(elem_type, x) for x in v ]

            return [ _coerce_to_type(elem_type, v) ]

        # Pydantic BaseModel
        try:
            if isinstance(t, type) and issubclass(t, BaseModel):
                if isinstance(v, t):
                    return v
                if isinstance(v, str) and _looks_like_json_array_or_obj(v):
                    try:
                        obj = json.loads(v)
                        if isinstance(obj, dict):
                            return t(**obj)
                    except Exception:
                        pass
                if isinstance(v, dict):
                    return t(**v)
                return v
        except Exception:
            pass

        return _coerce_primitive(t, v)

    # --- właściwa aktualizacja ---
    for key, incoming in (updates or {}).items():
        if key not in allowed_state_changes:
            continue
        if not hasattr(state, key):
            # top-level tylko; nie robimy "deep set" po kropkach
            continue

        target_type = _get_ann(key)

        decoded = incoming
        if isinstance(incoming, str) and _looks_like_json_array_or_obj(incoming):
            try:
                decoded = json.loads(incoming)
            except Exception:
                decoded = incoming

        coerced = _coerce_to_type(target_type, decoded)

        if isinstance(coerced, bool) or _is_nonempty(coerced):
            setattr(state, key, coerced)

    return state

def sanitize_message_for_ui(s: str) -> str:
    """Usuń niechciane JSON-owe escape'y przed wysłaniem do UI."""
    if not isinstance(s, str):
        return str(s)

    # jeśli mamy string opakowany cudzysłowami → spróbuj odczytać jako JSON
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        try:
            return json.loads(s)
        except Exception:
            pass

    # zamień uciekające sekwencje na realne znaki
    return s.replace("\\n", "\n").replace("\\t", "\t")

def _is_nonempty_value(value: Any) -> bool:
    """
    True, jeżeli wartość jest 'znana' i niepusta.
    - None -> False (nie aktualizujemy)
    - ""   -> False
    - []   -> False
    - dla bool -> zawsze True (jeśli klucz wystąpił, to świadomy update, nawet False)
    - dla liczb/strukt. -> True, jeśli nie None i niepuste (dla str/list)
    """
    if isinstance(value, bool):
        return True  # bool ma być aktualizowany, jeśli klucz istnieje
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return False
    return True

def _known_fields_of_dimension_to_create() -> Set[str]:
    try:
        # Pydantic v2
        return set(st.DimensionToCreate.model_fields.keys())
    except Exception:
        # Pydantic v1
        return set(st.DimensionToCreate.__fields__.keys())

def _known_fields_of_fact_to_create() -> Set[str]:
    try:
        # Pydantic v2
        return set(st.FactToCreate.model_fields.keys())
    except Exception:
        # Pydantic v1
        return set(st.FactToCreate.__fields__.keys())

def _filter_known_fields(
        d: Dict[str, Any],
        include_empty: Optional[Set[str]] = None,
        allowed_fields: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
    """
    Zwraca tylko znane pola z DimensionToCreate, pomijając puste/nieznane.
    Możesz dodatkowo ograniczyć do 'allowed_fields'.
    """
    include_empty = include_empty or set()
    known = _known_fields_of_dimension_to_create()
    out = {}
    for k, v in d.items():
        if k not in known:
            continue
        if allowed_fields is not None and k not in allowed_fields:
            continue
        if k in include_empty:
            out[k] = v
        else:
            if _is_nonempty_value(v):
                out[k] = v
    return out

def _update_dimension_in_place(target: st.DimensionToCreate, patch: Dict[str, Any]) -> None:
    """
    Aktualizuje istniejący obiekt DimensionToCreate tylko wartościami niepustymi.
    Boole są aktualizowane, jeśli klucz jest obecny (nawet jeśli False).
    """
    known = _known_fields_of_dimension_to_create()
    for k, v in patch.items():
        if k not in known:
            continue
        # Dla boola: aktualizuj zawsze, skoro klucz jest w patchu
        if isinstance(getattr(target, k, None), bool) or isinstance(v, bool):
            setattr(target, k, v)
            continue
        # Dla innych typów: tylko wartości 'niepuste'
        if _is_nonempty_value(v):
            setattr(target, k, v)

def _update_fact_in_place(target: st.FactToCreate, patch: Dict[str, Any]) -> None:
    """
    Aktualizuje istniejący obiekt FactToCreate tylko wartościami niepustymi.
    Boole są aktualizowane, jeśli klucz jest obecny (nawet jeśli False).
    """
    known = _known_fields_of_fact_to_create()
    for k, v in patch.items():
        if k not in known:
            continue
        # Dla boola: aktualizuj zawsze, skoro klucz jest w patchu
        if isinstance(getattr(target, k, None), bool) or isinstance(v, bool):
            setattr(target, k, v)
            continue
        # Dla innych typów: tylko wartości 'niepuste'
        if _is_nonempty_value(v):
            setattr(target, k, v)
   
def load_SAP_2_Snowflake_data_types_mapping_to_state (state: st.DesignState) -> st.DesignState:
    """ Wczytuje mapowanie z pliku SAP_2_Snowflake_data_types_mapping do obiektu DesignState. """

    # Ustawienie danych w stanie
    state.SAP_2_Snowflake_data_types = conf.SAP_2_Snowflake_data_types_mapping_file_content

    return state

def load_list_available_dimensions_to_state (state: st.DesignState) -> st.DesignState:
    """ Wczytuje dostępne wymiary z pliku LIST_AVAILABLE_DIMENSIONS do obiektu DesignState. """
    # Parsowanie JSON-a ze stringa
    raw_data = json.loads(conf.LIST_AVAILABLE_DIMENSIONS)

    # Zamiana listy słowników na listę obiektów AvailableDimensions
    parsed_dimensions: List[st.AvailableDimension] = [
        st.AvailableDimension(**item) for item in raw_data
    ]
    
    # Ustawienie danych w stanie
    state.available_dimensions = parsed_dimensions

    # Przepisywanie wartości z available_dimensions do parsed_essence_dimensions:
    parsed_essence_dimensions: List[st.AvailableDimensionEssence] = [
        st.AvailableDimensionEssence(
            dimension_name=dim.dimension_name,
            main_source_table=dim.main_source_table
        )
        for dim in parsed_dimensions
    ]

    # Ustawienie wersji "essence" w stanie
    state.available_dimensions_essence = parsed_essence_dimensions

    return state

def dimension_as_string(state: st.DesignState, name: str) -> str:
    dim = next((d for d in state.dimensions_to_create if d.dimension_name == name), None)
    if dim is None:
        raise ValueError(f"Nie znaleziono wymiaru: {name}")

    # Pydantic v2
    if hasattr(dim, "model_dump_json"):
        return dim.model_dump_json(indent=2, exclude_none=True)

    # Pydantic v1 (fallback)
    return dim.json(indent=2, exclude_none=True)

def fact_as_string(state: st.DesignState, name: str) -> str:
    fact = next((f for f in state.facts_to_create if f.fact_name == name), None)
    if fact is None:
        raise ValueError(f"Nie znaleziono faktu: {name}")

    # Pydantic v2
    if hasattr(fact, "model_dump_json"):
        return fact.model_dump_json(indent=2, exclude_none=True)

    # Pydantic v1 (fallback)
    return fact.json(indent=2, exclude_none=True)

def point_critical_columns_in_source(dimension_name: str, source_tables: str, source_tables_analyze_txt: str) -> str:   

    messages = [
        SystemMessage(content=conf.Point_Critical_Columns_In_Source_Text, ),
        HumanMessage(content=(
                f"Modelujesz wymiar (dimension_name): {dimension_name}\n"
                f"Tabele źródłowe to (source_tables): {source_tables}\n"
                f"Przeanalizuj kolumny z tabel źródłowych. Jeżeli user uploadował dodatkową analizę, znajdziesz ją tu (source_tables_analyze_txt). Uwaga - nie pomijaj w analizie żadnej z kolumn: {source_tables_analyze_txt}\n"
            ))
    ]
    response = conf.llm_point_critical_columns_in_source.invoke(messages).content.strip()
    return response

def _all_dim_elements_unapproved(dim) -> bool:
    # "świeży" wymiar: wszystkie flagi False (domyślne)
    return all([
        dim.design_on_entity_level_approved is False,
        dim.design_on_columns_level_approved is False,
        dim.feed_2_modeling_tool_approved is False,
        dim.scripts_approved is False,
    ]) 

def _all_fact_elements_unapproved(fact) -> bool:
    # "świeży" wymiar: wszystkie flagi False (domyślne)
    return all([
        fact.fact_design_on_entity_level_approved is False,
        fact.fact_design_on_columns_level_approved is False,
        fact.fact_feed_2_modeling_tool_approved is False,
        fact.fact_scripts_approved is False,
    ]) 

def _has_any_dim_elements_unapproved(dim) -> bool:
    return any([
        dim.design_on_entity_level_approved is False,
        dim.design_on_columns_level_approved is False,
        dim.feed_2_modeling_tool_approved is False,
        dim.scripts_approved is False,
    ])

def _has_any_fact_elements_unapproved(fact) -> bool:
    return any([
        fact.fact_design_on_entity_level_approved is False,
        fact.fact_design_on_columns_level_approved is False,
        fact.fact_feed_2_modeling_tool_approved is False,
        fact.fact_scripts_approved is False,
    ])

def _normalize_tables(required: Iterable[str] | None, additional: Iterable[str] | None) -> list[str]:
    items: list[str] = []
    if required:
        items.extend([t for t in required if isinstance(t, str)])
    if additional:
        items.extend([t for t in additional if isinstance(t, str)])
    seen = set()
    norm: list[str] = []
    for t in items:
        up = t.strip().upper()
        if up and up not in seen:
            seen.add(up)
            norm.append(up)
    return norm

async def _fetch_leanx(urls: list[str]) -> dict[str, str]:
    """Pobiera i czyści treści z leanx.eu używając wyłącznie LangChain Community."""
    loader = AsyncHtmlLoader(urls) 
    docs = await loader.aload()

    transformer = BeautifulSoupTransformer()
    docs = transformer.transform_documents(
        docs,
        tags_to_extract=["main", "article", "section", "div"],
        remove_comments=True,
    )

    out: dict[str, str] = {}
    for d in docs:
        src = d.metadata.get("source", "")
        text = (d.page_content or "").strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        out[src] = text[:40000]  # limit długości; trzymamy prompt w ryzach
    return out

def _fetch_html_sync(url: str) -> str:
    """Pobiera surowy HTML (synchron.). Bez dodatkowych zależności."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": os.environ.get("USER_AGENT", "DAPartner/1.0 (+https://leanx.eu)")}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def _extract_table_fields_as_csv(html: str, table_name: str) -> str | None:
    """
    Zwraca CSV (string) tylko dla sekcji '<TABLE> table fields'.
    Gdy nie znajdzie – zwraca None.
    """
    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")  # lub "html.parser" jeśli nie masz lxml
    wanted = f"{table_name.strip().upper()} table fields"

    # 1) znajdź nagłówek sekcji (H1/H2/H3/H4), np. 'BKPF table fields'
    pat = re.compile(rf"^{re.escape(table_name.strip())}\s+table\s+fields$", re.I)
    heading = None
    for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
        txt = tag.get_text(strip=True)
        if pat.match(txt):
            heading = tag
            break

    # mała tolerancja (gdyby wielkość liter/odstępy były inne)
    if heading is None:
        for tag in soup.find_all(["h2", "h3", "h4"]):
            txt = tag.get_text(strip=True).lower()
            if "table fields" in txt and table_name.strip().lower() in txt:
                heading = tag
                break

    if heading is None:
        return None

    # 2) pierwsza tabela po nagłówku = właściwa sekcja pól
    table = heading.find_next("table")
    if table is None:
        return None

    # 3) konwersja HTML table -> CSV
    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        row = []
        for c in cells:
            txt = c.get_text(separator=" ", strip=True)
            txt = " ".join(txt.split())
            row.append(txt)
        rows.append(row)

    if not rows:
        return None

    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    for r in rows:
        writer.writerow(r)
    return buf.getvalue().strip()

def _extract_html_tables_as_csv(html: str) -> list[str]:
    """Zwraca listę CSV (string) — po jednym na każdą tabelę <table> w HTML."""
    if not html or "<table" not in html.lower():
        return []
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return []

    soup = BeautifulSoup(html, "lxml")  # masz już lxml; jak go zabraknie, BeautifulSoup użyje wbudowanego parsera
    tables = soup.find_all("table")
    csv_list: list[str] = []

    for tbl in tables:
        rows = []
        # wiersze
        for tr in tbl.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if not cells:
                continue
            row = []
            for c in cells:
                # tekst komórki, bez nowych linii/zbędnych spacji
                txt = c.get_text(separator=" ", strip=True)
                txt = " ".join(txt.split())
                row.append(txt)
            rows.append(row)

        if not rows:
            continue

        # jeśli pierwsza linia to nagłówki (TH), zostawiamy jak jest; inaczej przyjmujemy pierwszy wiersz jako dane
        # serializacja do CSV
        buf = io.StringIO()
        writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        for r in rows:
            writer.writerow(r)
        csv_list.append(buf.getvalue().strip())

    return csv_list

def _ensure_source_table_analyze_from_documentation(state: st.DesignState, notify=None) -> None:
    # uzupełnij notify z globalnego hooka, jeśli nie przekazano
    if notify is None:
        notify = NOTIFY_HOOK

    tables = _normalize_tables(state.required_source_tables, state.additional_source_tables)
    if not tables:
        return

    if state.source_table_analyze is None:
        state.source_table_analyze = {}

    missing = []
    for t in tables:
        existing = (
            state.source_table_analyze.get(t)
            or state.source_table_analyze.get(t.upper())
            or state.source_table_analyze.get(t.lower())
        )
        if not (isinstance(existing, str) and len(existing.strip()) >= 30):
            missing.append(t)

    if not missing:
        return

    # <<< KLUCZOWE: komunikat na froncie, gdy zaczynamy sprawdzanie >>>
    if notify:
        notify("Checking documentation for source tables")

    for t in missing:
        url = f"https://leanx.eu/en/sap/table/{t.lower()}.html"
        html = _fetch_html_sync(url)
        csv_fields = _extract_table_fields_as_csv(html, t.upper())
        if csv_fields:
            state.source_table_analyze[t.upper()] = csv_fields
            if notify:
                # krótkie potwierdzenie każdej sprawdzonej tabeli
                notify(f"Checked: {t}")


# ---------------------------------------------------------------------------
# 3. Helpers - update state
# ---------------------------------------------------------------------------
def requirements_collection_update_state (state: st.DesignState) -> st.DesignState:
    """ LLM generuje jsona z required_source_tables i required_source_tables_approved na podstawie dialogu z użytkownikiem a reszta funkcji wpycha to do stanu """

    ALLOWED_STATE_CHANGES = {
        "required_source_tables",
        "required_source_tables_approved"
    }
    messages = [
        SystemMessage(content=conf.REQUIREMENTS_COLLECTION_PROMPT_Detect_State_TEXT),
        HumanMessage(content=f"Odpowiedź użytkownika: {state.last_5_messages}")
    ]
    response = conf.llm_Requirements_Collection_Detect_State.invoke(messages).content.strip()
    try:
        state = update_state_with_whitelist(state, response, ALLOWED_STATE_CHANGES)
    except Exception:
        print("Debug - issue in requirements_collection_update_state")
    return state

class requirements_analysis_update_patch(BaseModel):
    """
    Patch dla etapu 'requirements_analysis'.
    Zawiera wyłącznie pola dozwolone do zmiany w DesignState.
    Dla obiektów stosujemy upsert po nazwie (case-insensitive).
    """
    # --- TOP-LEVEL ---
    additional_source_tables: Optional[List[str]] = Field(
        default=None, description="Propozycje dodatkowych tabel źródłowych (zastępują bieżącą listę)."
    )
    additional_source_tables_approved: Optional[bool] = Field(
        default=None, description="Czy użytkownik zaakceptował dodatkowe tabele?"
    )
    objects_to_create_approved: Optional[bool] = Field(
        default=None, description="Czy użytkownik zaakceptował listę obiektów (fakty + wymiary)?"
    )
    # --- Łatki obiektów (upsert po nazwie) ---
    class FactPatch(BaseModel):
        fact_name:           str
        fact_main_source_table:   Optional[List[str]] = None
        fact_other_source_tables: Optional[List[str]] = None
        fact_connected_dimensions: Optional[List[str]] = None
    class DimensionPatch(BaseModel):
        dimension_name:               str
        main_source_table:            Optional[List[str]] = None
        business_key_on_source_side:  Optional[List[str]] = None
        other_source_tables:          Optional[List[str]] = None
        surrogate_key:                Optional[str]       = None
    class ContextFactPatch(BaseModel):
        context_fact_name:              		str 
        context_fact_source:      			    List[str] = Field(description="Name of main fact which is feeding curent fact" ) 
        context_fact_filtration:                bool = Field(default=False, description="Where condition which filtering proper subset for fact from main table" )
    # Nazwy jak w stanie – żeby LLM „trafił” bez extra mapowania
    facts_to_create: Optional[List[FactPatch]] = Field(
        default=None,
        description="Upsert faktów; nowe tworzymy TYLKO gdy podano main_source_table."
    )
    # Nazwy jak w stanie – żeby LLM „trafił” bez extra mapowania
    context_facts_to_create: Optional[List[ContextFactPatch]] = Field(
        default_factory=list, 
        description="Lista faktów kontekstowych do stworzenia wraz z ich szczegółami" 
    )
    dimensions_to_create: Optional[List[DimensionPatch]] = Field(
        default=None,
        description="Upsert wymiarów; nowe tworzymy TYLKO gdy podano main_source_table."
    )
def requirements_analysis_update_state(state: st.DesignState) -> st.DesignState:
    """
    LLM -> requirements_analysis_update_patch -> zastosowanie do DesignState.
    Polityka:
    - Top-level listy zastępujemy (z deduplikacją i odfiltrowaniem pustych),
      bo to etap 'requirements analysis'.
    - Fakty/Wymiary: upsert po nazwie (case-insensitive); tworzymy tylko gdy
      patch zawiera wymagane minimalne dane (main_source_table).
    - Bool-e ustawiamy zawsze, jeśli wystąpią w patchu.
    """
    structured_llm = conf.llm_Requirements_Analysis_Detect_State.with_structured_output(
        requirements_analysis_update_patch
    )

    messages = [
        SystemMessage(content=conf.REQUIREMENTS_ANALYSIS_PROMPT_Detect_State_TEXT),
        HumanMessage(content=(
            f"Odpowiedź użytkownika (last_5_messages): {state.last_5_messages}\n"
            f"Aktualne fakty: {[f.fact_name for f in state.facts_to_create]}\n"
            f"Aktualne wymiary: {[d.dimension_name for d in state.dimensions_to_create]}\n"
            f"Dodatkowe tabele (stan): {state.additional_source_tables}"
        ))
    ]

    try:
        patch: requirements_analysis_update_patch = structured_llm.invoke(messages)
    except Exception as e:
        print(f"Debug - błąd w requirements_analysis_update_state (LLM): {e}")
        return state

    # --- TOP-LEVEL ---
    if patch.additional_source_tables is not None:
        seen = set()
        dedup: List[str] = []
        for t in patch.additional_source_tables:
            if t and t not in seen:
                seen.add(t)
                dedup.append(t)
        state.additional_source_tables = dedup

    if patch.additional_source_tables_approved is not None:
        state.additional_source_tables_approved = patch.additional_source_tables_approved

    if patch.objects_to_create_approved is not None:
        state.objects_to_create_approved = patch.objects_to_create_approved

    # --- FAKTY: UPSERT PO NAZWIE ---
    if patch.facts_to_create:
        # indeks po nazwie (case-insensitive)
        facts_idx = {
            (f.fact_name or "").strip().lower(): f
            for f in state.facts_to_create
            if getattr(f, "fact_name", None)
        }

        for fpatch in patch.facts_to_create:
            name = (fpatch.fact_name or "").strip()
            if not name:
                continue
            key = name.lower()
            target = facts_idx.get(key)

            if target is None:
                # tworzymy tylko gdy mamy minimalne dane
                if fpatch.fact_main_source_table and len(fpatch.fact_main_source_table) > 0:
                    # utwórz FactToCreate z wypełnionych pól
                    new_fact = st.FactToCreate(
                        fact_name=name,
                        fact_main_source_table=fpatch.fact_main_source_table,
                        fact_other_source_tables=fpatch.fact_other_source_tables,
                        fact_connected_dimensions=fpatch.fact_connected_dimensions,
                    )
                    state.facts_to_create.append(new_fact)
                    facts_idx[key] = new_fact
                else:
                    print(f"Debug - pominięto utworzenie faktu '{name}', brak main_source_table.")
            else:
                # aktualizujemy tylko pola obecne w patchu (nie puste)
                if fpatch.fact_main_source_table is not None and fpatch.fact_main_source_table != []:
                    target.fact_main_source_table = fpatch.fact_main_source_table
                if fpatch.fact_other_source_tables is not None and fpatch.fact_other_source_tables != []:
                    target.fact_other_source_tables = fpatch.fact_other_source_tables
                if fpatch.fact_connected_dimensions is not None and fpatch.fact_connected_dimensions != []:
                    target.fact_connected_dimensions = fpatch.fact_connected_dimensions

    # --- WYMIARY: UPSERT PO NAZWIE ---
    if patch.dimensions_to_create:
        dims_idx = {
            (d.dimension_name or "").strip().lower(): d
            for d in state.dimensions_to_create
            if getattr(d, "dimension_name", None)
        }

        for dpatch in patch.dimensions_to_create:
            name = (dpatch.dimension_name or "").strip()
            if not name:
                continue
            key = name.lower()
            target = dims_idx.get(key)

            if target is None:
                if dpatch.main_source_table and len(dpatch.main_source_table) > 0:
                    new_dim = st.DimensionToCreate(
                        dimension_name=name,
                        main_source_table=dpatch.main_source_table,
                        business_key_on_source_side=dpatch.business_key_on_source_side,
                        other_source_tables=dpatch.other_source_tables,
                        surrogate_key=dpatch.surrogate_key,
                    )
                    state.dimensions_to_create.append(new_dim)
                    dims_idx[key] = new_dim
                else:
                    print(f"Debug - pominięto utworzenie wymiaru '{name}', brak main_source_table.")
            else:
                if dpatch.main_source_table is not None and dpatch.main_source_table != []:
                    target.main_source_table = dpatch.main_source_table
                if dpatch.business_key_on_source_side is not None and dpatch.business_key_on_source_side != []:
                    target.business_key_on_source_side = dpatch.business_key_on_source_side
                if dpatch.other_source_tables is not None and dpatch.other_source_tables != []:
                    target.other_source_tables = dpatch.other_source_tables
                if dpatch.surrogate_key is not None and dpatch.surrogate_key != "":
                    target.surrogate_key = dpatch.surrogate_key

    # Part related with collecting informtions about source tables
    _ensure_source_table_analyze_from_documentation(state)

    return state

class model_dimension_entity_level_update_patch(BaseModel):
    """
    Model Pydantic reprezentujący 'łatkę' (patch) dla obiektu DimensionToCreate.
    Zawiera tylko te pola, które LLM może zaktualizować na podstawie model_dimension_entity_level_update_state.
    """
    dimension_name:                         str  # mandatory
    dimension_comment:                      Optional[str]  = Field(default=None, description="Business description of dimension with information about main source table" )
    main_source_table:                      List[str]  # mandatory
    business_key_on_source_side:            Optional[List[str]] = Field(default=None, description="Point which column from source table is the busines key" )
    other_source_tables:                    Optional[List[str]] = Field(default=None, description="Point other tables which need to build compex dimension" )
    surrogate_key:                          Optional[str] = Field(default=None, description="Point surrodate key in final dim. Usually dim_<name>_key" )
    business_key_on_dimension_side:         Optional[List[str]] = Field(default=None, description="Point columns which are representing busines key on dimension side" )    
    history_type:                           Optional[str] = Field(default=None, description="Type of history: SCD1 - just current data, SCD2 - historization" )
    design_on_entity_level_approved:        bool = Field(default=False, description="Information if user approved designe for this dimension" )
def model_dimension_entity_level_update_state (state: st.DesignState) -> st.DesignState:
    """
    Aktualizuje stan wymiaru, używając `with_structured_output` do uzyskania
    zwalidowanej 'łatki' (patch) od LLM.
    """
    # 1. Przygotuj wywołanie LLM ze schematem Pydantic
    #    To jest kluczowa zmiana: "wiązemy" LLM z naszym modelem DimensionUpdatePatch
    structured_llm = conf.llm_Model_Dimension_Entity_Level_Detect_State.with_structured_output(model_dimension_entity_level_update_patch)
    
    messages = [
        SystemMessage(content=conf.Model_Dimension_Entity_Level_Detect_State_Text),
        HumanMessage(content=f"Odpowiedź użytkownika (last_5_messages): {state.last_5_messages}, "
                             f"Aktualny stan wymiaru: {dimension_as_string(state, state.currently_modeled_object)}")
    ]
    try:
        # 2. Wywołaj LLM - zamiast stringa, otrzymasz obiekt Pydantic!
        update_patch: model_dimension_entity_level_update_patch = structured_llm.invoke(messages)
        # 3. Znajdź wymiar do aktualizacji w stanie
        target_dim = next(
            (dim for dim in state.dimensions_to_create if dim.dimension_name == update_patch.dimension_name),
            None
        )
        if target_dim is None:
            # Jeśli wymiar nie istnieje, możesz go zignorować lub obsłużyć błąd
            print(f"Debug - nie znaleziono wymiaru '{update_patch.dimension_name}' do aktualizacji.")
            return state
        # 4. Zastosuj 'łatkę' do istniejącego wymiaru
        #    .model_dump(exclude_unset=True) zwraca tylko te pola, które LLM faktycznie wypełnił
        patch_dict = update_patch.model_dump(exclude_unset=True)        
        # Użyj istniejącego helpera do aktualizacji obiektu w miejscu (lub lepiej, wzorca immutable)
        _update_dimension_in_place(target_dim, patch_dict)
    except Exception as e:
        # Obsługa błędów, jeśli LLM nie zwróci poprawnych danych lub wystąpi inny problem
        print(f"Debug - Błąd w model_dimension_update_state: {e}")
    return state

class model_dimension_columns_level_update_patch(BaseModel):
    """
    Patch dla poziomu kolumn (bez twardych odniesień do st.* żeby uniknąć konfliktów klas).
    """
    dimension_name: str

    # top-level scope dla DimensionToCreate (opcjonalne, aby nie nadpisywać przez przypadek)
    other_source_tables: Optional[List[str]] = None
    design_on_columns_level_approved: Optional[bool] = None

    # reprezentacja kolumn używana TYLKO w patchu (nie st.ColumnDefinition)
    class ColumnPatch(BaseModel):
        order: Optional[int] = None
        column_name: Optional[str] = None
        column_source: Optional[str] = None
        load_logic: Optional[str] = None
        data_type: Optional[str] = None
        length: Optional[int] = None
        precision: Optional[int] = None
        column_comment: Optional[str] = None
        nullable: Optional[bool] = None
        PII: Optional[bool] = None
        column_confidentiality_level: Optional[str] = None
        PK: Optional[bool] = None
        UK: Optional[bool] = None
        FK: Optional[str] = None

    # operacje na liście kolumn
    upsert_columns: Optional[List[ColumnPatch]] = None
    remove_columns: Optional[List[str]] = None
    reorder_by: Optional[List[str]] = None
def model_dimension_columns_level_update_state(state: st.DesignState) -> st.DesignState:
    """
    LLM -> model_dimension_columns_level_update_patch -> zastosowanie do stanu.

    Reuse istniejących helperów:
    - _filter_known_fields + _update_dimension_in_place dla top-level pól wymiaru,
    - własny UPSERT/REMOVE/REORDER kolumn z konwersją ColumnPatch -> st.ColumnDefinition.
    """
    def _norm(s: Optional[str]) -> str:
        return (s or "").strip().lower()

    structured_llm = conf.llm_Model_Dimension_Columns_Level_Detect_State.with_structured_output(
        model_dimension_columns_level_update_patch
    )
    messages = [
        SystemMessage(content=conf.Model_Dimension_Columns_Level_Detect_State_Text),
        HumanMessage(content=(
            f"Odpowiedź użytkownika (last_5_messages): {state.last_5_messages}\n"
            f"Aktualny stan wymiaru: {dimension_as_string(state, state.currently_modeled_object)}"
        )),
    ]

    try:
        patch: model_dimension_columns_level_update_patch = structured_llm.invoke(messages)
    except Exception as e:
        print(f"Debug - błąd LLM w model_dimension_columns_level_update_state: {e}")
        return state

    # --- znajdź wymiar po nazwie ---
    key = _norm(patch.dimension_name)
    dim = next((d for d in state.dimensions_to_create if _norm(d.dimension_name) == key), None)
    if dim is None:
        print(f"Debug - nie znaleziono wymiaru '{patch.dimension_name}' do aktualizacji kolumn.")
        return state

    # --- TOP-LEVEL PATCH dla DimensionToCreate (reuse helperów) ---
    allowed = {"other_source_tables", "design_on_columns_level_approved"}
    incoming: Dict[str, Any] = {
        "other_source_tables": patch.other_source_tables,
        "design_on_columns_level_approved": patch.design_on_columns_level_approved,
    }
    filtered = _filter_known_fields({k: v for k, v in incoming.items() if k in allowed},
                                   include_empty={"other_source_tables"},
                                   allowed_fields=allowed)  # pozwól też czyścić listę
    if filtered:
        _update_dimension_in_place(dim, filtered)  # bezpieczny patch top-level pól  :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

    # --- OPERACJE NA KOLUMNACH ---
    if dim.detailed_column_list is None:
        dim.detailed_column_list = []

    # indeks po nazwie
    by_name: Dict[str, st.ColumnDefinition] = {
        _norm(c.column_name): c
        for c in dim.detailed_column_list
        if getattr(c, "column_name", None)
    }

    # UPSERT
    if patch.upsert_columns:
        current_orders = [c.order for c in dim.detailed_column_list if isinstance(c.order, int)]
        next_order = (max(current_orders) + 1) if current_orders else 1

        for cpatch in patch.upsert_columns:
            name_key = _norm(cpatch.column_name)
            if not name_key:
                continue

            existing = by_name.get(name_key)
            data = cpatch.model_dump(exclude_unset=True)

            if existing is None:
                if "order" not in data or data["order"] is None:
                    data["order"] = next_order
                    next_order += 1
                new_col = st.ColumnDefinition(**data)  # tu dopiero wiążemy się z aktualną klasą
                dim.detailed_column_list.append(new_col)
                by_name[name_key] = new_col
            else:
                # częściowy update tylko po polach dostarczonych przez LLM
                data.pop("column_name", None)  # nazwy nie zmieniamy w upsercie
                for k, v in data.items():
                    setattr(existing, k, v)

    # REMOVE
    if patch.remove_columns:
        to_remove = {_norm(n) for n in patch.remove_columns if _norm(n)}
        if to_remove:
            dim.detailed_column_list = [
                c for c in dim.detailed_column_list
                if _norm(getattr(c, "column_name", None)) not in to_remove
            ]
            by_name = {
                _norm(c.column_name): c
                for c in dim.detailed_column_list
                if getattr(c, "column_name", None)
            }

    # REORDER (renumeracja 1..N; nie wymaga pełnego pokrycia listy)
    if patch.reorder_by:
        desired = [_norm(n) for n in patch.reorder_by if _norm(n)]
        wanted = [by_name[k] for k in desired if k in by_name]
        rest = [c for k, c in by_name.items() if k not in set(desired)]
        rest.sort(key=lambda c: ((c.order if isinstance(c.order, int) else 10**9), _norm(c.column_name)))
        final = wanted + rest
        for i, col in enumerate(final, start=1):
            col.order = i
        dim.detailed_column_list = final

    return state

class model_dimension_create_modeling_feed_update_patch(BaseModel):
    """
    Patch dla etapu 'model_dimension_create_modeling_feed'.
    Zasady jak w model_dimension_columns_level_update_patch:
      • identyfikacja DIM po dimension_name,
      • ostrożne aktualizacje pól wymiaru, jeśli zostały dostarczone,
      • opcjonalne operacje na kolumnach (upsert/remove/reorder),
      • ustawienie feed_2_modeling_tool i feed_2_modeling_tool_approved.
    """
    # identyfikacja
    dimension_name: str

    # ewentualne korekty metadanych wymiaru dopuszczalne na tym etapie
    dimension_comment: Optional[str] = None
    main_source_table: Optional[List[str]] = None
    business_key_on_source_side: Optional[List[str]] = None
    other_source_tables: Optional[List[str]] = None
    surrogate_key: Optional[str] = None
    business_key_on_dimension_side: Optional[List[str]] = None
    history_type: Optional[str] = None

    # operacje na kolumnach – zgodnie z semantyką columns-level
    upsert_columns: Optional[List[st.ColumnDefinition]] = None
    remove_columns: Optional[List[str]] = None
    reorder_by: Optional[List[str]] = None

    # feed + akceptacja
    feed_2_modeling_tool: Optional[str] = None
    feed_2_modeling_tool_approved: Optional[bool] = None
def model_dimension_create_modeling_feed_update_state(state: st.DesignState) -> st.DesignState:
    """
    LLM -> model_dimension_create_modeling_feed_update_patch -> aktualizacja stanu:
      • bezpieczne zmiany w metadanych wymiaru,
      • opcjonalne upsert/remove/reorder kolumn,
      • ustawienie feed_2_modeling_tool i feed_2_modeling_tool_approved.
    """
    def _norm(s: Optional[str]) -> str:
        return (s or "").strip().lower()

    structured_llm = conf.llm_Model_Dimension_Create_Modeling_Feed_Detect_State.with_structured_output(
        model_dimension_create_modeling_feed_update_patch
    )

    messages = [
        SystemMessage(content=conf.Model_Dimension_Create_Modeling_Feed_Detect_State_Text),
        HumanMessage(content=(
            f"Odpowiedź użytkownika (last_5_messages): {state.last_5_messages}\n"
            f"Aktualny stan wymiaru: {dimension_as_string(state, state.currently_modeled_object)}"
        ))
    ]

    try:
        patch: model_dimension_create_modeling_feed_update_patch = structured_llm.invoke(messages)
    except Exception as e:
        print(f"Debug - błąd LLM w model_dimension_create_modeling_feed_update_state: {e}")
        return state

    # --- znajdź wymiar ---
    dim_key = _norm(patch.dimension_name)
    target_dim = next((d for d in state.dimensions_to_create if _norm(d.dimension_name) == dim_key), None)
    if target_dim is None:
        print(f"Debug - nie znaleziono wymiaru '{patch.dimension_name}' do aktualizacji feedu.")
        return state

    # --- bezpieczne aktualizacje pól wymiaru (jeśli dostarczone) ---
    if patch.dimension_comment is not None:
        target_dim.dimension_comment = patch.dimension_comment
    if patch.main_source_table is not None and patch.main_source_table != []:
        target_dim.main_source_table = patch.main_source_table
    if patch.business_key_on_source_side is not None and patch.business_key_on_source_side != []:
        target_dim.business_key_on_source_side = patch.business_key_on_source_side
    if patch.other_source_tables is not None:
        # dopuszczamy czyszczenie listy
        target_dim.other_source_tables = patch.other_source_tables
    if patch.surrogate_key is not None:
        target_dim.surrogate_key = patch.surrogate_key
    if patch.business_key_on_dimension_side is not None:
        target_dim.business_key_on_dimension_side = patch.business_key_on_dimension_side
    if patch.history_type is not None:
        target_dim.history_type = patch.history_type

    # --- operacje na kolumnach: upsert/remove/reorder ---
    if target_dim.detailed_column_list is None:
        target_dim.detailed_column_list = []

    by_name = {
        _norm(c.column_name): c
        for c in target_dim.detailed_column_list
        if getattr(c, "column_name", None)
    }

    # UPSERT
    if patch.upsert_columns:
        current_orders = [c.order for c in target_dim.detailed_column_list if isinstance(c.order, int)]
        next_order = (max(current_orders) + 1) if current_orders else 1

        for cpatch in patch.upsert_columns:
            name_raw = getattr(cpatch, "column_name", None)
            key = _norm(name_raw)
            if not key:
                continue

            existing = by_name.get(key)
            if existing is None:
                new_kwargs = cpatch.model_dump(exclude_unset=True)
                if new_kwargs.get("order") is None:
                    new_kwargs["order"] = next_order
                    next_order += 1
                new_col = st.ColumnDefinition(**new_kwargs)
                target_dim.detailed_column_list.append(new_col)
                by_name[key] = new_col
            else:
                updates = cpatch.model_dump(exclude_unset=True)
                updates.pop("column_name", None)
                for k, v in updates.items():
                    setattr(existing, k, v)

    # REMOVE
    if patch.remove_columns:
        to_remove = {_norm(n) for n in patch.remove_columns if _norm(n)}
        if to_remove:
            target_dim.detailed_column_list = [
                c for c in target_dim.detailed_column_list
                if _norm(getattr(c, "column_name", None)) not in to_remove
            ]
            by_name = {
                _norm(c.column_name): c
                for c in target_dim.detailed_column_list
                if getattr(c, "column_name", None)
            }

    # REORDER
    if patch.reorder_by:
        desired_keys = [_norm(n) for n in patch.reorder_by if _norm(n)]
        desired_set = set(desired_keys)

        ordered = [by_name[k] for k in desired_keys if k in by_name]
        remaining = [c for k, c in by_name.items() if k not in desired_set]
        remaining.sort(key=lambda c: ((c.order if isinstance(c.order, int) else 10**9), _norm(c.column_name)))

        final = ordered + remaining
        for i, col in enumerate(final, start=1):
            col.order = i
        target_dim.detailed_column_list = final

    # --- feed + akceptacja ---
    if patch.feed_2_modeling_tool is not None:
        target_dim.feed_2_modeling_tool = patch.feed_2_modeling_tool
    if patch.feed_2_modeling_tool_approved is not None:
        target_dim.feed_2_modeling_tool_approved = patch.feed_2_modeling_tool_approved

    return state

class model_dimension_scripts_creation_update_patch(BaseModel):
    """
    Patch dla etapu 'model_dimension_scripts_creation'.
    Aktualizuje wyłącznie poniższe pola w DimensionToCreate.
    """
    dimension_name:  str  # identyfikacja wymiaru
    ddl:             Optional[str] = Field(default=None, description="Definition of DDL script")
    sql:             Optional[str] = Field(default=None, description="Definition of SQL script")
    scripts_approved: Optional[bool] = Field(
        default=None, 
        description="Has user approved scripts (ddl/sql)?"
    )
def model_dimension_scripts_creation_update_state(state: st.DesignState) -> st.DesignState:
    """
    LLM -> model_dimension_scripts_creation_update_patch -> bezpieczne zastosowanie do stanu.
    Reuse: _filter_known_fields + _update_dimension_in_place (pozwalamy też czyścić ddl/sql).
    """
    def _norm(s: Optional[str]) -> str:
        return (s or "").strip().lower()

    # 1) LLM z Pydantic structured output (analogicznie do innych etapów)
    structured_llm = conf.llm_Model_Dimension_Scripts_Creation_Detect_State.with_structured_output(
        model_dimension_scripts_creation_update_patch
    )
    messages = [
        SystemMessage(content=conf.Model_Dimension_Scripts_Creation_Detect_State_Text),
        HumanMessage(content=(
            f"Odpowiedź użytkownika (last_5_messages): {state.last_5_messages}\n"
            f"Aktualny stan wymiaru: {dimension_as_string(state, state.currently_modeled_object)}"
        )),
    ]

    try:
        patch: model_dimension_scripts_creation_update_patch = structured_llm.invoke(messages)
    except Exception as e:
        print(f"Debug - błąd LLM w model_dimension_scripts_creation_update_state: {e}")
        return state

    # 2) Znajdź wymiar
    dim_key = _norm(patch.dimension_name)
    dim = next((d for d in state.dimensions_to_create if _norm(d.dimension_name) == dim_key), None)
    if dim is None:
        print(f"Debug - nie znaleziono wymiaru '{patch.dimension_name}' do aktualizacji skryptów.")
        return state

    # 3) Zastosuj bezpieczny patch (pozwalamy usuwać treść ddl/sql – include_empty)
    allowed = {"ddl", "sql", "scripts_approved"}
    incoming: Dict[str, Any] = {
        "ddl": patch.ddl,
        "sql": patch.sql,
        "scripts_approved": patch.scripts_approved,
    }
    filtered = _filter_known_fields(
        {k: v for k, v in incoming.items() if k in allowed},
        include_empty={"ddl", "sql"},
        allowed_fields=allowed
    )
    if filtered:
        _update_dimension_in_place(dim, filtered)

    return state

class model_fact_entity_level_update_patch(BaseModel):
    """
    Model Pydantic reprezentujący 'łatkę' (patch) dla obiektu FactToCreate.
    Zawiera tylko te pola, które LLM może zaktualizować na podstawie model_fact_entity_level_update_state.
    """
    # Pola identyfikacyjne / wymagane
    fact_name:                      str  # mandatory
    fact_main_source_table:         List[str]  # mandatory

    # Pola opcjonalne
    fact_comment:                   Optional[str] = Field(default=None, description="Business description of fact with information about main source table"  )
    fact_other_source_tables:       Optional[List[str]] = Field( default=None, description="Point other tables which need to build complex fact"  )
    fact_rls_collumns:              Optional[List[str]] = Field( default=None, description="Which columns will allow row level security filtration (rls)"  )
    fact_unique_key:                Optional[str] = Field( default=None, description="Point which column from source table is the business key" )
    fact_type:                      Optional[str] = Field( default=None, description="Type of fact: transactional (no history), ranges (valid_from, valid_to), snapshot (valid_in), accumulating (valid_from)"   )
    fact_design_on_entity_level_approved:   bool = Field(default=False, description="Information if user approved designe for this fact" )
def model_fact_entity_level_update_state(state: "st.DesignState") -> "st.DesignState":
    """
    Aktualizuje stan faktu, używając `with_structured_output` do uzyskania
    zwalidowanej 'łatki' (patch) od LLM — analogicznie do wariantu dla wymiarów.
    """
    # 1. Przygotuj wywołanie LLM ze schematem Pydantic (analogiczne do Dimension)
    structured_llm = conf.llm_Model_Fact_Entity_Level_Detect_State.with_structured_output(model_fact_entity_level_update_patch)

    # Przygotowanie tekstowej reprezentacji bieżącego stanu (bezpieczny fallback, aby uniknąć NameError)
    try:
        current_state_text = fact_as_string(state, state.currently_modeled_object)  # jeśli masz taką funkcję
    except Exception:
        try:
            # Minimalna, ale czytelna reprezentacja
            current_state_text = (
                "facts_to_create: " +
                str([getattr(f, "fact_name", None) for f in getattr(state, "facts_to_create", [])])
            )
        except Exception:
            current_state_text = "current state not stringified"

    messages = [
        SystemMessage(content=conf.Model_Fact_Entity_Level_Detect_State_Text),
        HumanMessage(content=(
            f"Odpowiedź użytkownika (last_5_messages): {getattr(state, 'last_5_messages', None)}, "
            f"Aktualny stan faktu: {current_state_text}"
        ))
    ]

    try:
        # 2. Wywołaj LLM - zamiast stringa, otrzymasz obiekt Pydantic!
        update_patch: model_fact_entity_level_update_patch = structured_llm.invoke(messages)

        # 3. Znajdź fact do aktualizacji w stanie
        facts = getattr(state, "facts_to_create", None)
        if facts is None:
            print("Debug - 'state.facts_to_create' nie istnieje.")
            return state

        target_fact = next(
            (f for f in facts if getattr(f, "fact_name", None) == update_patch.fact_name),
            None
        )

        if target_fact is None:
            print(f"Debug - nie znaleziono faktu '{update_patch.fact_name}' do aktualizacji.")
            return state

        # 4. Zastosuj 'łatkę' do istniejącego faktu
        patch_dict = update_patch.model_dump(exclude_unset=True)

        # Z reguły nie chcemy zmieniać fakt_name przy aktualizacji (służy do identyfikacji)
        patch_dict.pop("fact_name", None)

        # Bezpieczna próba użycia helpera, a jeśli go nie ma — prosta aktualizacja atrybutów
        try:
            _update_fact_in_place(target_fact, patch_dict)  # jeśli masz analogiczny helper jak dla wymiarów
        except NameError:
            for k, v in patch_dict.items():
                try:
                    setattr(target_fact, k, v)
                except Exception as e:
                    print(f"Debug - nie udało się ustawić '{k}' na '{v}': {e}")

    except Exception as e:
        # Obsługa błędów, jeśli LLM nie zwróci poprawnych danych lub wystąpi inny problem
        print(f"Debug - Błąd w model_fact_entity_level_update_state: {e}")

    return state

class model_fact_columns_level_update_patch(BaseModel):
    """
    Patch dla etapu 'model_fact_columns_level'.
    Zachowuje semantykę jak dla wymiarów:
      • identyfikacja faktu po fact_name,
      • ostrożne aktualizacje pól top-level,
      • operacje na kolumnach: upsert/remove/reorder (z renumeracją order 1..N).
    """
    # identyfikacja
    fact_name: str

    # top-level (pozwalamy też na wyczyszczenie listy connected_dimensions)
    fact_connected_dimensions: Optional[List[str]] = None
    fact_design_on_columns_level_approved: Optional[bool] = None

    # reprezentacja kolumn używana TYLKO w patchu (nie wiążemy się od razu z st.ColumnDefinition)
    class ColumnPatch(BaseModel):
        order: Optional[int] = None
        column_name: Optional[str] = None
        column_source: Optional[str] = None
        load_logic: Optional[str] = None
        data_type: Optional[str] = None
        length: Optional[int] = None
        precision: Optional[int] = None
        column_comment: Optional[str] = None
        nullable: Optional[bool] = None
        PII: Optional[bool] = None
        column_confidentiality_level: Optional[str] = None
        PK: Optional[bool] = None
        UK: Optional[bool] = None
        FK: Optional[str] = None

    # operacje na liście kolumn
    upsert_columns: Optional[List[ColumnPatch]] = None
    remove_columns: Optional[List[str]] = None
    reorder_by: Optional[List[str]] = None
def model_fact_columns_level_update_state(state: st.DesignState) -> st.DesignState:
    """
    LLM -> model_fact_columns_level_update_patch -> aktualizacja stanu faktu:
      • bezpieczne zmiany pól top-level (wprost, by umożliwić czyszczenie listy),
      • UPSERT/REMOVE/REORDER na fact_column_list.
    """
    def _norm(s: Optional[str]) -> str:
        return (s or "").strip().lower()

    # Tekst stanu dla promptu (jak w entity-level, z fallbackiem)
    try:
        current_state_text = fact_as_string(state, state.currently_modeled_object)
    except Exception:
        try:
            current_state_text = "facts_to_create: " + str([
                getattr(f, "fact_name", None) for f in getattr(state, "facts_to_create", [])
            ])
        except Exception:
            current_state_text = "current state not stringified"

    structured_llm = conf.llm_Model_Fact_Columns_Level_Detect_State.with_structured_output(
        model_fact_columns_level_update_patch
    )
    messages = [
        SystemMessage(content=conf.Model_Fact_Columns_Level_Detect_State_Text),
        HumanMessage(content=(
            f"Odpowiedź użytkownika (last_5_messages): {state.last_5_messages}\n"
            f"Aktualny stan faktu: {current_state_text}"
        )),
    ]

    try:
        patch: model_fact_columns_level_update_patch = structured_llm.invoke(messages)
    except Exception as e:
        print(f"Debug - błąd LLM w model_fact_columns_level_update_state: {e}")
        return state

    # --- znajdź fakt po nazwie (case-insensitive) ---
    key = _norm(patch.fact_name)
    fact = next((f for f in state.facts_to_create if _norm(f.fact_name) == key), None)
    if fact is None:
        print(f"Debug - nie znaleziono faktu '{patch.fact_name}' do aktualizacji kolumn.")
        return state

    # --- TOP-LEVEL PATCH dla FactToCreate (bezpośrednio, bo _filter_known_fields jest dla Dimension) ---
    if patch.fact_connected_dimensions is not None:
        fact.fact_connected_dimensions = patch.fact_connected_dimensions  # pozwala też wyczyścić []
    if patch.fact_design_on_columns_level_approved is not None:
        fact.fact_design_on_columns_level_approved = patch.fact_design_on_columns_level_approved

    # --- OPERACJE NA KOLUMNACH (fact.fact_column_list) ---
    if fact.fact_column_list is None:
        fact.fact_column_list = []

    # indeks po nazwie
    by_name: Dict[str, st.ColumnDefinition] = {
        _norm(c.column_name): c
        for c in fact.fact_column_list
        if getattr(c, "column_name", None)
    }

    # UPSERT
    if patch.upsert_columns:
        current_orders = [c.order for c in fact.fact_column_list if isinstance(c.order, int)]
        next_order = (max(current_orders) + 1) if current_orders else 1

        for cpatch in patch.upsert_columns:
            name_key = _norm(cpatch.column_name)
            if not name_key:
                continue
            existing = by_name.get(name_key)
            data = cpatch.model_dump(exclude_unset=True)

            if existing is None:
                if "order" not in data or data["order"] is None:
                    data["order"] = next_order
                    next_order += 1
                new_col = st.ColumnDefinition(**data)
                fact.fact_column_list.append(new_col)
                by_name[name_key] = new_col
            else:
                data.pop("column_name", None)  # nazwy nie zmieniamy przy upsercie
                for k, v in data.items():
                    setattr(existing, k, v)

    # REMOVE
    if patch.remove_columns:
        to_remove = {_norm(n) for n in patch.remove_columns if _norm(n)}
        if to_remove:
            fact.fact_column_list = [
                c for c in fact.fact_column_list
                if _norm(getattr(c, "column_name", None)) not in to_remove
            ]
            by_name = {
                _norm(c.column_name): c
                for c in fact.fact_column_list
                if getattr(c, "column_name", None)
            }

    # REORDER (renumeracja 1..N; nie wymaga pełnego pokrycia listy)
    if patch.reorder_by:
        desired = [_norm(n) for n in patch.reorder_by if _norm(n)]
        wanted = [by_name[k] for k in desired if k in by_name]
        rest = [c for k, c in by_name.items() if k not in set(desired)]
        rest.sort(key=lambda c: ((c.order if isinstance(c.order, int) else 10**9), _norm(c.column_name)))
        final = wanted + rest
        for i, col in enumerate(final, start=1):
            col.order = i
        fact.fact_column_list = final

    return state

class model_fact_create_modeling_feed_update_patch(BaseModel):
    """
    Patch dla etapu 'model_fact_create_modeling_feed'.
    Zakres:
      • identyfikacja faktu po fact_name,
      • opcjonalny upsert/remove/reorder kolumn (na fact_column_list),
      • ustawienie fact_feed_2_modeling_tool i fact_feed_2_modeling_tool_approved.
    """
    # identyfikacja
    fact_name: str

    # operacje na kolumnach (tutaj dopuszczamy bezpośrednio st.ColumnDefinition jak w DIM feed)
    upsert_columns: Optional[List[st.ColumnDefinition]] = None
    remove_columns: Optional[List[str]] = None
    reorder_by: Optional[List[str]] = None

    # feed + akceptacja
    fact_feed_2_modeling_tool: Optional[str] = None
    fact_feed_2_modeling_tool_approved: Optional[bool] = None
def model_fact_create_modeling_feed_update_state(state: st.DesignState) -> st.DesignState:
    """
    LLM -> model_fact_create_modeling_feed_update_patch -> aktualizacja stanu faktu:
      • opcjonalne upsert/remove/reorder na fact_column_list,
      • ustawienie fact_feed_2_modeling_tool i fact_feed_2_modeling_tool_approved.
    """
    def _norm(s: Optional[str]) -> str:
        return (s or "").strip().lower()

    # Tekst stanu dla promptu (z fallbackiem)
    try:
        current_state_text = fact_as_string(state, state.currently_modeled_object)
    except Exception:
        try:
            current_state_text = "facts_to_create: " + str([
                getattr(f, "fact_name", None) for f in getattr(state, "facts_to_create", [])
            ])
        except Exception:
            current_state_text = "current state not stringified"

    structured_llm = conf.llm_Model_Fact_Create_Modeling_Feed_Detect_State.with_structured_output(
        model_fact_create_modeling_feed_update_patch
    )
    messages = [
        SystemMessage(content=conf.Model_Fact_Create_Modeling_Feed_Detect_State_Text),
        HumanMessage(content=(
            f"Odpowiedź użytkownika (last_5_messages): {state.last_5_messages}\n"
            f"Aktualny stan faktu: {current_state_text}"
        )),
    ]

    try:
        patch: model_fact_create_modeling_feed_update_patch = structured_llm.invoke(messages)
    except Exception as e:
        print(f"Debug - błąd LLM w model_fact_create_modeling_feed_update_state: {e}")
        return state

    # --- znajdź fakt po nazwie ---
    key = _norm(patch.fact_name)
    fact = next((f for f in state.facts_to_create if _norm(f.fact_name) == key), None)
    if fact is None:
        print(f"Debug - nie znaleziono faktu '{patch.fact_name}' do aktualizacji feedu.")
        return state

    # --- operacje na kolumnach: upsert/remove/reorder ---
    if fact.fact_column_list is None:
        fact.fact_column_list = []

    by_name: Dict[str, st.ColumnDefinition] = {
        _norm(c.column_name): c
        for c in fact.fact_column_list
        if getattr(c, "column_name", None)
    }

    # UPSERT
    if patch.upsert_columns:
        current_orders = [c.order for c in fact.fact_column_list if isinstance(c.order, int)]
        next_order = (max(current_orders) + 1) if current_orders else 1

        for cpatch in patch.upsert_columns:
            name_key = _norm(cpatch.column_name)
            if not name_key:
                continue

            existing = by_name.get(name_key)
            data = cpatch.model_dump(exclude_unset=True)

            if existing is None:
                if "order" not in data or data["order"] is None:
                    data["order"] = next_order
                    next_order += 1
                new_col = st.ColumnDefinition(**data)
                fact.fact_column_list.append(new_col)
                by_name[name_key] = new_col
            else:
                data.pop("column_name", None)
                for k, v in data.items():
                    setattr(existing, k, v)

    # REMOVE
    if patch.remove_columns:
        to_remove = {_norm(n) for n in patch.remove_columns if _norm(n)}
        if to_remove:
            fact.fact_column_list = [
                c for c in fact.fact_column_list
                if _norm(getattr(c, "column_name", None)) not in to_remove
            ]
            by_name = {
                _norm(c.column_name): c
                for c in fact.fact_column_list
                if getattr(c, "column_name", None)
            }

    # REORDER
    if patch.reorder_by:
        desired_keys = [_norm(n) for n in patch.reorder_by if _norm(n)]
        desired_set = set(desired_keys)

        ordered = [by_name[k] for k in desired_keys if k in by_name]
        remaining = [c for k, c in by_name.items() if k not in desired_set]
        remaining.sort(key=lambda c: ((c.order if isinstance(c.order, int) else 10**9), _norm(c.column_name)))

        final = ordered + remaining
        for i, col in enumerate(final, start=1):
            col.order = i
        fact.fact_column_list = final

    # --- feed + akceptacja ---
    if patch.fact_feed_2_modeling_tool is not None:
        fact.fact_feed_2_modeling_tool = patch.fact_feed_2_modeling_tool
    if patch.fact_feed_2_modeling_tool_approved is not None:
        fact.fact_feed_2_modeling_tool_approved = patch.fact_feed_2_modeling_tool_approved

    return state

class model_fact_scripts_creation_update_patch(BaseModel):
    fact_name: str
    fact_ddl: Optional[str] = None
    fact_sql: Optional[str] = None
    fact_scripts_approved: Optional[bool] = None
def model_fact_scripts_creation_update_state(state: st.DesignState):
    if not state.last_user_message:
        return state
    try:
        data = json.loads(state.last_user_message)
    except Exception:
        return state

    patch = model_fact_scripts_creation_update_patch(**data)
    fact = next((f for f in state.facts_to_create
                 if f.fact_name == (patch.fact_name or state.currently_modeled_object)), None)
    if not fact:
        return state

    if patch.fact_ddl is not None: fact.fact_ddl = patch.fact_ddl
    if patch.fact_sql is not None: fact.fact_sql = patch.fact_sql
    if patch.fact_scripts_approved is not None:
        fact.fact_scripts_approved = patch.fact_scripts_approved
    return state


# ---------------------------------------------------------------------------
# 3. Nodes
# ---------------------------------------------------------------------------
# --- ROUTER (NODE) ---
def router_node(state: st.DesignState, config):
    return {}  # zawsze diff
def router_decider(state: st.DesignState) -> Optional[str]:

    # 1) Jeśli czekamy na odpowiedź, wróć do tego węzła
    if state.awaiting_input_for:
        return state.awaiting_input_for

    # 2) Wymagania wejściowe
    if not (state.required_source_tables and state.required_source_tables_approved):
        return "requirements_collection"  

    # 3) Analiza wymagań (gdy brak obiektów do stworzenia lub niezatwierdzona lista)
    if (not (state.facts_to_create or state.dimensions_to_create)) or (not state.objects_to_create_approved):
        return "requirements_analysis"

    # 4) Checkif we still modeling something (dimension or fact)
    if state.currently_modeled_object:
        # ) Check if we are modeling dimension
        if state.currently_modeled_object and state.currently_modeled_object.lower().startswith('dim'):
            # znajdź dokładnie ten wymiar
            dimension_to_continue = next(
                (dim for dim in state.dimensions_to_create
                if dim.dimension_name and dim.dimension_name.lower() == state.currently_modeled_object.lower()),
                None
            )
            # jeśli znaleziony i ma cokolwiek do dokończenia -> wejdź w odpowiedni krok
            if dimension_to_continue and _has_any_dim_elements_unapproved(dimension_to_continue):
                if dimension_to_continue.design_on_entity_level_approved is False:
                    return "model_dimension_entity_level"
                elif dimension_to_continue.design_on_columns_level_approved is False:
                    return "model_dimension_columns_level"
                elif dimension_to_continue.feed_2_modeling_tool_approved is False:
                    return "model_dimension_create_modeling_feed"
                elif dimension_to_continue.scripts_approved is False:
                    return "model_dimension_scripts_creation"
        # ) Check if we are modeling fact               
        if state.currently_modeled_object and state.currently_modeled_object.lower().startswith('fact'):
            # znajdź dokładnie ten fact
            fact_to_continue = next(
                (f for f in state.facts_to_create
                if f.fact_name and f.fact_name.lower() == state.currently_modeled_object.lower()),
                None
            )
             # jeśli znaleziony i ma cokolwiek do dokończenia -> wejdź w odpowiedni krok
            if fact_to_continue and _has_any_fact_elements_unapproved(fact_to_continue):
                if fact_to_continue.fact_design_on_entity_level_approved is False:
                    return "model_fact_entity_level"
                elif fact_to_continue.fact_design_on_columns_level_approved is False:
                    return "model_fact_columns_level"
                elif fact_to_continue.fact_feed_2_modeling_tool_approved is False:
                    return "model_fact_create_modeling_feed"
                elif fact_to_continue.fact_scripts_approved is False:
                    return "model_fact_scripts_creation"  
    
    new_dimension_candidate = next(
        (dim for dim in state.dimensions_to_create
        if dim.dimension_name and (not state.currently_modeled_object or dim.dimension_name.lower() != state.currently_modeled_object.lower()) and _all_dim_elements_unapproved(dim)),
        None
    )
    new_fact_candidate = next(
        (fact for fact in state.facts_to_create
        if fact.fact_name and (not state.currently_modeled_object or fact.fact_name.lower() != state.currently_modeled_object.lower()) and _all_fact_elements_unapproved(fact)),
        None
    )

    # 5) Chceck if we have any fresh dim to model
    if new_dimension_candidate:
        state.currently_modeled_object = new_dimension_candidate.dimension_name
        return "model_dimension_entity_level"

    # 6) Chceck if we have any fresh fact to model
    if new_fact_candidate:
        state.currently_modeled_object = new_fact_candidate.fact_name
        return "model_fact_entity_level"

    # 7) Domyślnie nic do zrobienia        
    return END  

def node_requirements_collection(state: st.DesignState, config) -> st.DesignState | Interrupt:
    session_id = config["configurable"]["session_id"]
    thread_id = config["configurable"]["thread_id"]
    memory_key = f"{session_id}:{thread_id}"
    # Historia rozmowy
    if memory_key not in memory_store:
        memory_store[memory_key] = ChatMessageHistory()
    history = memory_store[memory_key]
    # Dodaj wiadomość użytkownika do historii
    if state.last_user_message:
        history.add_user_message(state.last_user_message)
        state.last_5_messages.append(f"🧑 {state.last_user_message}")
        requirements_collection_update_state(state)
        state.last_user_message = None  # Wyczyść po interpretacji
        state.awaiting_input_for = None # Wyczyść po interpretacji

    # EARLY EXIT – jeśli już mamy komplet, nie generuj nowej wypowiedzi. Zabezpieczenie
    if state.required_source_tables and state.required_source_tables_approved:
        state.awaiting_input_for = None
        return state

    # Zbuduj prompt z obecnym stanem
    current_data = {
        "required_source_tables": state.required_source_tables,
        "required_source_tables_approved": state.required_source_tables_approved,
        "last_5_messages": state.last_5_messages
    }
    messages = conf.REQUIREMENTS_COLLECTION_PROMPT_TEMPLATE.format(current_data=json.dumps(current_data, ensure_ascii=False, indent=2))

    reply = conf.llm_Requirements_Collection.invoke(messages).content.strip()
    history.add_ai_message(reply)
    state.last_5_messages.append(f"🤖 {reply}")
    state.last_5_messages = state.last_5_messages[-5:]  # Trzymaj tylko 10 ostatnich

    requirements_collection_update_state(state)

    # Czy wszystko co potrzebujemy zostało zebrane?
    if not (state.required_source_tables and state.required_source_tables_approved):
        state.awaiting_input_for = "requirements_collection"
        return interrupt({
            "message": sanitize_message_for_ui(reply),
            "next_state": state
        })
        
    return state

def node_requirements_analysis(state: st.DesignState, config) -> st.DesignState | Interrupt:
    session_id = config["configurable"]["session_id"]
    thread_id = config["configurable"]["thread_id"]
    memory_key = f"{session_id}:{thread_id}"
    # Historia rozmowy
    if memory_key not in memory_store:
        memory_store[memory_key] = ChatMessageHistory()
    history = memory_store[memory_key]
    # Dodaj wiadomość użytkownika do historii
    if state.last_user_message:
        history.add_user_message(state.last_user_message)
        state.last_5_messages.append(f"🧑 {state.last_user_message}")
        requirements_analysis_update_state(state)
        state.last_user_message = None  # Wyczyść po interpretacji
        state.awaiting_input_for = None # Wyczyść po interpretacji

    # EARLY EXIT – nic nie mów, jeśli etap już zamknięty
    if (state.facts_to_create or state.dimensions_to_create) and state.objects_to_create_approved:
        state.awaiting_input_for = None
        return state
    
    def to_plain(obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()            # Pydantic v2
        if isinstance(obj, list):
            return [to_plain(x) for x in obj]
        if isinstance(obj, dict):
            return {k: to_plain(v) for k, v in obj.items()}
        return obj  

    _ensure_source_table_analyze_from_documentation(state)

    # Zbuduj prompt z obecnym stanem
    current_data = {
        "required_source_tables": state.required_source_tables,
        "additional_source_tables": state.additional_source_tables ,
        "available_dimensions_essence": state.available_dimensions_essence,
        "facts_to_create": state.facts_to_create,
        "dimensions_to_create": state.dimensions_to_create,
        "additional_source_tables_approved": state.additional_source_tables_approved ,    
        "objects_to_create_approved": state.objects_to_create_approved,
        "last_5_messages": state.last_5_messages,
        "source_table_analyze": state.source_table_analyze
    }

    current_data_plain = to_plain(current_data)  # <-- KONWERSJA
    human_msg = HumanMessage(
        content="Stan wejściowy:\n" + json.dumps(current_data_plain, ensure_ascii=False, indent=2)
    )
  
    system_msg = SystemMessage(content=conf.REQUIREMENTS_ANALYSIS_PROMPT_TEXT)
    reply = conf.llm_Requirements_Analysis.invoke([system_msg, human_msg]).content.strip()
  
    history.add_ai_message(reply)
    state.last_5_messages.append(f"🤖 {reply}")
    state.last_5_messages = state.last_5_messages[-5:]  # Trzymaj tylko 5 ostatnich

    #requirements_analysis_update_state(state)

    # Czy wszystko co potrzebujemy zostało zebrane?
    if not ((state.facts_to_create or state.dimensions_to_create) and state.objects_to_create_approved):
        state.awaiting_input_for = "requirements_analysis"
        return interrupt({"message": sanitize_message_for_ui(reply), "next_state": state})
    state.awaiting_input_for = None
    return state

def node_model_dimension_entity_level(state: st.DesignState, config) -> st.DesignState | Interrupt:
    session_id = config["configurable"]["session_id"]
    thread_id = config["configurable"]["thread_id"]
    memory_key = f"{session_id}:{thread_id}"
    # Historia rozmowy
    if memory_key not in memory_store:
        memory_store[memory_key] = ChatMessageHistory()
    history = memory_store[memory_key]
    # Dodaj wiadomość użytkownika do historii
    if state.last_user_message:
        history.add_user_message(state.last_user_message)
        state.last_5_messages.append(f"🧑 {state.last_user_message}")
        state.last_5_messages = state.last_5_messages[-5:]  # Trzymaj tylko 5 ostatnich
        model_dimension_entity_level_update_state(state)
        state.last_user_message = None  # Wyczyść po interpretacji
        state.awaiting_input_for = None # Wyczyść po interpretacji    

    if not state.currently_modeled_object:
        # You do not have any currently_modeled_object, take first new
        new_dimension_candidate = next(
            (dim for dim in state.dimensions_to_create
            if dim.dimension_name
            and (not state.currently_modeled_object
                or dim.dimension_name.lower() != state.currently_modeled_object.lower())
            and _all_dim_elements_unapproved(dim)),
            None
        )
        if new_dimension_candidate:
            state.currently_modeled_object = new_dimension_candidate.dimension_name   
        else:
            return interrupt({
                "message": "?!?",
                "next_state": state
            })

    #Find currently modeled dimension
    dimension = next(
        (dim for dim in state.dimensions_to_create if dim.dimension_name == state.currently_modeled_object), 
        None
    )
    # early-exit chceck if currently modeled dimension has complited on entity level
    if ( dimension.design_on_entity_level_approved ):
        # chcek if you have any other dimension to create
            #else:
            #no dimension to create
        state.awaiting_input_for = None
        return state

    dimension_state_txt = dimension_as_string(state, state.currently_modeled_object) 
    human_msg = HumanMessage(content=(
        f"Modeluj wymiar: {state.currently_modeled_object}\n"
        f"Na podstawie dialogu: {state.last_5_messages}\n"
        f"Aktualny stan wymiaru: \n{dimension_state_txt}\n"
    ))
    system_msg = SystemMessage(
        content=conf.Model_Dimension_Entity_Level_Text
    )    
    reply = conf.llm_Model_Dimension_Entity_Level.invoke([system_msg, human_msg]).content.strip()
    state.awaiting_input_for = "model_dimension_entity_level"
    state.last_5_messages.append(f"🤖 {reply}")
    state.last_5_messages = state.last_5_messages[-5:]  # Trzymaj tylko 5 ostatnich
    return interrupt({"message": sanitize_message_for_ui(reply), "next_state": state})    

def node_model_dimension_columns_level(state: st.DesignState, config) -> st.DesignState | Interrupt:
    session_id = config["configurable"]["session_id"]
    thread_id = config["configurable"]["thread_id"]
    memory_key = f"{session_id}:{thread_id}"
    # Historia rozmowy
    if memory_key not in memory_store:
        memory_store[memory_key] = ChatMessageHistory()
    history = memory_store[memory_key]
    # Dodaj wiadomość użytkownika do historii
    if state.last_user_message:
        history.add_user_message(state.last_user_message)
        state.last_5_messages.append(f"🧑 {state.last_user_message}")
        state.last_5_messages = state.last_5_messages[-5:]  # Trzymaj tylko 5 ostatnich
        model_dimension_columns_level_update_state(state)
        state.last_user_message = None  # Wyczyść po interpretacji
        state.awaiting_input_for = None # Wyczyść po interpretacji
    # focus on currently_modeled_object
    dimension = next(
        (dim for dim in state.dimensions_to_create if dim.dimension_name == state.currently_modeled_object), 
        None
    )
    # early-exit - chceck if currently modeled dimension has complited on columns level
    if ( dimension.design_on_columns_level_approved ):
        state.awaiting_input_for = None
        return state

    # Sprawdzamy jego wszystkie tabele źródłowe
    source_tables = (dimension.main_source_table or []) + (dimension.other_source_tables or [])    
    # Build critical_columns_analyze for this dimension if does not exist
    if not (dimension.critical_columns_analyze_txt and dimension.critical_columns_analyze_txt.strip()):
        # Zbuduj mapę: lower_key -> oryginalny_klucz
        key_map = {str(k).strip().lower(): k for k in state.source_table_analyze.keys()}
        parts = []
        for table in source_tables:
            t_norm = str(table).strip().lower()
            if t_norm in key_map:
                real_key = key_map[t_norm]                     # faktyczny klucz w dict
                val = state.source_table_analyze.get(real_key) # tekst analizy
                if val:                                        # pomiń puste
                    parts.append(f"### {table} table analyze: \n{val}")
        dimension.source_tables_analyze_txt = "\n\n".join(parts)
        dimension.critical_columns_analyze_txt = point_critical_columns_in_source(
            dimension.dimension_name,
            source_tables,
            dimension.source_tables_analyze_txt
        )
        
    dimension_state_txt = dimension_as_string(state, state.currently_modeled_object) 
    human_msg = HumanMessage(content=(
        f"Modeluj wymiar: {state.currently_modeled_object}\n"
        f"Na podstawie dialogu: {state.last_5_messages}\n"
        f"Aktualny stan wymiaru: \n{dimension_state_txt}\n"
        f"SAP to Snowflake data types mapping: \n{state.SAP_2_Snowflake_data_types}\n"
    ))
    system_msg = SystemMessage(
        content=conf.Model_Dimension_Columns_Level_Text
    )    
    reply = conf.llm_Model_Dimension_Columns_Level.invoke([system_msg, human_msg]).content.strip()
    state.awaiting_input_for = "model_dimension_columns_level"
    state.last_5_messages.append(f"🤖 {reply}")
    state.last_5_messages = state.last_5_messages[-5:]  # Trzymaj tylko 5 ostatnich
    return interrupt({"message": sanitize_message_for_ui(reply), "next_state": state})    

def node_model_dimension_create_modeling_feed(state: st.DesignState, config) -> st.DesignState | Interrupt:
    session_id = config["configurable"]["session_id"]
    thread_id = config["configurable"]["thread_id"]
    memory_key = f"{session_id}:{thread_id}"
    if memory_key not in memory_store:
        memory_store[memory_key] = ChatMessageHistory()
    history = memory_store[memory_key]

    if state.last_user_message:
        history.add_user_message(state.last_user_message)
        state.last_5_messages.append(f"🧑 {state.last_user_message}")
        state.last_5_messages = state.last_5_messages[-5:]
        model_dimension_create_modeling_feed_update_state(state)
        state.last_user_message = None
        state.awaiting_input_for = None
    # focus on currently_modeled_object
    dimension = next(
        (dim for dim in state.dimensions_to_create if dim.dimension_name == state.currently_modeled_object), 
        None
    )
    # early-exit - chceck if currently modeled dimension has complited on feed to modeling tool
    if ( dimension.feed_2_modeling_tool_approved ):
        state.awaiting_input_for = None
        return state
    dimension_state_txt = dimension_as_string(state, state.currently_modeled_object)
    human_msg = HumanMessage(content=(
        f"Przygotuj feed do narzędzia modelowania dla: {state.currently_modeled_object}\n"
        f"Na podstawie dialogu: {state.last_5_messages}\n"
        f"Aktualny stan wymiaru:\n{dimension_state_txt}\n"
    ))
    system_msg = SystemMessage(content=conf.Model_Dimension_Create_Modeling_Feed_Text)
    reply = conf.llm_Model_Dimension_Create_Modeling_Feed.invoke([system_msg, human_msg]).content.strip()
    state.awaiting_input_for = "model_dimension_create_modeling_feed"
    state.last_5_messages.append(f"🤖 {reply}")
    state.last_5_messages = state.last_5_messages[-5:]  # Trzymaj tylko 5 ostatnich
    return interrupt({"message": sanitize_message_for_ui(reply), "next_state": state})

def node_model_dimension_scripts_creation(state: st.DesignState, config) -> st.DesignState | Interrupt:
    session_id = config["configurable"]["session_id"]
    thread_id = config["configurable"]["thread_id"]
    memory_key = f"{session_id}:{thread_id}"
    # Historia rozmowy
    if memory_key not in memory_store:
        memory_store[memory_key] = ChatMessageHistory()
    history = memory_store[memory_key]
    # Dodaj wiadomość użytkownika do historii
    if state.last_user_message:
        history.add_user_message(state.last_user_message)
        state.last_5_messages.append(f"🧑 {state.last_user_message}")
        state.last_5_messages = state.last_5_messages[-5:]  # Trzymaj tylko 5 ostatnich
        model_dimension_scripts_creation_update_state(state)
        state.last_user_message = None  # Wyczyść po interpretacji
        state.awaiting_input_for = None # Wyczyść po interpretacji    
    
    # focus on currently_modeled_object
    dimension = next(
        (dim for dim in state.dimensions_to_create if dim.dimension_name == state.currently_modeled_object), 
        None
    )
    # early-exit- chceck if currently modeled dimension has complited on columns level
    if ( dimension.scripts_approved ):
        state.awaiting_input_for = None
        state.currently_modeled_object = None
        return state

    dimension_state_txt = dimension_as_string(state, state.currently_modeled_object) 
    human_msg = HumanMessage(content=(
        f"Modeluj wymiar: {state.currently_modeled_object}\n"
        f"Na podstawie dialogu: {state.last_5_messages}\n"
        f"Aktualny stan wymiaru: \n{dimension_state_txt}\n"
    ))
    system_msg = SystemMessage(
        content=conf.Model_Dimension_Scripts_Creation_Text
    )    
    reply = conf.llm_Model_Dimension_Scripts_Creation.invoke([system_msg, human_msg]).content.strip()
    state.awaiting_input_for = "model_dimension_scripts_creation"
    state.last_5_messages.append(f"🤖 {reply}")
    state.last_5_messages = state.last_5_messages[-5:]  # Trzymaj tylko 5 ostatnich
    return interrupt({"message": sanitize_message_for_ui(reply), "next_state": state})

def node_model_fact_entity_level(state: st.DesignState, config) -> st.DesignState | Interrupt:
    session_id = config["configurable"]["session_id"]
    thread_id = config["configurable"]["thread_id"]
    memory_key = f"{session_id}:{thread_id}"
    # Historia rozmowy
    if memory_key not in memory_store:
        memory_store[memory_key] = ChatMessageHistory()
    history = memory_store[memory_key]
    # Dodaj wiadomość użytkownika do historii
    if state.last_user_message:
        history.add_user_message(state.last_user_message)
        state.last_5_messages.append(f"🧑 {state.last_user_message}")
        state.last_5_messages = state.last_5_messages[-5:]  # Trzymaj tylko 5 ostatnich
        model_fact_entity_level_update_state(state)
        state.last_user_message = None  # Wyczyść po interpretacji
        state.awaiting_input_for = None # Wyczyść po interpretacji    

    if not state.currently_modeled_object:
        # You do not have any currently_modeled_object, take first new
        new_fact_candidate = next(
            (fact for fact in state.facts_to_create
            if fact.fact_name
            and (not state.currently_modeled_object
                or fact.fact_name.lower() != state.currently_modeled_object.lower())
            and _all_fact_elements_unapproved(fact)),
            None
        )
        if new_fact_candidate:
            state.currently_modeled_object = new_fact_candidate.fact_name   
        else:
            return interrupt({
                "message": "?!?",
                "next_state": state
            })

    #Find currently modeled fact
    fact = next(
        (f for f in state.facts_to_create if f.fact_name == state.currently_modeled_object), 
        None
    )
    # early-exit chceck if currently modeled fact has complited on entity level
    if ( fact.fact_design_on_entity_level_approved ):
        # chcek if you have any other fact to create
            #else:
            #no fact to create
        state.awaiting_input_for = None
        return state

    fact_state_txt = fact_as_string(state, state.currently_modeled_object) 
    human_msg = HumanMessage(content=(
        f"Modeluj fact: {state.currently_modeled_object}\n"
        f"Na podstawie dialogu: {state.last_5_messages}\n"
        f"Aktualny stan factu: \n{fact_state_txt}\n"
    ))
    system_msg = SystemMessage(
        content=conf.Model_Fact_Entity_Level_Text
    )    
    reply = conf.llm_Model_Fact_Entity_Level.invoke([system_msg, human_msg]).content.strip()
    state.awaiting_input_for = "model_fact_entity_level"
    state.last_5_messages.append(f"🤖 {reply}")
    state.last_5_messages = state.last_5_messages[-5:]  # Trzymaj tylko 5 ostatnich
    return interrupt({"message": sanitize_message_for_ui(reply), "next_state": state}) 

def node_model_fact_columns_level(state: st.DesignState, config) -> st.DesignState | Interrupt:
    session_id = config["configurable"]["session_id"]
    thread_id = config["configurable"]["thread_id"]
    memory_key = f"{session_id}:{thread_id}"
    # Historia rozmowy
    if memory_key not in memory_store:
        memory_store[memory_key] = ChatMessageHistory()
    history = memory_store[memory_key]
    # Dodaj wiadomość użytkownika do historii
    if state.last_user_message:
        history.add_user_message(state.last_user_message)
        state.last_5_messages.append(f"🧑 {state.last_user_message}")
        state.last_5_messages = state.last_5_messages[-5:]  # Trzymaj tylko 5 ostatnich
        model_fact_columns_level_update_state(state)
        state.last_user_message = None  # Wyczyść po interpretacji
        state.awaiting_input_for = None # Wyczyść po interpretacji
    # focus on currently_modeled_object
    fact = next(
        (f for f in state.facts_to_create if f.fact_name == state.currently_modeled_object), 
        None
    )
    # early-exit - chceck if currently modeled fact has complited on columns level
    if ( fact.fact_design_on_columns_level_approved ):
        state.awaiting_input_for = None
        return state

    # Sprawdzamy jego wszystkie tabele źródłowe
    source_tables = (fact.fact_main_source_table or []) + (fact.fact_other_source_tables or [])    
    # Build critical_columns_analyze for this fact if does not exist
    if not (fact.fact_critical_columns_analyze_txt and fact.fact_critical_columns_analyze_txt.strip()):
        # Zbuduj mapę: lower_key -> oryginalny_klucz
        key_map = {str(k).strip().lower(): k for k in state.source_table_analyze.keys()}
        parts = []
        for table in source_tables:
            t_norm = str(table).strip().lower()
            if t_norm in key_map:
                real_key = key_map[t_norm]                     # faktyczny klucz w dict
                val = state.source_table_analyze.get(real_key) # tekst analizy
                if val:                                        # pomiń puste
                    parts.append(f"### {table} table analyze: \n{val}")
        fact.fact_source_tables_analyze_txt = "\n\n".join(parts)
        fact.fact_critical_columns_analyze_txt = point_critical_columns_in_source(
            fact.fact_name,
            source_tables,
            fact.fact_source_tables_analyze_txt
        )
        
    fact_state_txt = fact_as_string(state, state.currently_modeled_object) 
    human_msg = HumanMessage(content=(
        f"Modeluj fakt: {state.currently_modeled_object}\n"
        f"Na podstawie dialogu: {state.last_5_messages}\n"
        f"Aktualny stan faktu: \n{fact_state_txt}\n"
    ))
    system_msg = SystemMessage(
        content=conf.Model_Fact_Columns_Level_Text
    )    
    reply = conf.llm_Model_Fact_Columns_Level.invoke([system_msg, human_msg]).content.strip()
    state.awaiting_input_for = "model_fact_columns_level"
    state.last_5_messages.append(f"🤖 {reply}")
    state.last_5_messages = state.last_5_messages[-5:]  # Trzymaj tylko 5 ostatnich
    return interrupt({"message": sanitize_message_for_ui(reply), "next_state": state})    

def node_model_fact_create_modeling_feed(state: st.DesignState, config) -> st.DesignState | Interrupt:
    session_id = config["configurable"]["session_id"]
    thread_id = config["configurable"]["thread_id"]
    memory_key = f"{session_id}:{thread_id}"
    if memory_key not in memory_store:
        memory_store[memory_key] = ChatMessageHistory()
    history = memory_store[memory_key]

    if state.last_user_message:
        history.add_user_message(state.last_user_message)
        state.last_5_messages.append(f"🧑 {state.last_user_message}")
        state.last_5_messages = state.last_5_messages[-5:]
        model_fact_create_modeling_feed_update_state(state)
        state.last_user_message = None
        state.awaiting_input_for = None
    # focus on currently_modeled_object
    fact = next(
        (fact for fact in state.facts_to_create if fact.fact_name == state.currently_modeled_object), 
        None
    )
    # early-exit - chceck if currently modeled fact_ has complited on feed to modeling tool
    if ( fact.fact_feed_2_modeling_tool_approved ):
        state.awaiting_input_for = None
        return state
    fact_state_txt = fact_as_string(state, state.currently_modeled_object)
    human_msg = HumanMessage(content=(
        f"Przygotuj feed do narzędzia modelowania dla: {state.currently_modeled_object}\n"
        f"Na podstawie dialogu: {state.last_5_messages}\n"
        f"Aktualny stan wymiaru:\n{fact_state_txt}\n"
    )) 
    system_msg = SystemMessage(content=conf.Model_Fact_Create_Modeling_Feed_Text)
    reply = conf.llm_Model_Fact_Create_Modeling_Feed.invoke([system_msg, human_msg]).content.strip()
    state.awaiting_input_for = "model_fact_create_modeling_feed"
    state.last_5_messages.append(f"🤖 {reply}")
    state.last_5_messages = state.last_5_messages[-5:]  # Trzymaj tylko 5 ostatnich
    return interrupt({"message": sanitize_message_for_ui(reply), "next_state": state})

def node_model_fact_scripts_creation(state: st.DesignState, config) -> st.DesignState | Interrupt:
    session_id = config["configurable"]["session_id"]
    thread_id = config["configurable"]["thread_id"]
    memory_key = f"{session_id}:{thread_id}"
    # Historia rozmowy
    if memory_key not in memory_store:
        memory_store[memory_key] = ChatMessageHistory()
    history = memory_store[memory_key]
    # Dodaj wiadomość użytkownika do historii
    if state.last_user_message:
        history.add_user_message(state.last_user_message)
        state.last_5_messages.append(f"🧑 {state.last_user_message}")
        state.last_5_messages = state.last_5_messages[-5:]  # Trzymaj tylko 5 ostatnich
        model_fact_scripts_creation_update_state(state)
        state.last_user_message = None  # Wyczyść po interpretacji
        state.awaiting_input_for = None # Wyczyść po interpretacji    
    
    # focus on currently_modeled_object
    fact = next(
        (f for f in state.facts_to_create if f.fact_name == state.currently_modeled_object), 
        None
    )
    # early-exit- chceck if currently modeled fact has complited on columns level
    if ( fact.fact_scripts_approved ):
        state.awaiting_input_for = None
        state.currently_modeled_object = None
        return state

    fact_state_txt = fact_as_string(state, state.currently_modeled_object) 
    human_msg = HumanMessage(content=(
        f"Modeluj fact: {state.currently_modeled_object}\n"
        f"Na podstawie dialogu: {state.last_5_messages}\n"
        f"Aktualny stan factu: \n{fact_state_txt}\n"
    ))
    system_msg = SystemMessage(
        content=conf.Model_Fact_Scripts_Creation_Text
    )    
    reply = conf.llm_Model_Fact_Scripts_Creation.invoke([system_msg, human_msg]).content.strip()
    state.awaiting_input_for = "model_fact_scripts_creation"
    state.last_5_messages.append(f"🤖 {reply}")
    state.last_5_messages = state.last_5_messages[-5:]  # Trzymaj tylko 5 ostatnich
    return interrupt({"message": sanitize_message_for_ui(reply), "next_state": state})

# ---------------------------------------------------------------------------
# 5. Graph
# ---------------------------------------------------------------------------
checkpointer = InMemorySaver()
graph = StateGraph(state_schema=st.DesignState)

graph.add_node("router", router_node)  # <- NODE
graph.add_node("requirements_collection", node_requirements_collection)
graph.add_node("requirements_analysis", node_requirements_analysis)
graph.add_node("model_dimension_entity_level", node_model_dimension_entity_level)
graph.add_node("model_dimension_columns_level", node_model_dimension_columns_level)
graph.add_node("model_dimension_create_modeling_feed", node_model_dimension_create_modeling_feed)
graph.add_node("model_dimension_scripts_creation", node_model_dimension_scripts_creation)
graph.add_node("model_fact_entity_level", node_model_fact_entity_level)
graph.add_node("model_fact_columns_level", node_model_fact_columns_level)
graph.add_node("model_fact_create_modeling_feed", node_model_fact_create_modeling_feed)
graph.add_node("model_fact_scripts_creation", node_model_fact_scripts_creation)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    router_decider,  # <- DECIDER
    {
        "requirements_collection": "requirements_collection",
        "requirements_analysis": "requirements_analysis",
        "model_dimension_entity_level": "model_dimension_entity_level",
        "model_dimension_columns_level": "model_dimension_columns_level",
        "model_dimension_create_modeling_feed": "model_dimension_create_modeling_feed",
        "model_dimension_scripts_creation": "model_dimension_scripts_creation",
        "model_fact_entity_level": "model_fact_entity_level",
        "model_fact_columns_level": "model_fact_columns_level",
        "model_fact_create_modeling_feed": "model_fact_create_modeling_feed",
        "model_fact_scripts_creation": "model_fact_scripts_creation",
        END: END,
    },
)

graph.add_edge("requirements_collection", "router")
graph.add_edge("requirements_analysis", "router")
graph.add_edge("model_dimension_entity_level", "router")
graph.add_edge("model_dimension_columns_level", "router")
graph.add_edge("model_dimension_create_modeling_feed", "router")
graph.add_edge("model_dimension_scripts_creation", "router")
graph.add_edge("model_fact_entity_level", "router")
graph.add_edge("model_fact_columns_level", "router")
graph.add_edge("model_fact_create_modeling_feed", "router")
graph.add_edge("model_fact_scripts_creation", "router")

runnable = graph.compile(checkpointer=InMemorySaver())

# ---------------------------------------------------------------------------
# 4. Initial
# ---------------------------------------------------------------------------

def run_cli():
    rprint("[bold green] Let's get to work! (CLI) [/bold green]")
    state = st.DesignState()
    load_list_available_dimensions_to_state(state)
    load_SAP_2_Snowflake_data_types_mapping_to_state(state)

    while True:
        result = runnable.invoke(
            state,
            config={"configurable": {"session_id": session_id, "thread_id": thread_id}}
        )
        if "__interrupt__" in result:
            interrupt_obj = result["__interrupt__"][0]
            msg = interrupt_obj.value["message"]
            state = interrupt_obj.value["next_state"]
            rprint(f"\n🤖 [cyan]LLM:[/cyan] {msg}")
            last_user_message = input("🧑 Ty: ")
            state.last_user_message = last_user_message
        else:
            state = result
            rprint(state)
            break


if __name__ == "__main__":
    run_cli()

