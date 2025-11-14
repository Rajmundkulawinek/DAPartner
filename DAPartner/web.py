import os
import json
import re
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as strl

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DAPartner.State import state as st
from DAPartner.app import (
    runnable,
    load_list_available_dimensions_to_state,
    load_SAP_2_Snowflake_data_types_mapping_to_state,
    set_notify_hook, 
)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
strl.set_page_config(page_title="DAPartner", page_icon="💬", layout="centered")

# Wymagamy klucza do LLM z ENV (zgodnie z Twoim kodem .env → load_dotenv())
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
    strl.warning("Brakuje OPENAI_API_KEY w środowisku (.env). Ustaw i odpal ponownie.")

strl.title("💬 DAPartner – a data architect’s best friend ")

# ------------------------------------------------------------
# Session initialization
# ------------------------------------------------------------
if "initialized" not in strl.session_state:
    # Pierwsze uruchomienie: zbuduj stan i doładuj globalne pliki
    strl.session_state.state = st.DesignState()
    strl.session_state.state = load_list_available_dimensions_to_state(strl.session_state.state)
    strl.session_state.state = load_SAP_2_Snowflake_data_types_mapping_to_state(strl.session_state.state)

    # Metadane sesji do configu invoke (z Twojego kodu)
    strl.session_state.session_id = "streamlit-user"
    strl.session_state.thread_id = "ui-thread"

    # Historia bąbelków: list[dict(role, content)]
    strl.session_state.chat = []
    strl.session_state.initialized = True

if "notifier_registered" not in strl.session_state:
    placeholder = strl.empty()

    def streamlit_notify(msg: str):
        # spróbuj toast; jeśli brak w wersji, pokaż info-box
        try:
            strl.toast(msg)
        except Exception:
            placeholder.info(msg)

    set_notify_hook(streamlit_notify)
    strl.session_state.notifier_registered = True

# ------------------------------------------------------------
# Helpers – ekstrakcja WIELU plików z jednej wiadomości
# ------------------------------------------------------------
# Wspierane znaczniki (case-insensitive):
#  1) **SQL_SCRIPT_START**...**SQL_SCRIPT_END**           → .sql
#  2) **CSV_START**...**CSV_END**                         → .csv
#  3) **FILE_START: <nazwa z rozszerzeniem>**...**FILE_END**  → dowolny typ (np. .ddl, .sql, .csv, .yaml)
#     Przykład: **FILE_START: dim_customer.ddl** ... **FILE_END**

_GENERIC_FILE_RE = re.compile(r"\*\*FILE_START:\s*([^*\n]+?)\*\*(.*?)\*\*FILE_END\*\*", re.IGNORECASE | re.DOTALL)
#_GENERIC_FILE_RE = re.compile(r"\*\*FILE_START:\s*([^*\n]+?)\*\*(.*?)\*\*FILE_END\*\*", re.IGNORECASE | re.DOTALL)
#_SQL_FILE_RE     = re.compile(r"\*\*SQL_SCRIPT_START\*\*(.*?)\*\*SQL_SCRIPT_END\*\*", re.IGNORECASE | re.DOTALL)
#_CSV_FILE_RE     = re.compile(r"\*\*CSV_START\*\*(.*?)\*\*CSV_END\*\*", re.IGNORECASE | re.DOTALL)
#_DDL_FILE_RE     = re.compile(r"\*\*DDL_SCRIPT_START\*\*(.*?)\*\*DDL_SCRIPT_(?:END|STOP)\*\*", re.IGNORECASE | re.DOTALL)

def _extract_files_and_clean_text(text: str):
    """Zwraca (cleaned_text, files), gdzie files to lista słowników:
       { 'type': 'sql'|'csv'|'file', 'ext': 'sql'|'csv'|inny, 'name': str|None, 'content': str }
       Obsługuje WIELE bloków w jednej wiadomości, zachowuje kolejność.
    """
    if not text:
        return "", []

    matches = []  # każdy: dict(start, end, name, type, ext, content)

    # Znajdź wszystkie dopasowania do uniwersalnego formatu **FILE_START:...**
    for m in _GENERIC_FILE_RE.finditer(text):
        raw_name = m.group(1).strip()
        body = m.group(2).strip()
        ext = Path(raw_name).suffix.lower().lstrip(".") or "txt"
        
        # Określ typ na podstawie rozszerzenia dla kolorowania i MIME type
        ftype = "file"
        if ext in {"sql", "ddl"}:
            ftype = "sql"
        elif ext == "csv":
            ftype = "csv"
            
        matches.append({
            "start": m.start(),
            "end": m.end(),
            "name": raw_name,
            "type": ftype,
            "ext": ext,
            "content": body,
        })

    # Sortuj po pozycji w tekście, żeby zachować naturalną kolejność
    matches.sort(key=lambda d: d["start"]) 

    # Usuń wszystkie bloki z treści bąbla (od końca, by nie przesuwać indeksów)
    cleaned = text
    for m in sorted(matches, key=lambda d: d["start"], reverse=True):
        cleaned = cleaned[: m["start"]] + cleaned[m["end"] :]

    # Zwróć listę "plików" w ustalonej kolejności
    files = [
        {
            "type": m["type"],
            "ext": m["ext"],
            "name": m["name"],
            "content": m["content"],
        }
        for m in matches
    ]

    # Finalne czyszczenie tekstu
    cleaned = cleaned.strip()
    return cleaned, files


def _pretty_guess_filename(base: str, i: int, ext: str, provided: str | None) -> str:
    """Wymyśla sensowną nazwę: jeśli nadana w znaczniku – użyj, wpp. {base}_{i}.{ext}."""
    if provided:
        return provided
    base = (base or "attachment").replace(" ", "_")
    ext = (ext or "txt").lstrip(".")
    return f"{base}_{i}.{ext}"


def run_until_interrupt(user_text_or_none: str | None):
    """
    - Jeśli user_text_or_none jest podany: wrzuca do `state.last_user_message`.
    - Wykonuje kroki grafu aż:
        a) pojawi się `__interrupt__` (wtedy dodajemy bąbelek asystenta i STOP),
        b) albo graf skończy (END) – wtedy dorzucamy finalny stan/info.
    - Zwraca (status, reply_text_or_None)
        status: "need_input" | "done"
    """
    state = strl.session_state.state

    if user_text_or_none:
        state.last_user_message = user_text_or_none

    # Pętla bezpieczeństwa (żeby nie zapętlić UI)
    for _ in range(20):
        result = runnable.invoke(
            state,
            config={
                "configurable": {
                    "session_id": strl.session_state.session_id,
                    "thread_id": strl.session_state.thread_id,
                }
            },
        )

        if "__interrupt__" in result:
            interrupt_obj = result["__interrupt__"][0].value
            msg = interrupt_obj["message"]
            strl.session_state.state = interrupt_obj["next_state"]
            return "need_input", msg

        # brak interruptu → graf przesunął stan do przodu
        strl.session_state.state = result

        # Czy to koniec?
        if not getattr(strl.session_state.state, "awaiting_input_for", None):
            return "done", None

    return "need_input", "Coś się przyblokowało – przerwano pętlę bezpieczeństwa."


# AUTO-START: pierwszy krok grafu, żeby dostać pierwszą wiadomość bota
if not strl.session_state.chat:
    status, reply = run_until_interrupt(None)  # brak wiadomości od usera
    if status == "need_input" and reply:
        strl.session_state.chat.append({"role": "assistant", "content": reply})
        strl.rerun()  # odśwież UI, żeby bąbelek się pojawił

# ------------------------------------------------------------
# 4) Wgrywanie analiz źródeł (CSV) -> zapis do state.source_table_analyze
# ------------------------------------------------------------
with strl.expander("📎 Przydatne analizy źródeł (CSV)"):
    strl.markdown(
        "Wrzuć jeden lub więcej plików CSV z analizą kolumn. "
        "Domyślnie nazwa pliku (bez rozszerzenia) stanie się nazwą źródła. "
        "Jeśli nazwa nie pasuje do tabel używanych w modelu, możesz ją zmienić poniżej."
    )

    uploaded_files = strl.file_uploader(
        "Pliki CSV",
        type=["csv", "txt"],
        accept_multiple_files=True,
        key="csv_uploader",
    )

    # Przycisk zapisu
    save_clicked = strl.button("💾 Zapisz wgrane pliki do stanu", type="primary", disabled=not uploaded_files)

    if save_clicked and uploaded_files:
        if strl.session_state.state.source_table_analyze is None:
            strl.session_state.state.source_table_analyze = {}

        saved = []
        for uf in uploaded_files:
            source_name = Path(uf.name).stem  # np. BKPF z BKPF.csv
            # pobierz bytes z pamięci, zdekoduj jako tekst
            raw = uf.getvalue()
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw.decode("latin-1", errors="ignore")
            # normalizacja końców linii
            text = text.replace("\r\n", "\n").replace("\r", "\n")

            # zapis do stanu: { "NAZWA_TABELI": "surowy_csv" }
            strl.session_state.state.source_table_analyze[source_name] = text
            saved.append(source_name)

        strl.success(f"Zapisano: {', '.join(saved)}")

    # Podgląd i szybka edycja nazw
    if strl.session_state.state.source_table_analyze:
        strl.markdown("**Wgrane analizy:**")

        # rename pojedynczego wpisu
        with strl.form("rename_source_key"):
            keys = list(strl.session_state.state.source_table_analyze.keys())
            selected = strl.selectbox("Zmień nazwę źródła", options=keys)
            new_name = strl.text_input("Nowa nazwa źródła", value=selected)
            rename_ok = strl.form_submit_button("Zmień nazwę")
            if rename_ok and new_name and new_name != selected:
                # przenieś treść pod nowy klucz
                strl.session_state.state.source_table_analyze[new_name] = (
                    strl.session_state.state.source_table_analyze.pop(selected)
                )
                strl.success(f"Zmieniono nazwę: {selected} → {new_name}")

        # listowanie z podglądem
        for name, txt in strl.session_state.state.source_table_analyze.items():
            with strl.expander(f"🔎 {name} — podgląd (pierwsze wiersze)"):
                # próbuj pokazać 8 wierszy CSV jako tabelę
                try:
                    df = pd.read_csv(StringIO(txt))
                    strl.dataframe(df.head(8))
                except Exception:
                    strl.code("\n".join(txt.splitlines()[:12]))

        # czyszczenie całości
        if strl.button("🗑️ Wyczyść wszystkie analizy"):
            strl.session_state.state.source_table_analyze = {}
            strl.info("Wyczyszczono wszystkie wgrane analizy.")


# ------------------------------------------------------------
# 5) Render historii – teraz obsługa WIELU plików na bąbelek
# ------------------------------------------------------------
for idx, m in enumerate(strl.session_state.chat):
    with strl.chat_message(m["role"]):
        if m["role"] == "assistant":
            body, files = _extract_files_and_clean_text(m["content"])  # <— NOWE
            if body:
                strl.markdown(body)

            # Sugerowana baza nazwy (np. nazwa wymiaru)
            dim_name = getattr(strl.session_state.state, "currently_modeled_object", None) or "dimension"

            # Wyświetl WSZYSTKIE załączone pliki w oryginalnej kolejności
            for i, f in enumerate(files, start=1):
                label = "SQL script" if f["type"] == "sql" else ("CSV" if f["type"] == "csv" else "Plik")
                file_name = _pretty_guess_filename(dim_name, i, f["ext"], f["name"])

                strl.subheader(f"{label} — {file_name}")

                # Edytowalny obszar treści
                edited = strl.text_area(
                    "Edytuj zawartość (zostanie pobrane przyciskiem poniżej)",
                    value=f["content"],
                    height=240,
                    key=f"file_area_{idx}_{i}",
                )

                # Pobierz
                strl.download_button(
                    "💾 Pobierz plik",
                    data=edited,
                    file_name=file_name,
                    mime = "text/sql" if f["type"] == "sql" else ("text/csv" if f["type"] == "csv" else "text/plain"),
                    key=f"dl_file_{idx}_{i}",
                )

                # Dodatkowy podgląd CSV (jeśli możliwy)
                if f["ext"] == "csv":
                    with strl.expander("Podgląd CSV (pierwsze wiersze)"):
                        try:
                            df = pd.read_csv(StringIO(edited))
                            strl.dataframe(df.head(8))
                        except Exception as e:
                            strl.info("Nie udało się sparsować CSV – pokazuję fragment surowy.")
                            strl.code("\n".join(edited.splitlines()[:12]))

        else:
            strl.markdown(m["content"])

# ------------------------------------------------------------
# 6) Input użytkownika + wykonanie kroku
# ------------------------------------------------------------
prompt = strl.chat_input("Napisz wiadomość…")
if prompt:
    # 1) pokaż bąbelek usera
    strl.session_state.chat.append({"role": "user", "content": prompt})
    with strl.chat_message("user"):
        strl.markdown(prompt)

    # 2) uruchom graf aż poprosi o kolejne dane (Interrupt) lub skończy
    status, reply = run_until_interrupt(prompt)

    if status == "need_input" and reply:
        strl.session_state.chat.append({"role": "assistant", "content": reply})
        with strl.chat_message("assistant"):
            body, files = _extract_files_and_clean_text(reply)
            if body:
                strl.markdown(body)

            dim_name = getattr(strl.session_state.state, "currently_modeled_object", None) or "dimension"

            for i, f in enumerate(files, start=1):
                label = "SQL script" if f["type"] == "sql" else ("CSV" if f["type"] == "csv" else "Plik")
                file_name = _pretty_guess_filename(dim_name, i, f["ext"], f["name"])

                strl.subheader(f"{label} — {file_name}")
                edited = strl.text_area(
                    "Edytuj zawartość (zostanie pobrane przyciskiem poniżej)",
                    value=f["content"],
                    height=240,
                    key=f"file_area_live_{i}",
                )
                strl.download_button(
                    "💾 Pobierz plik",
                    data=edited,
                    file_name=file_name,
                    mime = "text/sql" if f["type"] == "sql" else ("text/csv" if f["type"] == "csv" else "text/plain"),
                    key=f"dl_file_live_{i}",
                )

                if f["ext"] == "csv":
                    with strl.expander("Podgląd CSV (pierwsze wiersze)"):
                        try:
                            df = pd.read_csv(StringIO(edited))
                            strl.dataframe(df.head(8))
                        except Exception:
                            strl.info("Nie udało się sparsować CSV – pokazuję fragment surowy.")
                            strl.code("\n".join(edited.splitlines()[:12]))

    elif status == "done":
        # Opcjonalnie: pokaż podsumowanie finalne / podgląd stanu
        with strl.chat_message("assistant"):
            strl.markdown("✅ **Zakończono aktualny etap.** Możesz kontynuować rozmowę lub podejrzeć stan poniżej.")

# ------------------------------------------------------------
# 7) Panel diagnostyczny (rozwiń jeśli chcesz)
# ------------------------------------------------------------
with strl.expander("🔎 Podgląd stanu (diag)"):
    # Pydantic v2: model_dump; jeśli masz v1 – można użyć .dict()
    try:
        state_json = strl.session_state.state.model_dump()
    except Exception:
        state_json = strl.session_state.state.dict()
    strl.json(state_json)

