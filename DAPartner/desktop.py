# desktop.py

import os, io, re, sys, platform, json

from pathlib import Path

from typing import Optional, Callable



from dotenv import load_dotenv

import flet as ft

import pandas as pd


# --- Icon compatibility between Flet versions ---
import flet as ft

Icons = getattr(ft, "Icons", None)
if Icons is None:
    # awaryjny fallback – nazwy stringowe też działają
    class Icons:
        SEND = "send"
        SAVE = "save"
        VPN_KEY = "vpn_key"
        UPLOAD_FILE = "upload_file"
        TABLE_VIEW = "table_view"


# === BACKEND: like in your project ===

from DAPartner.State import state as st

from DAPartner.app import (

    runnable,

    load_list_available_dimensions_to_state,

    load_SAP_2_Snowflake_data_types_mapping_to_state,

    set_notify_hook,

)



# Read the key from .env just like in the backend

load_dotenv()  # requires OPENAI_API_KEY



# Minimal attachment parser in assistant replies: **FILE_START: name** ... **FILE_END**

_GENERIC_FILE_RE = re.compile(

    r"\*\*FILE_START:\s*([^*\n]+?)\*\*(.*?)\*\*FILE_END\*\*",

    re.IGNORECASE | re.DOTALL,

)



APP_NAME = "DAPartner"

# Set default USER_AGENT to silence warnings

os.environ.setdefault("USER_AGENT", f"{APP_NAME}/desktop")



def _repo_root() -> Path:

    # .../DAPartner_repo  (this file is in .../DAPartner_repo/DAPartner/desktop.py)

    return Path(__file__).resolve().parents[1]



def _runtime_dir() -> Path:

    # Directory with .exe when bundled, otherwise repo directory (for clarity)

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):

        return Path(sys.executable).resolve().parent

    return _repo_root()



# ---- Ikons ----

ASSETS_DIR = _runtime_dir() / "DAPartner" / "Assets"

WINDOW_ICON = (

    ASSETS_DIR / "dapartner.ico" if platform.system() == "Windows"

    else ASSETS_DIR / "dapartner.png"

)



def _fallback_env_dir() -> Path:

    # User's home profile directory (when runtime_dir is not writable)

    if platform.system() == "Windows":

        return Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming")) / APP_NAME

    if platform.system() == "Darwin":

        return Path.home() / "Library" / "Application Support" / APP_NAME

    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_NAME



def _primary_env_path() -> Path:

    return _runtime_dir() / ".env"



def _fallback_env_path() -> Path:

    return _fallback_env_dir() / ".env"



def _ensure_parent(p: Path):

    p.parent.mkdir(parents=True, exist_ok=True)



def _read_api_key_from_env_files() -> str | None:

    # 1) ENV var

    k = os.getenv("OPENAI_API_KEY")

    if k:

        return k

    # 2) .env next to the runtime (repo root in dev / next to .exe in build)

    env1 = _primary_env_path()

    if env1.exists():

        load_dotenv(dotenv_path=env1, override=False)

        k = os.getenv("OPENAI_API_KEY")

        if k:

            return k

    # 3) fallback in user's profile

    env2 = _fallback_env_path()

    if env2.exists():

        load_dotenv(dotenv_path=env2, override=False)

        k = os.getenv("OPENAI_API_KEY")

        if k:

            return k

    return None



def _write_api_key_to_env_file(key: str) -> Path:

    """

    Write/update OPENAI_API_KEY in .env:

    - prefer .env in runtime_dir (repo in dev / next to .exe in build),

    - if it fails, write to fallback (~/.DAPartner/.env).

    Returns the path where the key was written.

    """

    def _upsert(path: Path, key: str):

        _ensure_parent(path)

        lines = []

        if path.exists():

            lines = path.read_text(encoding="utf-8").splitlines()

            lines = [ln for ln in lines if not ln.strip().startswith("OPENAI_API_KEY=")]

        lines.append(f"OPENAI_API_KEY={key}")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")



    target = _primary_env_path()

    try:

        _upsert(target, key)

    except Exception:

        target = _fallback_env_path()

        _upsert(target, key)



    # set for current session immediately

    os.environ["OPENAI_API_KEY"] = key

    # and reload so libraries using dotenv see it as well (override=True for certainty)

    load_dotenv(dotenv_path=target, override=True)

    return target





def extract_files_and_clean_text(text: str):

    if not text:

        return "", []

    files = []

    cleaned = text

    for m in sorted(_GENERIC_FILE_RE.finditer(text), key=lambda x: x.start(), reverse=True):

        name = m.group(1).strip()

        body = m.group(2).strip()

        files.append({"name": name, "content": body})

        cleaned = cleaned[:m.start()] + cleaned[m.end():]

    files.reverse()

    return cleaned.strip(), files





def main(page: ft.Page):



    page.title = "DAPartner"

    page.window_width = 1100

    page.window_height = 800

    icon_path = str(WINDOW_ICON if WINDOW_ICON.exists() else (ASSETS_DIR / "dapartner.ico"))





    page.window_icon = icon_path

    page.horizontal_alignment = "stretch"

    page.vertical_alignment = "stretch"



    # ====== API KEY DIALOG ======

    api_key_field = ft.TextField(

        label="OpenAI API key",

        hint_text="sk-...",

        password=True,

        can_reveal_password=True,

        width=420

    )

    api_msg = ft.Text("", size=12)

    dlg: Optional[ft.AlertDialog] = None



    def _mask_key(k: str) -> str:

        if not k:

            return ""

        return (k[:7] + "…" + k[-4:]) if len(k) >= 12 else "•••"



    _current_key = _read_api_key_from_env_files()

    key_status = ft.Text(

        ("OpenAI key status: set" if _current_key else "OpenAI key status: not set"),

        size=12,

        italic=True,

        color=("#2e7d32" if _current_key else "#c62828"),

    )



    def show_api_key_dialog(on_success: Optional[Callable[[], None]] = None):

        nonlocal dlg  # reuse and overwrite dlg from the outer scope



        def _save(_):

            k = (api_key_field.value or "").strip()

            if not k:

                api_msg.value = "Enter the key."

                page.update()

                return



            path = _write_api_key_to_env_file(k)



            # update status

            key_status.value = "OpenAI key status: set"

            key_status.color = "#2e7d32"

            key_status.update()



            api_msg.value = f"Saved to: {path}"

            # close dialog (Flet 0.28+)

            try:

                page.close(dlg)

            except Exception:

                # fallback for older Flet versions

                dlg.open = False

                page.update()



            if on_success:

                on_success()



        # prepare dialog controls

        api_key_field.value = ""   # clear on each open

        api_msg.value = ""



        dlg = ft.AlertDialog(

            modal=True,

            title=ft.Text("Enter your OpenAI key"),

            content=ft.Column([api_key_field, api_msg], tight=True),

            actions=[ft.ElevatedButton("Save", on_click=_save)],

            actions_alignment=ft.MainAxisAlignment.END,

        )



        # open dialog (Flet 0.28+)

        try:

            page.open(dlg)

        except Exception:

            # fallback for older Flet versions

            page.dialog = dlg

            dlg.open = True

            page.update()



    # ------ App state (instead of streamlit.session_state) ------

    app_state = st.DesignState()

    app_state = load_list_available_dimensions_to_state(app_state)

    app_state = load_SAP_2_Snowflake_data_types_mapping_to_state(app_state)



    session_id = "desktop-user"

    thread_id = "ui-thread"



    # Hook for backend notifications

    def notify(msg: str):

        page.snack_bar = ft.SnackBar(ft.Text(msg))

        page.snack_bar.open = True

        page.update()

    set_notify_hook(notify)



    # ------ UI: Chat + helpers ------

    chat = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)

    input_field = ft.TextField(

        hint_text="Type a message…",

        multiline=True,

        min_lines=1,

        max_lines=4,

        expand=True

    )

    send_btn = ft.ElevatedButton("Send", icon=Icons.SEND)

    busy_ring = ft.ProgressRing(visible=False, width=22, height=22, tooltip="Working…")



    # File pickers for CSV (local files)

    open_picker = ft.FilePicker(on_result=lambda e: on_pick_files(e))

    open_dimensions_picker = ft.FilePicker(on_result=lambda e: on_pick_dimensions(e))

    save_picker = ft.FilePicker(on_result=lambda e: on_save_file(e))

    page.overlay.extend([open_picker, open_dimensions_picker, save_picker])



    # Temporary buffer: which attachment to save after choosing a path

    pending_save = {"filename": None, "content": None}



    # Side panel: manage source analyses (CSV -> state.source_table_analyze)

    uploaded_list = ft.Column()



    def on_pick_dimensions(e: ft.FilePickerResultEvent):

        if not e.files:

            return



        parts = []

        files_ok, files_err = [], []

        for f in e.files:

            p = Path(f.path)

            try:

                raw = p.read_bytes()

                try:

                    text = raw.decode("utf-8-sig")

                except UnicodeDecodeError:

                    text = raw.decode("latin-1", errors="ignore")

                parts.append(f"\n=== {p.name} ===\n{text.strip()}\n")

                files_ok.append(p.name)

            except Exception as ex:

                files_err.append(f"{p.name}: {ex}")



        if not parts:

            chat.controls.append(ft.Text("?? No data loaded from CSV.", color="red"))

            if files_err:

                chat.controls.append(ft.Text("Errors:\n" + "\n".join(files_err)))

            page.update()

            return



        combined_text = "\n".join(parts).strip()



        # Keep this as TEXT in available_dimensions or — if Pydantic rejects — next to it in source_table_analyze

        try:

            app_state.available_dimensions = combined_text

        except Exception:

            if app_state.source_table_analyze is None:

                app_state.source_table_analyze = {}

            app_state.source_table_analyze["__available_dimensions_text__"] = combined_text



        chat.controls.append(ft.Text(f"? Loaded dimension info from: {', '.join(files_ok)}"))

        if files_err:

            chat.controls.append(ft.Text("?? Some files were skipped:\n" + "\n".join(files_err)))

        page.update()



    def ask_open_dimensions_csv():

        open_dimensions_picker.pick_files(

            allow_multiple=True,

            allowed_extensions=["csv", "txt"]

        )



    def refresh_uploaded_list():

        uploaded_list.controls.clear()

        if not app_state.source_table_analyze:

            uploaded_list.controls.append(

                ft.Text("No saved analyses.", size=12, italic=True, color="black")

            )

        else:

            for name, txt in app_state.source_table_analyze.items():

                preview = "\n".join(txt.splitlines()[:10])

                exp = ft.ExpansionTile(

                    title=ft.Text(f"?? {name} — preview (first lines)"),

                    controls=[ft.Text(preview or "(empty)")],

                )

                uploaded_list.controls.append(exp)

        uploaded_list.update()



    def on_pick_files(e: ft.FilePickerResultEvent):

        if not e.files:

            return

        if app_state.source_table_analyze is None:

            app_state.source_table_analyze = {}



        saved = []

        for f in e.files:

            p = Path(f.path)

            try:

                raw = p.read_bytes()

                try:

                    text = raw.decode("utf-8-sig")

                except UnicodeDecodeError:

                    text = raw.decode("latin-1", errors="ignore")

                text = text.replace("\r\n", "\n").replace("\r", "\n")

                app_state.source_table_analyze[p.stem] = text

                saved.append(p.stem)

            except Exception as ex:

                chat.controls.append(ft.Text(f"?? Failed to read {p.name}: {ex}", color="red"))

        if saved:

            chat.controls.append(ft.Text(f"?? Saved analyses: {', '.join(saved)}"))

        refresh_uploaded_list()

        page.update()



    def ask_open_csv():

        open_picker.pick_files(

            allow_multiple=True,

            allowed_extensions=["csv", "txt"]

        )



    def ask_clear_all():

        app_state.source_table_analyze = {}

        chat.controls.append(ft.Text("?? Cleared all CSV analyses"))

        refresh_uploaded_list()

        page.update()



    # Saving a file that came from the assistant's response

    def on_save_file(e: ft.FilePickerResultEvent):

        if not e.path or not pending_save["content"]:

            return

        try:

            Path(e.path).write_text(pending_save["content"], encoding="utf-8")

            chat.controls.append(ft.Text(f"? Saved file: {e.path}"))

        except Exception as ex:

            chat.controls.append(ft.Text(f"?? Save error: {ex}", color="red"))

        page.update()



    def save_attachment(default_name: str, content: str):

        pending_save["filename"] = default_name

        pending_save["content"] = content

        save_picker.save_file(file_name=default_name)



    # Loop similar to your Streamlit app: until interrupt / END

    def run_until_interrupt(user_text_or_none: str | None):

        nonlocal app_state

        if user_text_or_none:

            app_state.last_user_message = user_text_or_none



        for _ in range(20):

            result = runnable.invoke(

                app_state,

                config={"configurable": {"session_id": session_id, "thread_id": thread_id}}

            )

            if "__interrupt__" in result:

                interrupt_obj = result["__interrupt__"][0].value

                msg = interrupt_obj["message"]

                app_state = interrupt_obj["next_state"]

                return "need_input", msg

            app_state = result

            if not getattr(app_state, "awaiting_input_for", None):

                return "done", None

        return "need_input", "Something got stuck — safety loop aborted."



    def start_session():

        status, reply = run_until_interrupt(None)

        if status == "need_input" and reply:

            render_assistant_message(reply)

        page.update()



    def render_assistant_message(text: str):

        body, files = extract_files_and_clean_text(text)

        if body:

            chat.controls.append(ft.Text(body))

        # Show attachments and allow saving locally

        for f in files:

            fname = f["name"] or "file.txt"

            preview = "\n".join(f["content"].splitlines()[:14])

            row = ft.Column(

                [

                    ft.Text(f"?? {fname}", weight=ft.FontWeight.BOLD),

                    ft.Text(preview if preview else "(empty)"),

                    ft.Row(

                        [

                            ft.ElevatedButton(

                                "Save as…",

                                icon=Icons.SAVE,

                                on_click=lambda _, n=fname, c=f["content"]: save_attachment(n, c),

                            )

                        ]

                    ),

                ],

                spacing=6,

            )

            chat.controls.append(row)



    def send_message(_):

        text = input_field.value.strip()

        if not text:

            return



        chat.controls.append(ft.Text(f"?? {text}", selectable=True))

        input_field.value = ""



        # === START: busy UI ===

        busy_ring.visible = True

        send_btn.disabled = True

        input_field.disabled = True

        page.update()

        # === END: busy UI ===



        try:

            status, reply = run_until_interrupt(text)

            if status == "need_input" and reply:

                render_assistant_message(reply)

            elif status == "done":

                chat.controls.append(ft.Text("? Current stage completed."))

        except Exception as ex:

            chat.controls.append(ft.Text(f"?? Error: {ex}", color="red"))

        finally:

            # === hide busy UI ===

            busy_ring.visible = False

            send_btn.disabled = False

            input_field.disabled = False

            page.update()



    send_btn.on_click = send_message



    # Auto-start first step

    status, reply = run_until_interrupt(None)

    if status == "need_input" and reply:

        render_assistant_message(reply)



    # ====== LAYOUT (Left panel + Right panel) ======



    # 1) SECTION: Source analyses (with the list directly under the button)

    section_sources = ft.Column(

        [

            ft.ElevatedButton(

                "Source analysis (.csv)",

                icon=Icons.UPLOAD_FILE,

                on_click=lambda _: ask_open_csv(),

            ),

            uploaded_list,  # <= HERE the list/message "No saved analyses."

        ],

        spacing=8,

    )



    # 2) SECTION: Existing dimensions info

    section_dimensions = ft.Column(

        [

            ft.ElevatedButton(

                "Existing dimensions (.csv)",

                icon=Icons.UPLOAD_FILE,

                on_click=lambda _: ask_open_dimensions_csv(),

            ),

        ],

        spacing=8,

    )



    # 3) Left panel: key status + key button + sections separated with dividers

    left_panel = ft.Container(

        ft.Column(

            [

#                key_status,

#                ft.ElevatedButton(

#                    "Set OpenAI key",

#                    icon=ft.Icons.VPN_KEY,

#                    on_click=lambda _: show_api_key_dialog(),

#                ),





            ft.ElevatedButton(

                "Set OpenAI key",

                # (opcjonalnie) ikona klucza:

                icon=Icons.VPN_KEY,

                on_click=lambda _: show_api_key_dialog(),

            ),

            key_status,







                ft.Divider(),          # divider above source analyses section

                section_sources,       # 1) Analyses (list right under the button)

                ft.Divider(),          # divider between sections

                section_dimensions,    # 2) Dimensions

            ],

            expand=True,

        ),

        width=260,

    )



    page.add(

        ft.Row(

            [

                left_panel,

                ft.VerticalDivider(),

                ft.Column([chat, ft.Row([input_field, send_btn, busy_ring])], expand=True),

            ],

            expand=True,

        )

    )



    refresh_uploaded_list()

    # If the key already exists — start, otherwise ask for the key then start

    if _read_api_key_from_env_files():

        start_session()

    else:

        show_api_key_dialog(on_success=start_session)





ft.app(target=main, view=ft.AppView.FLET_APP, assets_dir=str(ASSETS_DIR))



