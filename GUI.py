from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
import sounddevice as sd
import soundfile as sf

import sys

sys.stdout.reconfigure(
    encoding="utf-8"
)

sys.stderr.reconfigure(
    encoding="utf-8"
)


BASE_DIR = Path(__file__).resolve().parent
SONGS_FOLDER = BASE_DIR / "songs"
FINGERPRINTS_FOLDER = BASE_DIR / "fingerprints"
DATABASE_FILE = BASE_DIR / "database.json"
MIC_CONFIG_FILE = BASE_DIR / "mic_config.json"
QUERY_WAV_FILE = BASE_DIR / "query.wav"

CREATE_DATABASE_SCRIPT = BASE_DIR / "create_database.py"
RECORDER_SCRIPT = BASE_DIR / "record_npy.py"
MATCHER_SCRIPT = BASE_DIR / "match_npy.py"


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class SongRecognizerGUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Direct Song Recognition System")
        self.geometry("1000x700")
        self.minsize(880, 620)

        self.busy = False
        self.microphone_map: dict[str, int | None] = {}

        SONGS_FOLDER.mkdir(parents=True, exist_ok=True)
        FINGERPRINTS_FOLDER.mkdir(parents=True, exist_ok=True)

        self._build_interface()
        self.refresh_microphones(show_message=False)
        self.refresh_database_status()

        self.protocol("WM_DELETE_WINDOW", self.close_app)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def _build_interface(self) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Direct Song Recognition System",
            font=ctk.CTkFont(size=28, weight="bold"),
        ).grid(row=0, column=0, padx=24, pady=(18, 4), sticky="w")

        ctk.CTkLabel(
            header,
            text="Build the database, record a song sample, and identify it.",
            text_color="gray75",
        ).grid(row=1, column=0, padx=24, pady=(0, 16), sticky="w")

        controls = ctk.CTkFrame(self, width=300)
        controls.grid(
            row=1,
            column=0,
            padx=(18, 9),
            pady=18,
            sticky="nsw",
        )
        controls.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            controls,
            text="Microphone selection",
            font=ctk.CTkFont(size=17, weight="bold"),
        ).grid(row=0, column=0, padx=15, pady=(18, 8), sticky="w")

        self.microphone_box = ctk.CTkComboBox(
            controls,
            values=["System default microphone"],
            width=270,
            command=self.select_microphone,
        )
        self.microphone_box.grid(row=1, column=0, padx=15, pady=5, sticky="ew")

        self.mic_index_label = ctk.CTkLabel(
            controls,
            text="Selected index: default",
            text_color="gray75",
        )
        self.mic_index_label.grid(row=2, column=0, padx=15, pady=(2, 6), sticky="w")

        self.refresh_mic_button = ctk.CTkButton(
            controls,
            text="Refresh microphone list",
            command=self.refresh_microphones,
        )
        self.refresh_mic_button.grid(row=3, column=0, padx=15, pady=(4, 18), sticky="ew")

        ctk.CTkLabel(
            controls,
            text="System operations",
            font=ctk.CTkFont(size=17, weight="bold"),
        ).grid(row=4, column=0, padx=15, pady=(4, 8), sticky="w")

        self.database_button = ctk.CTkButton(
            controls,
            text="1. Build / refresh database",
            command=self.build_database,
            height=40,
        )
        self.database_button.grid(row=5, column=0, padx=15, pady=6, sticky="ew")

        self.open_songs_button = ctk.CTkButton(
            controls,
            text="Open songs folder",
            command=self.open_songs_folder,
            fg_color="gray35",
            hover_color="gray28",
        )
        self.open_songs_button.grid(row=6, column=0, padx=15, pady=(0, 14), sticky="ew")

        self.record_button = ctk.CTkButton(
            controls,
            text="2. Record 12-second query",
            command=self.record_query,
            height=40,
        )
        self.record_button.grid(row=7, column=0, padx=15, pady=6, sticky="ew")

        self.play_button = ctk.CTkButton(
            controls,
            text="Play recorded query",
            command=self.play_query,
        )
        self.play_button.grid(row=8, column=0, padx=15, pady=6, sticky="ew")

        self.stop_button = ctk.CTkButton(
            controls,
            text="Stop playback",
            command=self.stop_playback,
            fg_color="gray35",
            hover_color="gray28",
        )
        self.stop_button.grid(row=9, column=0, padx=15, pady=(0, 14), sticky="ew")

        self.search_button = ctk.CTkButton(
            controls,
            text="3. Search song",
            command=self.search_song,
            height=44,
        )
        self.search_button.grid(row=10, column=0, padx=15, pady=6, sticky="ew")

        self.database_status_label = ctk.CTkLabel(
            controls,
            text="Database status: checking...",
            justify="left",
            wraplength=270,
            text_color="gray75",
        )
        self.database_status_label.grid(
            row=11,
            column=0,
            padx=15,
            pady=(18, 18),
            sticky="w",
        )

        main = ctk.CTkFrame(self)
        main.grid(
            row=1,
            column=1,
            padx=(9, 18),
            pady=18,
            sticky="nsew",
        )
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(3, weight=1)

        status_frame = ctk.CTkFrame(main)
        status_frame.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Ready",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.status_label.grid(row=0, column=0, padx=14, pady=(12, 5), sticky="w")

        self.progress_bar = ctk.CTkProgressBar(status_frame)
        self.progress_bar.grid(row=1, column=0, padx=14, pady=(5, 12), sticky="ew")
        self.progress_bar.set(0)

        result_frame = ctk.CTkFrame(main)
        result_frame.grid(row=1, column=0, padx=16, pady=8, sticky="ew")
        result_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            result_frame,
            text="Recognition result",
            font=ctk.CTkFont(size=19, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=14, pady=(12, 8), sticky="w")

        ctk.CTkLabel(result_frame, text="Status:").grid(
            row=1, column=0, padx=(14, 8), pady=4, sticky="w"
        )
        self.result_status_label = ctk.CTkLabel(
            result_frame,
            text="No search performed",
            font=ctk.CTkFont(weight="bold"),
        )
        self.result_status_label.grid(
            row=1, column=1, padx=(0, 14), pady=4, sticky="w"
        )

        ctk.CTkLabel(result_frame, text="Best candidate:").grid(
            row=2, column=0, padx=(14, 8), pady=(4, 12), sticky="w"
        )
        self.best_match_label = ctk.CTkLabel(
            result_frame,
            text="-",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.best_match_label.grid(
            row=2, column=1, padx=(0, 14), pady=(4, 12), sticky="w"
        )

        ctk.CTkLabel(
            main,
            text="Output and ranking",
            font=ctk.CTkFont(size=19, weight="bold"),
        ).grid(row=2, column=0, padx=18, pady=(10, 5), sticky="w")

        self.output_box = ctk.CTkTextbox(
            main,
            font=("Consolas", 13),
            wrap="word",
        )
        self.output_box.grid(row=3, column=0, padx=16, pady=(5, 16), sticky="nsew")
        self.output_box.configure(state="disabled")

    # ------------------------------------------------------------------
    # Output and task management
    # ------------------------------------------------------------------

    def append_output(self, text: str) -> None:
        self.output_box.configure(state="normal")
        self.output_box.insert("end", text)
        self.output_box.see("end")
        self.output_box.configure(state="disabled")

    def clear_output(self) -> None:
        self.output_box.configure(state="normal")
        self.output_box.delete("1.0", "end")
        self.output_box.configure(state="disabled")

    def set_busy(self, busy: bool, status: str) -> None:
        self.busy = busy
        self.status_label.configure(text=status)

        state = "disabled" if busy else "normal"
        for button in (
            self.database_button,
            self.record_button,
            self.search_button,
            self.refresh_mic_button,
        ):
            button.configure(state=state)

        if busy:
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start()
        else:
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self.progress_bar.set(0)

    def run_script(
        self,
        script: Path,
        status: str,
        on_complete=None,
    ) -> None:
        if self.busy:
            messagebox.showinfo("Busy", "Another operation is already running.")
            return

        if not script.exists():
            messagebox.showerror("Missing file", f"Could not find:\n{script.name}")
            return

        self.set_busy(True, status)
        self.clear_output()
        self.append_output(f"Running: {script.name}\n")
        self.append_output("=" * 60 + "\n")

        def worker() -> None:
            output_lines: list[str] = []
            return_code = 1

            try:
                process = subprocess.Popen(
                    [sys.executable, "-u", str(script)],
                    cwd=BASE_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )

                assert process.stdout is not None

                for line in process.stdout:
                    output_lines.append(line)
                    self.after(0, lambda value=line: self.append_output(value))

                return_code = process.wait()

            except Exception as error:
                error_text = f"\nGUI execution error: {error}\n"
                output_lines.append(error_text)
                self.after(0, lambda: self.append_output(error_text))

            complete_output = "".join(output_lines)

            def finish() -> None:
                self.set_busy(False, "Ready")

                if on_complete is not None:
                    on_complete(return_code, complete_output)

                if return_code != 0:
                    messagebox.showerror(
                        "Operation failed",
                        f"{script.name} ended with an error.\n"
                        "Check the output box for details.",
                    )

            self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Microphones
    # ------------------------------------------------------------------

    def load_saved_microphone(self) -> int | None:
        if not MIC_CONFIG_FILE.exists():
            return None

        try:
            with MIC_CONFIG_FILE.open("r", encoding="utf-8") as file:
                data = json.load(file)

            value = data.get("device")
            return int(value) if value is not None else None

        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def refresh_microphones(self, show_message: bool = True) -> None:
        try:
            devices = sd.query_devices()
            saved_index = self.load_saved_microphone()

            self.microphone_map = {"System default microphone": None}
            selected_label = "System default microphone"

            for index, device in enumerate(devices):
                if int(device["max_input_channels"]) <= 0:
                    continue

                name = str(device["name"])
                channels = int(device["max_input_channels"])
                label = f"{index}: {name} ({channels} input)"
                self.microphone_map[label] = index

                if saved_index == index:
                    selected_label = label

            values = list(self.microphone_map.keys())
            self.microphone_box.configure(values=values)
            self.microphone_box.set(selected_label)

            selected_index = self.microphone_map[selected_label]
            self.update_mic_index_label(selected_index)

            if show_message:
                messagebox.showinfo(
                    "Microphones",
                    f"Found {len(values) - 1} microphone input device(s).",
                )

        except Exception as error:
            messagebox.showerror(
                "Microphone error",
                f"Could not load microphone devices:\n{error}",
            )

    def select_microphone(self, choice: str) -> None:
        index = self.microphone_map.get(choice)

        try:
            with MIC_CONFIG_FILE.open("w", encoding="utf-8") as file:
                json.dump({"device": index}, file, indent=2)

            self.update_mic_index_label(index)
            self.status_label.configure(
                text=(
                    "System default microphone selected"
                    if index is None
                    else f"Microphone index {index} selected"
                )
            )

        except OSError as error:
            messagebox.showerror(
                "Configuration error",
                f"Could not save microphone selection:\n{error}",
            )

    def update_mic_index_label(self, index: int | None) -> None:
        text = "Selected index: default" if index is None else f"Selected index: {index}"
        self.mic_index_label.configure(text=text)

    # ------------------------------------------------------------------
    # Main operations
    # ------------------------------------------------------------------

    def build_database(self) -> None:
        self.run_script(
            CREATE_DATABASE_SCRIPT,
            "Building fingerprint database...",
            self.database_finished,
        )

    def database_finished(self, return_code: int, output: str) -> None:
        self.refresh_database_status()

        if return_code == 0 and "Completed:" in output:
            messagebox.showinfo(
                "Database complete",
                "The fingerprints and database.json were created.",
            )

    def record_query(self) -> None:
        self.result_status_label.configure(text="Recording query...")
        self.best_match_label.configure(text="-")

        self.run_script(
            RECORDER_SCRIPT,
            "Recording for 12 seconds...",
            self.recording_finished,
        )

    def recording_finished(self, return_code: int, output: str) -> None:
        if return_code == 0 and QUERY_WAV_FILE.exists():
            self.result_status_label.configure(text="Query recorded")
            messagebox.showinfo(
                "Recording complete",
                "The query was recorded and query.npy was generated.",
            )
        else:
            self.result_status_label.configure(text="Recording failed")

    def search_song(self) -> None:
        self.result_status_label.configure(text="Searching...")
        self.best_match_label.configure(text="-")

        self.run_script(
            MATCHER_SCRIPT,
            "Searching fingerprint database...",
            self.search_finished,
        )

    def search_finished(self, return_code: int, output: str) -> None:
        if return_code != 0:
            self.result_status_label.configure(text="Search failed")
            return

        status, candidate = self.parse_match_result(output)
        self.result_status_label.configure(text=status)
        self.best_match_label.configure(text=candidate or "-")

        if status == "Reliable match":
            messagebox.showinfo("Song detected", f"Best match:\n{candidate}")
        elif status == "Uncertain match":
            messagebox.showwarning(
                "Uncertain result",
                f"Most likely song:\n{candidate}\n\n"
                "The top candidates are too close.",
            )
        else:
            messagebox.showwarning(
                "No reliable match",
                "The recording did not produce a sufficiently distinct match.",
            )

    @staticmethod
    def parse_match_result(output: str) -> tuple[str, str]:
        lines = [line.strip() for line in output.splitlines()]

        try:
            marker = lines.index("===== Best Match =====")
            result_lines = [line for line in lines[marker + 1 :] if line]
        except ValueError:
            return "No reliable match", ""

        if not result_lines:
            return "No reliable match", ""

        first = result_lines[0]

        if first == "No reliable match.":
            return "No reliable match", ""

        if first == "Match is uncertain.":
            for line in result_lines:
                if line.startswith("Most likely:"):
                    return "Uncertain match", line.split(":", 1)[1].strip()
            return "Uncertain match", ""

        return "Reliable match", first

    # ------------------------------------------------------------------
    # Playback and folders
    # ------------------------------------------------------------------

    def play_query(self) -> None:
        if not QUERY_WAV_FILE.exists():
            messagebox.showwarning("No recording", "Record a query first.")
            return

        try:
            audio, sample_rate = sf.read(
                QUERY_WAV_FILE,
                dtype="float32",
                always_2d=False,
            )
            sd.stop()
            sd.play(audio, sample_rate)
            self.status_label.configure(text="Playing query.wav...")

        except Exception as error:
            messagebox.showerror("Playback error", str(error))

    def stop_playback(self) -> None:
        sd.stop()
        self.status_label.configure(text="Playback stopped")

    def open_songs_folder(self) -> None:
        SONGS_FOLDER.mkdir(parents=True, exist_ok=True)

        try:
            system = platform.system()

            if system == "Windows":
                os.startfile(SONGS_FOLDER)  # type: ignore[attr-defined]
            elif system == "Darwin":
                subprocess.Popen(["open", str(SONGS_FOLDER)])
            else:
                subprocess.Popen(["xdg-open", str(SONGS_FOLDER)])

        except Exception as error:
            messagebox.showerror("Folder error", str(error))

    def refresh_database_status(self) -> None:
        song_count = len(
            [
                path
                for path in SONGS_FOLDER.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            ]
        )
        fingerprint_count = len(list(FINGERPRINTS_FOLDER.glob("*.npy")))
        metadata = "created" if DATABASE_FILE.exists() else "not created"

        self.database_status_label.configure(
            text=(
                f"Songs: {song_count}\n"
                f"Fingerprints: {fingerprint_count}\n"
                f"database.json: {metadata}"
            )
        )

    def close_app(self) -> None:
        try:
            sd.stop()
        finally:
            self.destroy()


if __name__ == "__main__":
    app = SongRecognizerGUI()
    app.mainloop()
