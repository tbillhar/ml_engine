"""PySide6 desktop GUI for running the FX ML pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Ensure imports work when launched via: python app/gui_app.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import HORIZON, STEP_DAYS, TEST_DAYS, TRAIN_DAYS
from src.data_pipeline import load_csv, parse_and_sort_dates
from src.feature_engineering import compute_future_returns
from src.long_format import build_long
from src.pnl_analysis import (
    compute_equal_weight_pnl,
    compute_topK_pnl,
    cumulative_curve,
    perf_stats,
)
from src.walkforward_model import run_walkforward_model


class PipelineWorker(QObject):
    """Background worker to run the pipeline and stream progress/log events."""

    log = Signal(str)
    progress = Signal(int)
    completed = Signal(object, str)
    failed = Signal(str)

    def __init__(
        self,
        csv_path: str,
        train_days: int,
        test_days: int,
        step_days: int,
        horizon: int,
        output_dir: Path,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.horizon = horizon
        self.output_dir = output_dir

    def run(self) -> None:
        try:
            stats_df, plot_path = run_pipeline(
                csv_path=self.csv_path,
                train_days=self.train_days,
                test_days=self.test_days,
                step_days=self.step_days,
                horizon=self.horizon,
                output_dir=self.output_dir,
                log_fn=self.log.emit,
                progress_fn=self.progress.emit,
            )
            self.completed.emit(stats_df, str(plot_path))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


def run_pipeline(
    csv_path: str,
    train_days: int,
    test_days: int,
    step_days: int,
    horizon: int,
    output_dir: Path,
    log_fn=None,
    progress_fn=None,
):
    """Run the existing FX pipeline and save artifacts to output_dir."""

    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    def set_progress(value: int) -> None:
        if progress_fn:
            progress_fn(value)

    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading input CSV: {csv_path}")
    set_progress(10)
    df = load_csv(csv_path)
    df = parse_and_sort_dates(df)

    ret_cols = [c for c in df.columns if c.startswith("ret_")]
    pairs = [c.replace("ret_", "") for c in ret_cols]
    log(f"Detected {len(pairs)} pairs")

    log(f"Computing {horizon}-day future returns")
    set_progress(25)
    df = compute_future_returns(df, horizon=horizon)

    log("Building long-format ranking table")
    set_progress(40)
    long_df = build_long(df, pairs)

    log(
        "Running walk-forward model "
        f"(train={train_days}, test={test_days}, step={step_days})"
    )
    set_progress(60)
    pred_df = run_walkforward_model(
        long_df,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )

    log("Computing PnL series")
    set_progress(75)
    pnl_top1 = compute_topK_pnl(pred_df, 1)
    pnl_top3 = compute_topK_pnl(pred_df, 3)
    pnl_top5 = compute_topK_pnl(pred_df, 5)
    pnl_eq = compute_equal_weight_pnl(pred_df)

    cum1 = cumulative_curve(pnl_top1)
    cum3 = cumulative_curve(pnl_top3)
    cum5 = cumulative_curve(pnl_top5)
    cum_eq = cumulative_curve(pnl_eq)

    top1_stats = perf_stats(pnl_top1, horizon_days=horizon)
    top3_stats = perf_stats(pnl_top3, horizon_days=horizon)
    top5_stats = perf_stats(pnl_top5, horizon_days=horizon)
    eq_stats = perf_stats(pnl_eq, horizon_days=horizon)

    log("Saving CSV outputs")
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    pnl_top1.to_csv(output_dir / "pnl_top1.csv", index=False)
    pnl_top3.to_csv(output_dir / "pnl_top3.csv", index=False)
    pnl_top5.to_csv(output_dir / "pnl_top5.csv", index=False)
    pnl_eq.to_csv(output_dir / "pnl_equal_weight.csv", index=False)

    stats_df = pd.DataFrame(
        [
            {"strategy": "Top-1", **top1_stats},
            {"strategy": "Top-3", **top3_stats},
            {"strategy": "Top-5", **top5_stats},
            {"strategy": "Equal-weight", **eq_stats},
        ]
    )
    stats_df.to_csv(output_dir / "performance_summary.csv", index=False)

    log("Saving PnL plot")
    set_progress(90)
    plot_path = output_dir / "pnl_curves.png"
    plt.figure(figsize=(12, 6))
    plt.plot(cum1["Date"], cum1["cum"], label="Top-1", linewidth=2.5)
    plt.plot(cum3["Date"], cum3["cum"], label="Top-3", linewidth=2.2)
    plt.plot(cum5["Date"], cum5["cum"], label="Top-5", linewidth=2.0)
    plt.plot(
        cum_eq["Date"],
        cum_eq["cum"],
        label="Equal-Weight",
        linestyle="--",
        linewidth=1.8,
    )
    plt.title("Cumulative PnL")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    set_progress(100)
    log(f"Done. Outputs saved to: {output_dir.resolve()}")
    return stats_df, plot_path


class FXPipelineWindow(QMainWindow):
    """Main desktop GUI window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FX ML Pipeline")
        self.resize(1000, 700)

        self.output_dir = ROOT_DIR / "outputs"
        self.worker_thread: QThread | None = None
        self.worker: PipelineWorker | None = None

        self.csv_path_label = QLabel(str(ROOT_DIR / "data" / "fx_features_wide.csv"))
        self.csv_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        select_csv_btn = QPushButton("Select CSV")
        select_csv_btn.clicked.connect(self.select_csv)

        self.train_input = QLineEdit(str(TRAIN_DAYS))
        self.test_input = QLineEdit(str(TEST_DAYS))
        self.step_input = QLineEdit(str(STEP_DAYS))
        self.horizon_input = QLineEdit(str(HORIZON))

        self.run_btn = QPushButton("Run Pipeline")
        self.run_btn.clicked.connect(self.start_pipeline)

        self.open_output_btn = QPushButton("Open Output Folder")
        self.open_output_btn.clicked.connect(self.open_output_folder)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.status_label = QLabel("Status: Idle")

        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)

        self.summary_table = QTableWidget(0, 5)
        self.summary_table.setHorizontalHeaderLabels(
            [
                "Strategy",
                "Annualized Return",
                "Annualized Vol",
                "Sharpe",
                "Cumulative Return",
            ]
        )

        self.plot_label = QLabel("PnL plot will appear here after a run.")
        self.plot_label.setAlignment(Qt.AlignCenter)

        top_controls = QHBoxLayout()
        top_controls.addWidget(select_csv_btn)
        top_controls.addWidget(self.csv_path_label, 1)

        params_form = QFormLayout()
        params_form.addRow("TRAIN_DAYS", self.train_input)
        params_form.addRow("TEST_DAYS", self.test_input)
        params_form.addRow("STEP_DAYS", self.step_input)
        params_form.addRow("HORIZON", self.horizon_input)

        action_row = QHBoxLayout()
        action_row.addWidget(self.run_btn)
        action_row.addWidget(self.open_output_btn)
        action_row.addStretch(1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addLayout(top_controls)
        left_layout.addLayout(params_form)
        left_layout.addLayout(action_row)
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(QLabel("Logs"))
        left_layout.addWidget(self.logs, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("Performance Summary"))
        right_layout.addWidget(self.summary_table, 1)
        right_layout.addWidget(QLabel("PnL Plot"))
        right_layout.addWidget(self.plot_label, 2)

        splitter = QSplitter()
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([420, 580])

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(splitter)
        self.setCentralWidget(container)

    def append_log(self, msg: str) -> None:
        self.logs.appendPlainText(msg)

    def select_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select input CSV",
            str(ROOT_DIR / "data"),
            "CSV Files (*.csv)",
        )
        if path:
            self.csv_path_label.setText(path)

    def _read_int(self, widget: QLineEdit, name: str) -> int:
        text = widget.text().strip()
        value = int(text)
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def start_pipeline(self) -> None:
        try:
            csv_path = self.csv_path_label.text().strip()
            if not csv_path:
                raise ValueError("Please select a CSV file")
            if not Path(csv_path).exists():
                raise ValueError(f"CSV file does not exist: {csv_path}")

            train_days = self._read_int(self.train_input, "TRAIN_DAYS")
            test_days = self._read_int(self.test_input, "TEST_DAYS")
            step_days = self._read_int(self.step_input, "STEP_DAYS")
            horizon = self._read_int(self.horizon_input, "HORIZON")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Invalid Input", str(exc))
            return

        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Running...")
        self.logs.clear()
        self.append_log("Starting pipeline...")

        self.worker_thread = QThread()
        self.worker = PipelineWorker(
            csv_path=csv_path,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            horizon=horizon,
            output_dir=self.output_dir,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)

        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.completed.connect(self.on_pipeline_completed)
        self.worker.failed.connect(self.on_pipeline_failed)

        self.worker.completed.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def on_pipeline_completed(self, stats_df: pd.DataFrame, plot_path: str) -> None:
        self.run_btn.setEnabled(True)
        self.status_label.setText("Status: Done")
        self.populate_summary_table(stats_df)
        self.load_plot(plot_path)
        self.append_log("Pipeline completed successfully.")

    def on_pipeline_failed(self, error_msg: str) -> None:
        self.run_btn.setEnabled(True)
        self.status_label.setText("Status: Error")
        self.append_log(f"Error: {error_msg}")
        QMessageBox.critical(self, "Pipeline Error", error_msg)

    def populate_summary_table(self, stats_df: pd.DataFrame) -> None:
        self.summary_table.setRowCount(len(stats_df))
        for i, row in stats_df.iterrows():
            self.summary_table.setItem(i, 0, QTableWidgetItem(str(row["strategy"])))
            self.summary_table.setItem(i, 1, QTableWidgetItem(f"{row['Annualized Return']:.6f}"))
            self.summary_table.setItem(i, 2, QTableWidgetItem(f"{row['Annualized Vol']:.6f}"))
            self.summary_table.setItem(i, 3, QTableWidgetItem(f"{row['Sharpe']:.6f}"))
            self.summary_table.setItem(i, 4, QTableWidgetItem(f"{row['Cumulative Return']:.6f}"))
        self.summary_table.resizeColumnsToContents()

    def load_plot(self, plot_path: str) -> None:
        pixmap = QPixmap(plot_path)
        if pixmap.isNull():
            self.plot_label.setText("Failed to load pnl_curves.png")
            return
        scaled = pixmap.scaled(
            self.plot_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.plot_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        current = self.output_dir / "pnl_curves.png"
        if current.exists() and self.plot_label.pixmap() is not None:
            self.load_plot(str(current))

    def open_output_folder(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(self.output_dir)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(self.output_dir)], check=False)
        else:
            subprocess.run(["xdg-open", str(self.output_dir)], check=False)


def main() -> None:
    app = QApplication(sys.argv)
    window = FXPipelineWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
