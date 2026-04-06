"""PySide6 desktop GUI for running the FX ML pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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

from src.config import (
    CALIBRATION_DAYS,
    FEATURE_DATA_FILENAME,
    FIT_DAYS,
    HORIZON,
    P_WIN_THRESHOLD,
    RAW_DATA_FILENAME,
    STEP_DAYS,
    TEST_DAYS,
    TRADING_DAYS_PER_YEAR,
    TRANSACTION_LOSS_PCT,
)
from src.feature_dataset_builder import build_feature_dataset
from src.pipeline_runner import run_pipeline
from src.raw_data_loader import download_raw_fx_data


class PipelineWorker(QObject):
    """Background worker to run the pipeline and stream progress/log events."""

    log = Signal(str)
    progress = Signal(int)
    completed = Signal(object, str)
    failed = Signal(str)

    def __init__(
        self,
        csv_path: str,
        fit_days: int,
        calibration_days: int,
        test_days: int,
        step_days: int,
        horizon: int,
        transaction_loss_pct: float,
        trading_days_per_year: int,
        p_win_threshold: float,
        output_dir: Path,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.fit_days = fit_days
        self.calibration_days = calibration_days
        self.test_days = test_days
        self.step_days = step_days
        self.horizon = horizon
        self.transaction_loss_pct = transaction_loss_pct
        self.trading_days_per_year = trading_days_per_year
        self.p_win_threshold = p_win_threshold
        self.output_dir = output_dir

    def run(self) -> None:
        try:
            stats_df, plot_path = run_pipeline(
                csv_path=self.csv_path,
                fit_days=self.fit_days,
                calibration_days=self.calibration_days,
                test_days=self.test_days,
                step_days=self.step_days,
                horizon=self.horizon,
                transaction_loss_pct=self.transaction_loss_pct,
                trading_days_per_year=self.trading_days_per_year,
                p_win_threshold=self.p_win_threshold,
                output_dir=self.output_dir,
                log_fn=self.log.emit,
                progress_fn=self.progress.emit,
            )
            self.completed.emit(stats_df, str(plot_path))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class RawDataWorker(QObject):
    """Background worker to download raw Yahoo Finance data."""

    log = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, output_path: Path) -> None:
        super().__init__()
        self.output_path = output_path

    def run(self) -> None:
        try:
            path = download_raw_fx_data(self.output_path, log_fn=self.log.emit)
            self.completed.emit(str(path))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class FeatureBuildWorker(QObject):
    """Background worker to build a feature CSV from raw OHLC data."""

    log = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, raw_csv_path: Path, output_path: Path) -> None:
        super().__init__()
        self.raw_csv_path = raw_csv_path
        self.output_path = output_path

    def run(self) -> None:
        try:
            path = build_feature_dataset(
                raw_csv_path=self.raw_csv_path,
                output_csv_path=self.output_path,
                log_fn=self.log.emit,
            )
            self.completed.emit(str(path))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class FXPipelineWindow(QMainWindow):
    """Main desktop GUI window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FX ML Pipeline")
        self.resize(1000, 700)

        self.output_dir = ROOT_DIR / "outputs"
        self.raw_data_path = ROOT_DIR / "data" / RAW_DATA_FILENAME
        self.feature_data_path = ROOT_DIR / "data" / FEATURE_DATA_FILENAME
        self.pipeline_thread: QThread | None = None
        self.pipeline_worker: PipelineWorker | None = None
        self.task_thread: QThread | None = None
        self.task_worker: RawDataWorker | FeatureBuildWorker | None = None

        self.csv_path_label = QLabel(str(self.feature_data_path))
        self.csv_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.raw_data_path_label = QLabel(str(self.raw_data_path))
        self.raw_data_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.feature_data_path_label = QLabel(str(self.feature_data_path))
        self.feature_data_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        select_csv_btn = QPushButton("Select CSV")
        select_csv_btn.clicked.connect(self.select_csv)
        self.download_raw_btn = QPushButton("Download Yahoo Data")
        self.download_raw_btn.clicked.connect(self.start_raw_download)
        self.build_features_btn = QPushButton("Build Feature CSV")
        self.build_features_btn.clicked.connect(self.start_feature_build)

        self.fit_input = QLineEdit(str(FIT_DAYS))
        self.calibration_input = QLineEdit(str(CALIBRATION_DAYS))
        self.test_input = QLineEdit(str(TEST_DAYS))
        self.step_input = QLineEdit(str(STEP_DAYS))
        self.horizon_input = QLineEdit(str(HORIZON))
        self.transaction_loss_input = QLineEdit(str(TRANSACTION_LOSS_PCT))
        self.trading_days_input = QLineEdit(str(TRADING_DAYS_PER_YEAR))
        self.p_win_threshold_input = QLineEdit(str(P_WIN_THRESHOLD))

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
                "Annualized Volatility",
                "Sharpe",
                "Cumulative Return",
            ]
        )

        self.plot_label = QLabel("PnL plot will appear here after a run.")
        self.plot_label.setAlignment(Qt.AlignCenter)

        top_controls = QHBoxLayout()
        top_controls.addWidget(select_csv_btn)
        top_controls.addWidget(self.csv_path_label, 1)

        data_form = QFormLayout()
        data_form.addRow("RAW_DATA_CSV", self.raw_data_path_label)
        data_form.addRow("FEATURE_CSV", self.feature_data_path_label)

        params_form = QFormLayout()
        params_form.addRow("FIT_DAYS", self.fit_input)
        params_form.addRow("CALIBRATION_DAYS", self.calibration_input)
        params_form.addRow("TEST_DAYS", self.test_input)
        params_form.addRow("STEP_DAYS", self.step_input)
        params_form.addRow("HORIZON", self.horizon_input)
        params_form.addRow("TRANSACTION_LOSS_PCT", self.transaction_loss_input)
        params_form.addRow("TRADING_DAYS_PER_YEAR", self.trading_days_input)
        params_form.addRow("P_WIN_THRESHOLD", self.p_win_threshold_input)

        action_row = QHBoxLayout()
        action_row.addWidget(self.download_raw_btn)
        action_row.addWidget(self.build_features_btn)
        action_row.addWidget(self.run_btn)
        action_row.addWidget(self.open_output_btn)
        action_row.addStretch(1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addLayout(top_controls)
        left_layout.addLayout(data_form)
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

    def _set_busy(self, busy: bool) -> None:
        self.download_raw_btn.setEnabled(not busy)
        self.build_features_btn.setEnabled(not busy)
        self.run_btn.setEnabled(not busy)

    def _task_running(self) -> bool:
        return bool(
            (self.pipeline_thread and self.pipeline_thread.isRunning())
            or (self.task_thread and self.task_thread.isRunning())
        )

    def select_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select input CSV",
            str(ROOT_DIR / "data"),
            "CSV Files (*.csv)",
        )
        if path:
            self.csv_path_label.setText(path)

    def start_raw_download(self) -> None:
        if self._task_running():
            QMessageBox.warning(self, "Busy", "Please wait for the current task to finish.")
            return

        self._set_busy(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Downloading raw Yahoo data...")
        self.append_log("Starting Yahoo Finance download...")

        self.task_thread = QThread()
        self.task_worker = RawDataWorker(self.raw_data_path)
        self.task_worker.moveToThread(self.task_thread)
        self.task_thread.started.connect(self.task_worker.run)

        self.task_worker.log.connect(self.append_log)
        self.task_worker.completed.connect(self.on_raw_download_completed)
        self.task_worker.failed.connect(self.on_background_task_failed)

        self.task_worker.completed.connect(self.task_thread.quit)
        self.task_worker.failed.connect(self.task_thread.quit)
        self.task_thread.finished.connect(self.task_thread.deleteLater)
        self.task_thread.finished.connect(self.on_task_thread_finished)

        self.task_thread.start()

    def start_feature_build(self) -> None:
        if self._task_running():
            QMessageBox.warning(self, "Busy", "Please wait for the current task to finish.")
            return
        if not self.raw_data_path.exists():
            QMessageBox.critical(
                self,
                "Missing Raw Data",
                f"Raw data file does not exist: {self.raw_data_path}",
            )
            return

        self._set_busy(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Building feature CSV...")
        self.append_log("Building feature CSV from raw Yahoo data...")

        self.task_thread = QThread()
        self.task_worker = FeatureBuildWorker(self.raw_data_path, self.feature_data_path)
        self.task_worker.moveToThread(self.task_thread)
        self.task_thread.started.connect(self.task_worker.run)

        self.task_worker.log.connect(self.append_log)
        self.task_worker.completed.connect(self.on_feature_build_completed)
        self.task_worker.failed.connect(self.on_background_task_failed)

        self.task_worker.completed.connect(self.task_thread.quit)
        self.task_worker.failed.connect(self.task_thread.quit)
        self.task_thread.finished.connect(self.task_thread.deleteLater)
        self.task_thread.finished.connect(self.on_task_thread_finished)

        self.task_thread.start()

    def _read_int(self, widget: QLineEdit, name: str) -> int:
        text = widget.text().strip()
        value = int(text)
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def _read_nonnegative_float(self, widget: QLineEdit, name: str) -> float:
        text = widget.text().strip()
        value = float(text)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    def start_pipeline(self) -> None:
        try:
            csv_path = self.csv_path_label.text().strip()
            if not csv_path:
                raise ValueError("Please select a CSV file")
            if not Path(csv_path).exists():
                raise ValueError(f"CSV file does not exist: {csv_path}")

            fit_days = self._read_int(self.fit_input, "FIT_DAYS")
            calibration_days = self._read_int(self.calibration_input, "CALIBRATION_DAYS")
            test_days = self._read_int(self.test_input, "TEST_DAYS")
            step_days = self._read_int(self.step_input, "STEP_DAYS")
            horizon = self._read_int(self.horizon_input, "HORIZON")
            transaction_loss_pct = self._read_nonnegative_float(
                self.transaction_loss_input,
                "TRANSACTION_LOSS_PCT",
            )
            trading_days_per_year = self._read_int(
                self.trading_days_input,
                "TRADING_DAYS_PER_YEAR",
            )
            p_win_threshold = self._read_nonnegative_float(
                self.p_win_threshold_input,
                "P_WIN_THRESHOLD",
            )
            if p_win_threshold > 1:
                raise ValueError("P_WIN_THRESHOLD must be between 0 and 1")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Invalid Input", str(exc))
            return

        self.run_btn.setEnabled(False)
        self.download_raw_btn.setEnabled(False)
        self.build_features_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Running...")
        self.logs.clear()
        self.append_log("Starting pipeline...")

        self.pipeline_thread = QThread()
        self.pipeline_worker = PipelineWorker(
            csv_path=csv_path,
            fit_days=fit_days,
            calibration_days=calibration_days,
            test_days=test_days,
            step_days=step_days,
            horizon=horizon,
            transaction_loss_pct=transaction_loss_pct,
            trading_days_per_year=trading_days_per_year,
            p_win_threshold=p_win_threshold,
            output_dir=self.output_dir,
        )
        self.pipeline_worker.moveToThread(self.pipeline_thread)
        self.pipeline_thread.started.connect(self.pipeline_worker.run)

        self.pipeline_worker.log.connect(self.append_log)
        self.pipeline_worker.progress.connect(self.progress_bar.setValue)
        self.pipeline_worker.completed.connect(self.on_pipeline_completed)
        self.pipeline_worker.failed.connect(self.on_pipeline_failed)

        self.pipeline_worker.completed.connect(self.pipeline_thread.quit)
        self.pipeline_worker.failed.connect(self.pipeline_thread.quit)
        self.pipeline_thread.finished.connect(self.pipeline_thread.deleteLater)
        self.pipeline_thread.finished.connect(self.on_pipeline_thread_finished)

        self.pipeline_thread.start()

    def on_pipeline_completed(self, stats_df: pd.DataFrame, plot_path: str) -> None:
        self.status_label.setText("Status: Done")
        self.populate_summary_table(stats_df)
        self.load_plot(plot_path)
        self.append_log("Pipeline completed successfully.")

    def on_pipeline_failed(self, error_msg: str) -> None:
        self.status_label.setText("Status: Error")
        self.append_log(f"Error: {error_msg}")
        QMessageBox.critical(self, "Pipeline Error", error_msg)

    def on_raw_download_completed(self, raw_csv_path: str) -> None:
        self.progress_bar.setValue(100)
        self.raw_data_path_label.setText(raw_csv_path)
        self.status_label.setText("Status: Raw Yahoo data downloaded")
        self.append_log(f"Raw data download completed: {raw_csv_path}")

    def on_feature_build_completed(self, feature_csv_path: str) -> None:
        self.progress_bar.setValue(100)
        self.feature_data_path_label.setText(feature_csv_path)
        self.csv_path_label.setText(feature_csv_path)
        self.status_label.setText("Status: Feature CSV ready")
        self.append_log(f"Feature CSV build completed: {feature_csv_path}")

    def on_background_task_failed(self, error_msg: str) -> None:
        self.status_label.setText("Status: Error")
        self.append_log(f"Error: {error_msg}")
        QMessageBox.critical(self, "Background Task Error", error_msg)

    def on_pipeline_thread_finished(self) -> None:
        self.pipeline_thread = None
        self.pipeline_worker = None
        self._set_busy(False)

    def on_task_thread_finished(self) -> None:
        self.task_thread = None
        self.task_worker = None
        self._set_busy(False)

    def populate_summary_table(self, stats_df: pd.DataFrame) -> None:
        self.summary_table.setRowCount(len(stats_df))
        for i, row in stats_df.iterrows():
            self.summary_table.setItem(i, 0, QTableWidgetItem(str(row["strategy"])))
            self.summary_table.setItem(i, 1, QTableWidgetItem(f"{row['Annualized Return'] * 100:.1f}%"))
            self.summary_table.setItem(i, 2, QTableWidgetItem(f"{row['Annualized Vol'] * 100:.1f}%"))
            self.summary_table.setItem(i, 3, QTableWidgetItem(f"{row['Sharpe']:.6f}"))
            self.summary_table.setItem(i, 4, QTableWidgetItem(f"{row['Cumulative Return'] * 100:.1f}%"))
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
