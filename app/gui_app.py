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
    QComboBox,
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
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)

# Ensure imports work when launched via: python app/gui_app.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    ALPHAVANTAGE_API_KEY,
    FEATURE_DATA_FILENAME,
    FIT_DAYS,
    HORIZON,
    HOLDOUT_DAYS,
    LIVE_MODEL,
    MODEL_FIT_WINDOWS,
    MODEL_ROUTER_CANDIDATES,
    REBALANCE_DAYS,
    RAW_DATA_SOURCE,
    RAW_DATA_FILENAME,
    RETRAIN_DETERIORATION_LOOKBACK_DAYS,
    RETRAIN_DETERIORATION_MAX_AVG_EV,
    RETRAIN_DETERIORATION_MIN_WIN_RATE,
    SPECIALIST_ENSEMBLE_MEMBERS,
    SPECIALIST_MIN_MODEL_HOLD_DAYS,
    SPECIALIST_SWITCH_MARGIN_MIN_AVG_EV,
    SPECIALIST_SWITCH_REQUIRE_POSITIVE_EV,
    SPECIALIST_WEIGHTING_MODE,
    SPECIALIST_WEIGHT_LOOKBACK_DAYS,
    STEP_DAYS,
    TRADING_DAYS_PER_YEAR,
    TRANSACTION_LOSS_PCT,
)
from src.feature_dataset_builder import build_feature_dataset
from src.pipeline_runner import diagnostics_guide_text, run_pipeline
from src.raw_data_loader import DATA_SOURCE_INFO, SUPPORTED_DATA_SOURCES, download_raw_fx_data


class PipelineWorker(QObject):
    """Background worker to run the pipeline and stream progress/log events."""

    log = Signal(str)
    progress = Signal(int)
    completed = Signal(object, str, str, str)
    failed = Signal(str)

    def __init__(
        self,
        csv_path: str,
        fit_days: int,
        model_fit_windows: list[int],
        step_days: int,
        rebalance_days: int,
        horizon: int,
        transaction_loss_pct: float,
        trading_days_per_year: int,
        holdout_days: int,
        live_model: str,
        specialist_weighting_mode: str,
        specialist_ensemble_members: list[str],
        model_router_candidates: list[str],
        specialist_weight_lookback_days: int,
        specialist_min_model_hold_days: int,
        specialist_switch_margin_min_avg_ev: float,
        specialist_switch_require_positive_ev: bool,
        retrain_deterioration_lookback_days: int,
        retrain_deterioration_min_win_rate: float,
        retrain_deterioration_max_avg_ev: float,
        output_dir: Path,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.fit_days = fit_days
        self.model_fit_windows = model_fit_windows
        self.step_days = step_days
        self.rebalance_days = rebalance_days
        self.horizon = horizon
        self.transaction_loss_pct = transaction_loss_pct
        self.trading_days_per_year = trading_days_per_year
        self.holdout_days = holdout_days
        self.live_model = live_model
        self.specialist_weighting_mode = specialist_weighting_mode
        self.specialist_ensemble_members = specialist_ensemble_members
        self.model_router_candidates = model_router_candidates
        self.specialist_weight_lookback_days = specialist_weight_lookback_days
        self.specialist_min_model_hold_days = specialist_min_model_hold_days
        self.specialist_switch_margin_min_avg_ev = specialist_switch_margin_min_avg_ev
        self.specialist_switch_require_positive_ev = specialist_switch_require_positive_ev
        self.retrain_deterioration_lookback_days = retrain_deterioration_lookback_days
        self.retrain_deterioration_min_win_rate = retrain_deterioration_min_win_rate
        self.retrain_deterioration_max_avg_ev = retrain_deterioration_max_avg_ev
        self.output_dir = output_dir

    def run(self) -> None:
        try:
            stats_df, plot_path, heatmap_path, diagnostics_summary = run_pipeline(
                csv_path=self.csv_path,
                fit_days=self.fit_days,
                model_fit_windows=self.model_fit_windows,
                step_days=self.step_days,
                rebalance_days=self.rebalance_days,
                horizon=self.horizon,
                transaction_loss_pct=self.transaction_loss_pct,
                trading_days_per_year=self.trading_days_per_year,
                holdout_days=self.holdout_days,
                live_model=self.live_model,
                specialist_weighting_mode=self.specialist_weighting_mode,
                specialist_ensemble_models=self.specialist_ensemble_members,
                model_router_candidates=self.model_router_candidates,
                specialist_weight_lookback_days=self.specialist_weight_lookback_days,
                specialist_min_model_hold_days=self.specialist_min_model_hold_days,
                specialist_switch_margin_min_avg_ev=self.specialist_switch_margin_min_avg_ev,
                specialist_switch_require_positive_ev=self.specialist_switch_require_positive_ev,
                retrain_deterioration_lookback_days=self.retrain_deterioration_lookback_days,
                retrain_deterioration_min_win_rate=self.retrain_deterioration_min_win_rate,
                retrain_deterioration_max_avg_ev=self.retrain_deterioration_max_avg_ev,
                output_dir=self.output_dir,
                log_fn=self.log.emit,
                progress_fn=self.progress.emit,
            )
            self.completed.emit(stats_df, str(plot_path), str(heatmap_path), diagnostics_summary)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class RawDataWorker(QObject):
    """Background worker to download raw FX data."""

    log = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, output_path: Path, source: str, alphavantage_api_key: str) -> None:
        super().__init__()
        self.output_path = output_path
        self.source = source
        self.alphavantage_api_key = alphavantage_api_key

    def run(self) -> None:
        try:
            path = download_raw_fx_data(
                self.output_path,
                source=self.source,
                alphavantage_api_key=self.alphavantage_api_key,
                log_fn=self.log.emit,
            )
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
        self.resize(1500, 920)

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
        self.raw_source_input = QComboBox()
        self.raw_source_input.addItems(list(SUPPORTED_DATA_SOURCES))
        self.raw_source_input.setCurrentText(RAW_DATA_SOURCE)
        self.raw_source_input.currentTextChanged.connect(self.update_raw_source_info)
        self.alphavantage_key_input = QLineEdit(ALPHAVANTAGE_API_KEY)
        self.alphavantage_key_input.setPlaceholderText("Alpha Vantage API key (only needed for alphavantage)")
        self.alphavantage_key_input.setEchoMode(QLineEdit.Password)
        self.raw_source_info_label = QLabel()
        self.raw_source_info_label.setWordWrap(True)
        self.raw_source_info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.download_raw_btn = QPushButton("Download Raw Data")
        self.download_raw_btn.clicked.connect(self.start_raw_download)
        self.build_features_btn = QPushButton("Build Feature CSV")
        self.build_features_btn.clicked.connect(self.start_feature_build)

        self.fit_input = QLineEdit(str(FIT_DAYS))
        self.model_fit_windows_input = QLineEdit(",".join(str(window) for window in MODEL_FIT_WINDOWS))
        self.step_input = QLineEdit(str(STEP_DAYS))
        self.rebalance_input = QLineEdit(str(REBALANCE_DAYS))
        self.horizon_input = QLineEdit(str(HORIZON))
        self.transaction_loss_input = QLineEdit(str(TRANSACTION_LOSS_PCT))
        self.trading_days_input = QLineEdit(str(TRADING_DAYS_PER_YEAR))
        self.holdout_days_input = QLineEdit(str(HOLDOUT_DAYS))
        self.specialist_weighting_mode_input = QComboBox()
        self.specialist_weighting_mode_input.addItems(
            ["equal", "soft_dynamic", "winner_take_all", "winner_take_most", "sticky_winner"]
        )
        self.specialist_weighting_mode_input.setCurrentText(SPECIALIST_WEIGHTING_MODE)
        self.specialist_members_input = QLineEdit(",".join(SPECIALIST_ENSEMBLE_MEMBERS))
        self.model_router_candidates_input = QLineEdit(",".join(MODEL_ROUTER_CANDIDATES))
        self.specialist_lookback_input = QLineEdit(str(SPECIALIST_WEIGHT_LOOKBACK_DAYS))
        self.specialist_min_hold_input = QLineEdit(str(SPECIALIST_MIN_MODEL_HOLD_DAYS))
        self.specialist_switch_margin_input = QLineEdit(str(SPECIALIST_SWITCH_MARGIN_MIN_AVG_EV))
        self.specialist_positive_ev_input = QComboBox()
        self.specialist_positive_ev_input.addItems(["true", "false"])
        self.specialist_positive_ev_input.setCurrentText(
            "true" if SPECIALIST_SWITCH_REQUIRE_POSITIVE_EV else "false"
        )
        self.retrain_lookback_input = QLineEdit(str(RETRAIN_DETERIORATION_LOOKBACK_DAYS))
        self.retrain_win_rate_input = QLineEdit(str(RETRAIN_DETERIORATION_MIN_WIN_RATE))
        self.retrain_avg_ev_input = QLineEdit(str(RETRAIN_DETERIORATION_MAX_AVG_EV))
        self.live_model_input = QComboBox()
        self.live_model_input.addItems(
            [
                "specialist_ensemble",
                "rf_corr_regime",
                "logreg_returns_momentum",
                "lgbm_deep_corr_regime",
                "ensemble",
                "lgbm_deep",
                "lgbm_deep_returns_momentum",
                "lgbm_deep_volatility",
                "rf",
                "rf_returns_momentum",
                "logreg",
                "logreg_corr_regime",
                "logreg_volatility",
            ]
        )
        self.live_model_input.setCurrentText(LIVE_MODEL)

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
                "Model",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe",
                "Cumulative Return",
            ]
        )
        self.summary_table.setAlternatingRowColors(True)
        self.summary_table.setSortingEnabled(True)
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setMinimumHeight(420)

        self.diagnostics_summary = QPlainTextEdit()
        self.diagnostics_summary.setReadOnly(True)
        self.diagnostics_summary.setPlaceholderText("Key observations from diagnostics will appear here after a run.")
        self.diagnostics_summary.setMinimumHeight(110)
        self.diagnostics_guide = QPlainTextEdit()
        self.diagnostics_guide.setReadOnly(True)
        self.diagnostics_guide.setMinimumHeight(130)
        self.diagnostics_guide.setPlainText(
            diagnostics_guide_text(
                SPECIALIST_ENSEMBLE_MEMBERS,
                MODEL_ROUTER_CANDIDATES,
                MODEL_FIT_WINDOWS,
                SPECIALIST_WEIGHT_LOOKBACK_DAYS,
                REBALANCE_DAYS,
                SPECIALIST_WEIGHTING_MODE,
                SPECIALIST_MIN_MODEL_HOLD_DAYS,
                SPECIALIST_SWITCH_MARGIN_MIN_AVG_EV,
                SPECIALIST_SWITCH_REQUIRE_POSITIVE_EV,
                RETRAIN_DETERIORATION_LOOKBACK_DAYS,
                RETRAIN_DETERIORATION_MIN_WIN_RATE,
                RETRAIN_DETERIORATION_MAX_AVG_EV,
            )
        )

        self.plot_label = QLabel("PnL plot will appear here after a run.")
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setMinimumHeight(320)
        self.plot_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.heatmap_label = QLabel("Correlation heatmap will appear here after a run.")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setMinimumHeight(320)
        self.heatmap_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        top_controls = QHBoxLayout()
        top_controls.addWidget(select_csv_btn)
        top_controls.addWidget(self.csv_path_label, 1)

        data_form = QFormLayout()
        data_form.addRow("RAW_DATA_CSV", self.raw_data_path_label)
        data_form.addRow("FEATURE_CSV", self.feature_data_path_label)
        data_form.addRow("RAW_DATA_SOURCE", self.raw_source_input)
        data_form.addRow("ALPHAVANTAGE_API_KEY", self.alphavantage_key_input)
        data_form.addRow("SOURCE_INFO", self.raw_source_info_label)

        params_form = QFormLayout()
        params_form.addRow("FIT_DAYS", self.fit_input)
        params_form.addRow("MODEL_FIT_WINDOWS", self.model_fit_windows_input)
        params_form.addRow("STEP_DAYS", self.step_input)
        params_form.addRow("REBALANCE_DAYS", self.rebalance_input)
        params_form.addRow("HORIZON", self.horizon_input)
        params_form.addRow("TRANSACTION_LOSS_PCT", self.transaction_loss_input)
        params_form.addRow("TRADING_DAYS_PER_YEAR", self.trading_days_input)
        params_form.addRow("HOLDOUT_DAYS", self.holdout_days_input)
        params_form.addRow("SPECIALIST_WEIGHTING_MODE", self.specialist_weighting_mode_input)
        params_form.addRow("SPECIALIST_ENSEMBLE_MEMBERS", self.specialist_members_input)
        params_form.addRow("MODEL_ROUTER_CANDIDATES", self.model_router_candidates_input)
        params_form.addRow("SPECIALIST_WEIGHT_LOOKBACK_DAYS", self.specialist_lookback_input)
        params_form.addRow("SPECIALIST_MIN_MODEL_HOLD_DAYS", self.specialist_min_hold_input)
        params_form.addRow("SPECIALIST_SWITCH_MARGIN_MIN_AVG_EV", self.specialist_switch_margin_input)
        params_form.addRow("SPECIALIST_SWITCH_REQUIRE_POSITIVE_EV", self.specialist_positive_ev_input)
        params_form.addRow("RETRAIN_DET_LOOKBACK_DAYS", self.retrain_lookback_input)
        params_form.addRow("RETRAIN_DET_MIN_WIN_RATE", self.retrain_win_rate_input)
        params_form.addRow("RETRAIN_DET_MAX_AVG_EV", self.retrain_avg_ev_input)
        params_form.addRow("LIVE_MODEL", self.live_model_input)

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

        summary_panel = QWidget()
        summary_layout = QVBoxLayout(summary_panel)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.addWidget(QLabel("Performance Summary"))
        summary_layout.addWidget(self.summary_table, 6)
        summary_layout.addWidget(QLabel("Key Observations"))
        summary_layout.addWidget(self.diagnostics_summary, 1)
        summary_layout.addWidget(QLabel("Diagnostics Guide"))
        summary_layout.addWidget(self.diagnostics_guide, 1)

        charts_panel = QWidget()
        charts_layout = QVBoxLayout(charts_panel)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.addWidget(QLabel("PnL Plot"))
        charts_layout.addWidget(self.plot_label, 1)
        charts_layout.addWidget(QLabel("Correlation Heatmap"))
        charts_layout.addWidget(self.heatmap_label, 1)

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(summary_panel)
        right_splitter.addWidget(charts_panel)
        right_splitter.setStretchFactor(0, 5)
        right_splitter.setStretchFactor(1, 4)
        right_splitter.setSizes([520, 400])

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(right_splitter)

        splitter = QSplitter()
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([320, 1180])

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(splitter)
        self.setCentralWidget(container)
        self.update_raw_source_info(self.raw_source_input.currentText())

    def append_log(self, msg: str) -> None:
        self.logs.appendPlainText(msg)

    def update_raw_source_info(self, source: str) -> None:
        source_key = source.strip().lower()
        info = DATA_SOURCE_INFO.get(source_key, {})
        self.raw_source_info_label.setText(
            "\n".join(
                [
                    f"Type: {info.get('type', 'Unknown')}",
                    f"Expected history depth: {info.get('expected_history_depth', 'Unknown')}",
                    f"Warning: {info.get('warning', '')}",
                ]
            )
        )

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
        source = self.raw_source_input.currentText().strip().lower()
        api_key = self.alphavantage_key_input.text().strip()

        self.status_label.setText(f"Status: Downloading raw {source} data...")
        self.append_log(f"Starting raw FX download from {source}...")

        self.task_thread = QThread()
        self.task_worker = RawDataWorker(
            self.raw_data_path,
            source=source,
            alphavantage_api_key=api_key,
        )
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
        self.append_log("Building feature CSV from raw FX data...")

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
            model_fit_windows = [
                int(item.strip())
                for item in self.model_fit_windows_input.text().split(",")
                if item.strip()
            ]
            if not model_fit_windows:
                raise ValueError("MODEL_FIT_WINDOWS must contain at least one fit window")
            if any(window <= 0 for window in model_fit_windows):
                raise ValueError("MODEL_FIT_WINDOWS must contain only positive integers")
            step_days = self._read_int(self.step_input, "STEP_DAYS")
            rebalance_days = self._read_int(self.rebalance_input, "REBALANCE_DAYS")
            horizon = self._read_int(self.horizon_input, "HORIZON")
            transaction_loss_pct = self._read_nonnegative_float(
                self.transaction_loss_input,
                "TRANSACTION_LOSS_PCT",
            )
            trading_days_per_year = self._read_int(
                self.trading_days_input,
                "TRADING_DAYS_PER_YEAR",
            )
            holdout_days = self._read_int(self.holdout_days_input, "HOLDOUT_DAYS")
            live_model = self.live_model_input.currentText().strip()
            specialist_weighting_mode = self.specialist_weighting_mode_input.currentText().strip()
            specialist_weight_lookback_days = self._read_int(
                self.specialist_lookback_input,
                "SPECIALIST_WEIGHT_LOOKBACK_DAYS",
            )
            specialist_min_model_hold_days = self._read_int(
                self.specialist_min_hold_input,
                "SPECIALIST_MIN_MODEL_HOLD_DAYS",
            )
            specialist_switch_margin_min_avg_ev = float(self.specialist_switch_margin_input.text().strip())
            specialist_switch_require_positive_ev = (
                self.specialist_positive_ev_input.currentText().strip().lower() == "true"
            )
            retrain_deterioration_lookback_days = self._read_int(
                self.retrain_lookback_input,
                "RETRAIN_DETERIORATION_LOOKBACK_DAYS",
            )
            retrain_deterioration_min_win_rate = self._read_nonnegative_float(
                self.retrain_win_rate_input,
                "RETRAIN_DETERIORATION_MIN_WIN_RATE",
            )
            retrain_deterioration_max_avg_ev = float(self.retrain_avg_ev_input.text().strip())
            specialist_ensemble_members = [
                item.strip()
                for item in self.specialist_members_input.text().split(",")
                if item.strip()
            ]
            model_router_candidates = [
                item.strip()
                for item in self.model_router_candidates_input.text().split(",")
                if item.strip()
            ]
            if len(specialist_ensemble_members) < 2:
                raise ValueError("SPECIALIST_ENSEMBLE_MEMBERS must contain at least two model names")
            if not model_router_candidates:
                raise ValueError("MODEL_ROUTER_CANDIDATES must contain at least one model name")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Invalid Input", str(exc))
            return

        self.run_btn.setEnabled(False)
        self.download_raw_btn.setEnabled(False)
        self.build_features_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Running...")
        self.logs.clear()
        self.diagnostics_summary.clear()
        self.diagnostics_guide.setPlainText(
            diagnostics_guide_text(
                specialist_ensemble_members,
                model_router_candidates,
                model_fit_windows,
                specialist_weight_lookback_days,
                rebalance_days,
                specialist_weighting_mode,
                specialist_min_model_hold_days,
                specialist_switch_margin_min_avg_ev,
                specialist_switch_require_positive_ev,
                retrain_deterioration_lookback_days,
                retrain_deterioration_min_win_rate,
                retrain_deterioration_max_avg_ev,
            )
        )
        self.append_log("Starting pipeline...")

        self.pipeline_thread = QThread()
        self.pipeline_worker = PipelineWorker(
            csv_path=csv_path,
            fit_days=fit_days,
            model_fit_windows=model_fit_windows,
            step_days=step_days,
            rebalance_days=rebalance_days,
            horizon=horizon,
            transaction_loss_pct=transaction_loss_pct,
            trading_days_per_year=trading_days_per_year,
            holdout_days=holdout_days,
            live_model=live_model,
            specialist_weighting_mode=specialist_weighting_mode,
            specialist_ensemble_members=specialist_ensemble_members,
            model_router_candidates=model_router_candidates,
            specialist_weight_lookback_days=specialist_weight_lookback_days,
            specialist_min_model_hold_days=specialist_min_model_hold_days,
            specialist_switch_margin_min_avg_ev=specialist_switch_margin_min_avg_ev,
            specialist_switch_require_positive_ev=specialist_switch_require_positive_ev,
            retrain_deterioration_lookback_days=retrain_deterioration_lookback_days,
            retrain_deterioration_min_win_rate=retrain_deterioration_min_win_rate,
            retrain_deterioration_max_avg_ev=retrain_deterioration_max_avg_ev,
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

    def on_pipeline_completed(
        self,
        stats_df: pd.DataFrame,
        plot_path: str,
        heatmap_path: str,
        diagnostics_summary: str,
    ) -> None:
        self.status_label.setText("Status: Done")
        self.populate_summary_table(stats_df)
        self.diagnostics_summary.setPlainText(diagnostics_summary)
        self.load_plot(plot_path)
        self.load_heatmap(heatmap_path)
        self.append_log("Pipeline completed successfully.")

    def on_pipeline_failed(self, error_msg: str) -> None:
        self.status_label.setText("Status: Error")
        self.append_log(f"Error: {error_msg}")
        QMessageBox.critical(self, "Pipeline Error", error_msg)

    def on_raw_download_completed(self, raw_csv_path: str) -> None:
        self.progress_bar.setValue(100)
        self.raw_data_path_label.setText(raw_csv_path)
        self.status_label.setText("Status: Raw FX data downloaded")
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
        self.summary_table.setSortingEnabled(False)
        self.summary_table.setRowCount(len(stats_df))
        for i, row in stats_df.iterrows():
            display_name = row["display_name"] if "display_name" in stats_df.columns else row["model"]
            self.summary_table.setItem(i, 0, QTableWidgetItem(str(display_name)))
            self.summary_table.setItem(i, 1, QTableWidgetItem(f"{row['Annualized Return'] * 100:.1f}%"))
            self.summary_table.setItem(i, 2, QTableWidgetItem(f"{row['Annualized Vol'] * 100:.1f}%"))
            self.summary_table.setItem(i, 3, QTableWidgetItem(f"{row['Sharpe']:.6f}"))
            self.summary_table.setItem(i, 4, QTableWidgetItem(f"{row['Cumulative Return'] * 100:.1f}%"))
        self.summary_table.setSortingEnabled(True)
        self.summary_table.sortItems(3, Qt.DescendingOrder)

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

    def load_heatmap(self, heatmap_path: str) -> None:
        pixmap = QPixmap(heatmap_path)
        if pixmap.isNull():
            self.heatmap_label.setText("Failed to load model_correlation_heatmap.png")
            return
        scaled = pixmap.scaled(
            self.heatmap_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.heatmap_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        current = self.output_dir / "pnl_curves.png"
        if current.exists() and self.plot_label.pixmap() is not None:
            self.load_plot(str(current))
        heatmap = self.output_dir / "model_correlation_heatmap.png"
        if heatmap.exists() and self.heatmap_label.pixmap() is not None:
            self.load_heatmap(str(heatmap))

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
