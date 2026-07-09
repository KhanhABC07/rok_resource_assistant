from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from rok_assistant.gui.style import (
    TOKENS,
    button_variant_qss,
    empty_table_state_qss,
    metric_card_qss,
    section_card_qss,
    status_badge_qss,
    status_color,
    status_text_qss,
)


def set_table_item(
    table: QTableWidget,
    row: int,
    column: int,
    value: object,
) -> QTableWidgetItem:
    text = "" if value is None else str(value)
    item = QTableWidgetItem(text)
    item.setToolTip(text)
    table.setItem(row, column, item)
    return item


def configure_table(
    table: QTableWidget,
    *,
    stretch_last: bool = True,
    selection_mode: QAbstractItemView.SelectionMode | None = (
        QAbstractItemView.SelectionMode.SingleSelection
    ),
    read_only: bool = True,
    sorting: bool = False,
    min_height: int | None = None,
) -> None:
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.setWordWrap(False)
    table.setSortingEnabled(sorting)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    if selection_mode is not None:
        table.setSelectionMode(selection_mode)
    if read_only:
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(32)
    table.horizontalHeader().setHighlightSections(False)
    table.horizontalHeader().setStretchLastSection(stretch_last)
    table.horizontalHeader().setMinimumSectionSize(64)
    if min_height is not None:
        table.setMinimumHeight(min_height)


def set_empty_table_state(
    table: QTableWidget,
    message: str,
    *,
    visible: bool | None = None,
) -> None:
    label = _empty_state_label(table)
    label.setText(message)
    label.setGeometry(table.viewport().rect())
    label.setVisible(table.rowCount() == 0 if visible is None else visible)
    if label.isVisible():
        label.raise_()


def apply_button_variant(button: QPushButton, variant: str) -> None:
    button.setProperty("variant", variant)
    button.setStyleSheet(button_variant_qss(variant))
    button.style().unpolish(button)
    button.style().polish(button)


def apply_status_text_style(label: QLabel, kind: str) -> None:
    label.setStyleSheet(status_text_qss(kind))


def status_qcolor(kind: str) -> QColor:
    return QColor(status_color(kind))


class StatusBadgeLabel(QLabel):
    def __init__(self, text: str = "", kind: str = "neutral") -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(24)
        self.set_status(text, kind)

    def set_status(self, text: str, kind: str = "neutral") -> None:
        self.setText(text)
        self.setProperty("status", kind)
        self.setStyleSheet(status_badge_qss(kind))


class SectionCardWidget(QFrame):
    def __init__(self, title: str, subtitle: str | None = None) -> None:
        super().__init__()
        self.setObjectName("sectionCard")
        self.setStyleSheet(section_card_qss())
        self.content_layout = QVBoxLayout(self)
        self.content_layout.setContentsMargins(
            TOKENS.spacing.lg,
            TOKENS.spacing.lg,
            TOKENS.spacing.lg,
            TOKENS.spacing.lg,
        )
        self.content_layout.setSpacing(TOKENS.spacing.md)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("sectionTitle")
        self.content_layout.addWidget(self.title_label)

        self.subtitle_label: QLabel | None = None
        if subtitle:
            self.subtitle_label = QLabel(subtitle)
            self.subtitle_label.setObjectName("sectionSubtitle")
            self.subtitle_label.setWordWrap(True)
            self.content_layout.addWidget(self.subtitle_label)

    def addWidget(self, widget, stretch: int = 0) -> None:  # type: ignore[no-untyped-def]
        self.content_layout.addWidget(widget, stretch)

    def addLayout(self, layout, stretch: int = 0) -> None:  # type: ignore[no-untyped-def]
        self.content_layout.addLayout(layout, stretch)


class MetricCardLabel(QLabel):
    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.value = "-"
        self.setMinimumHeight(72)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setStyleSheet(metric_card_qss())
        self.set_value("-")

    def set_value(self, value: object) -> None:
        self.value = str(value)
        self.setText(
            "<span style='font-size:20px; font-weight:700;'>"
            f"{self.value}"
            "</span><br>"
            f"<span style='color:{TOKENS.palette.text_muted}; font-size:12px;'>"
            f"{self.title}"
            "</span>"
        )


class MetricLabel(MetricCardLabel):
    """Backward-compatible name for the dashboard metric card."""


class _EmptyTableStateLabel(QLabel):
    def __init__(self, table: QTableWidget) -> None:
        super().__init__(table.viewport())
        self.table = table
        self.setObjectName("emptyTableState")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet(empty_table_state_qss())
        table.viewport().installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        if watched is self.table.viewport() and event.type() == QEvent.Type.Resize:
            self.setGeometry(self.table.viewport().rect())
        return super().eventFilter(watched, event)


def _empty_state_label(table: QTableWidget) -> _EmptyTableStateLabel:
    label = table.viewport().findChild(_EmptyTableStateLabel, "emptyTableState")
    if label is None:
        label = _EmptyTableStateLabel(table)
    return label


StatusBadge = StatusBadgeLabel
SectionCard = SectionCardWidget
MetricCard = MetricCardLabel
