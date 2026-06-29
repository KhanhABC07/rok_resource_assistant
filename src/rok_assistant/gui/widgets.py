from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem


def set_table_item(table: QTableWidget, row: int, column: int, value: object) -> None:
    item = QTableWidgetItem("" if value is None else str(value))
    table.setItem(row, column, item)


class MetricLabel(QLabel):
    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.value = "-"
        self.setMinimumHeight(58)
        self.setStyleSheet(
            """
            QLabel {
                background: #ffffff;
                border: 1px solid #d8dde6;
                border-radius: 6px;
                padding: 8px 10px;
            }
            """
        )
        self.set_value("-")

    def set_value(self, value: object) -> None:
        self.value = str(value)
        self.setText(f"<b>{self.value}</b><br><span style='color:#5c6675'>{self.title}</span>")
