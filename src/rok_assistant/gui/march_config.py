from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from rok_assistant.app import AppContext
from rok_assistant.db.models import March
from rok_assistant.gui.widgets import set_table_item


class MarchConfigWidget(QWidget):
    def __init__(self, context: AppContext):
        super().__init__()
        self.context = context

        self.character_input = QComboBox()
        self.refresh_button = QPushButton("Refresh Characters")
        self.save_button = QPushButton("Save Marches")
        self.schedule_button = QPushButton("Create Tasks")

        top = QHBoxLayout()
        top.addWidget(self.character_input, 1)
        top.addWidget(self.refresh_button)
        top.addWidget(self.save_button)
        top.addWidget(self.schedule_button)

        self.table = QTableWidget(5, 4)
        self.table.setHorizontalHeaderLabels(
            [
                "Slot",
                "Status",
                "Next Action Time",
                "Expected Return Time",
            ]
        )
        self.table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.table)

        self.refresh_button.clicked.connect(self.refresh_characters)
        self.save_button.clicked.connect(self.save)
        self.schedule_button.clicked.connect(self.create_tasks)
        self.character_input.currentIndexChanged.connect(self.load_marches)
        self.refresh_characters()

    def refresh_characters(self) -> None:
        selected = self.character_input.currentData()
        self.character_input.clear()
        for character in self.context.characters.list_all():
            self.character_input.addItem(
                f"{character.instance_name} / {character.name}",
                character.id,
            )
        if selected is not None:
            index = self.character_input.findData(selected)
            if index >= 0:
                self.character_input.setCurrentIndex(index)
        self.load_marches()

    def load_marches(self) -> None:
        character_id = self.character_input.currentData()
        for row in range(5):
            set_table_item(self.table, row, 0, row + 1)
            status = QLineEdit("idle")
            next_action = QLineEdit()
            expected_return = QLineEdit()
            self.table.setCellWidget(row, 1, status)
            self.table.setCellWidget(row, 2, next_action)
            self.table.setCellWidget(row, 3, expected_return)

        if character_id is None:
            return

        marches = self.context.marches.list_for_character(int(character_id))
        for row, march in enumerate(marches):
            status = self.table.cellWidget(row, 1)
            next_action = self.table.cellWidget(row, 2)
            expected_return = self.table.cellWidget(row, 3)
            if isinstance(status, QLineEdit):
                status.setText(march.status)
            if isinstance(next_action, QLineEdit):
                next_action.setText(march.next_action_time or "")
            if isinstance(expected_return, QLineEdit):
                expected_return.setText(march.expected_return_time or "")

    def save(self) -> None:
        character_id = self.character_input.currentData()
        if character_id is None:
            QMessageBox.warning(self, "Validation", "Create a character first.")
            return

        for row in range(5):
            status = self.table.cellWidget(row, 1)
            next_action = self.table.cellWidget(row, 2)
            expected_return = self.table.cellWidget(row, 3)
            self.context.marches.save(
                March(
                    character_id=int(character_id),
                    march_slot=row + 1,
                    status=status.text() if isinstance(status, QLineEdit) else "idle",
                    next_action_time=next_action.text().strip()
                    if isinstance(next_action, QLineEdit) and next_action.text().strip()
                    else None,
                    expected_return_time=expected_return.text().strip()
                    if isinstance(expected_return, QLineEdit) and expected_return.text().strip()
                    else None,
                )
            )
        QMessageBox.information(self, "Saved", "March configuration saved.")

    def create_tasks(self) -> None:
        self.save()
        created = self.context.schedule_enabled_work()
        QMessageBox.information(self, "Tasks Created", f"Created {created} task(s).")
