from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from rok_assistant.app import AppContext
from rok_assistant.db.models import Character
from rok_assistant.gui.widgets import set_table_item


class CharacterManagerWidget(QWidget):
    def __init__(self, context: AppContext):
        super().__init__()
        self.context = context
        self.selected_id: int | None = None

        self.name_input = QLineEdit()
        self.account_input = QLineEdit()
        self.instance_input = QComboBox()
        self.enabled_input = QCheckBox("Enabled")
        self.help_input = QCheckBox("Alliance Help")
        self.donate_input = QCheckBox("Alliance Donate")
        self.gift_input = QCheckBox("Gift Collection")
        for checkbox in (
            self.enabled_input,
            self.help_input,
            self.donate_input,
            self.gift_input,
        ):
            checkbox.setChecked(True)

        form = QFormLayout()
        form.addRow("Character Name", self.name_input)
        form.addRow("Account Name", self.account_input)
        form.addRow("Assigned Instance", self.instance_input)
        form.addRow("", self.enabled_input)
        form.addRow("", self.help_input)
        form.addRow("", self.donate_input)
        form.addRow("", self.gift_input)

        self.save_button = QPushButton("Save")
        self.clear_button = QPushButton("Clear")
        self.delete_button = QPushButton("Delete")
        buttons = QHBoxLayout()
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.clear_button)
        buttons.addWidget(self.delete_button)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Instance", "Account", "Character", "Enabled", "Help", "Donate", "Gifts"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.table)

        self.save_button.clicked.connect(self.save)
        self.clear_button.clicked.connect(self.clear_form)
        self.delete_button.clicked.connect(self.delete)
        self.table.cellClicked.connect(self.load_selected)
        self.refresh()

    def refresh_instances(self) -> None:
        selected = self.instance_input.currentData()
        self.instance_input.clear()
        for instance in self.context.instances.list_all():
            self.instance_input.addItem(instance.name, instance.id)
        if selected is not None:
            index = self.instance_input.findData(selected)
            if index >= 0:
                self.instance_input.setCurrentIndex(index)

    def refresh(self) -> None:
        self.refresh_instances()
        rows = self.context.characters.list_all()
        self.table.setRowCount(len(rows))
        for row, character in enumerate(rows):
            set_table_item(self.table, row, 0, character.id)
            set_table_item(self.table, row, 1, character.instance_name)
            set_table_item(self.table, row, 2, character.account_name)
            set_table_item(self.table, row, 3, character.name)
            set_table_item(self.table, row, 4, "Yes" if character.enabled else "No")
            set_table_item(self.table, row, 5, "Yes" if character.alliance_help_enabled else "No")
            set_table_item(
                self.table, row, 6, "Yes" if character.alliance_donate_enabled else "No"
            )
            set_table_item(
                self.table, row, 7, "Yes" if character.gift_collection_enabled else "No"
            )

    def load_selected(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if item is None:
            return
        character = self.context.characters.get(int(item.text()))
        if character is None:
            return
        self.selected_id = character.id
        self.name_input.setText(character.name)
        self.account_input.setText(character.account_name)
        index = self.instance_input.findData(character.instance_id)
        if index >= 0:
            self.instance_input.setCurrentIndex(index)
        self.enabled_input.setChecked(character.enabled)
        self.help_input.setChecked(character.alliance_help_enabled)
        self.donate_input.setChecked(character.alliance_donate_enabled)
        self.gift_input.setChecked(character.gift_collection_enabled)

    def save(self) -> None:
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "Validation", "Character name is required.")
            return
        instance_id = self.instance_input.currentData()
        if instance_id is None:
            QMessageBox.warning(self, "Validation", "Create an instance first.")
            return
        self.context.characters.save(
            Character(
                id=self.selected_id,
                name=self.name_input.text(),
                account_name=self.account_input.text(),
                instance_id=int(instance_id),
                enabled=self.enabled_input.isChecked(),
                alliance_help_enabled=self.help_input.isChecked(),
                alliance_donate_enabled=self.donate_input.isChecked(),
                gift_collection_enabled=self.gift_input.isChecked(),
            )
        )
        self.clear_form()
        self.refresh()

    def delete(self) -> None:
        if self.selected_id is None:
            return
        self.context.characters.delete(self.selected_id)
        self.clear_form()
        self.refresh()

    def clear_form(self) -> None:
        self.selected_id = None
        self.name_input.clear()
        self.account_input.clear()
        self.enabled_input.setChecked(True)
        self.help_input.setChecked(True)
        self.donate_input.setChecked(True)
        self.gift_input.setChecked(True)
