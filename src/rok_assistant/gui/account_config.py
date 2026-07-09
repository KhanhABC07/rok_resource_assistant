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

from rok_assistant.account_config import (
    AccountConfigError,
    AccountConfigInput,
    AccountConfigService,
    MAX_CONFIGURED_ACCOUNTS,
)
from rok_assistant.app import AppContext
from rok_assistant.security import SecretStoreError, redacted_exception_message
from rok_assistant.gui.widgets import set_table_item


class AccountConfigWidget(QWidget):
    def __init__(self, context: AppContext):
        super().__init__()
        self.context = context
        self.service = AccountConfigService(context.accounts, context.secret_store)
        self.selected_id: int | None = None

        self.account_name_input = QLineEdit()
        self.display_name_input = QLineEdit()
        self.provider_input = QComboBox()
        self.provider_input.setEditable(True)
        self.provider_input.addItems(["email", "facebook", "google", "apple"])
        self.external_id_input = QLineEdit()
        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.enabled_input = QCheckBox("Enabled")
        self.enabled_input.setChecked(True)

        form = QFormLayout()
        form.addRow("Account Name", self.account_name_input)
        form.addRow("Display Name", self.display_name_input)
        form.addRow("Login Provider", self.provider_input)
        form.addRow("Provider Account ID", self.external_id_input)
        form.addRow("Username", self.username_input)
        form.addRow("Password", self.password_input)
        form.addRow("Token", self.token_input)
        form.addRow("", self.enabled_input)

        self.save_button = QPushButton("Save")
        self.clear_button = QPushButton("Clear")
        self.delete_button = QPushButton("Delete")
        buttons = QHBoxLayout()
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.clear_button)
        buttons.addWidget(self.delete_button)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Account", "Display", "Provider", "Provider ID", "Enabled", "Credential"]
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

    def refresh(self) -> None:
        accounts = self.service.list_accounts()
        self.table.setRowCount(len(accounts))
        for row, account in enumerate(accounts):
            set_table_item(self.table, row, 0, account.id)
            set_table_item(self.table, row, 1, account.account_name)
            set_table_item(self.table, row, 2, account.display_name)
            set_table_item(self.table, row, 3, account.provider)
            set_table_item(self.table, row, 4, account.external_id)
            set_table_item(self.table, row, 5, "Yes" if account.enabled else "No")
            set_table_item(self.table, row, 6, "Stored" if account.secret_ref else "Missing")

    def load_selected(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if item is None:
            return
        account = self.service.get_account(int(item.text()))
        if account is None:
            return
        self.selected_id = account.id
        self.account_name_input.setText(account.account_name)
        self.display_name_input.setText(account.display_name)
        provider_index = self.provider_input.findText(account.provider)
        if provider_index >= 0:
            self.provider_input.setCurrentIndex(provider_index)
        else:
            self.provider_input.setEditText(account.provider)
        self.external_id_input.setText(account.external_id)
        self.username_input.clear()
        self.password_input.clear()
        self.token_input.clear()
        self.enabled_input.setChecked(account.enabled)

    def save(self) -> None:
        try:
            self.service.save_account(
                AccountConfigInput(
                    account_id=self.selected_id,
                    account_name=self.account_name_input.text(),
                    display_name=self.display_name_input.text(),
                    provider=self.provider_input.currentText(),
                    external_id=self.external_id_input.text(),
                    username=self.username_input.text(),
                    password=self.password_input.text(),
                    token=self.token_input.text(),
                    enabled=self.enabled_input.isChecked(),
                )
            )
        except AccountConfigError as exc:
            QMessageBox.warning(self, "Validation", str(exc))
            return
        except SecretStoreError as exc:
            QMessageBox.warning(self, "Credential Store", redacted_exception_message(exc))
            return

        self.clear_form()
        self.refresh()
        QMessageBox.information(self, "Saved", "Account configuration saved.")

    def delete(self) -> None:
        if self.selected_id is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete Account",
            "Delete this account configuration and its stored credential?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.service.delete_account(self.selected_id)
        except SecretStoreError as exc:
            QMessageBox.warning(self, "Credential Store", redacted_exception_message(exc))
            return
        self.clear_form()
        self.refresh()

    def clear_form(self) -> None:
        self.selected_id = None
        self.account_name_input.clear()
        self.display_name_input.clear()
        self.provider_input.setCurrentIndex(0)
        self.external_id_input.clear()
        self.username_input.clear()
        self.password_input.clear()
        self.token_input.clear()
        self.enabled_input.setChecked(True)

    @property
    def max_supported_accounts(self) -> int:
        return MAX_CONFIGURED_ACCOUNTS
