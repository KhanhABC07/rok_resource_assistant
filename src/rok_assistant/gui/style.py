APP_STYLE = """
QMainWindow, QWidget {
    background: #f6f7f9;
    color: #1d2433;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #d8dde6;
    background: #ffffff;
}
QTabBar::tab {
    background: #e9edf3;
    border: 1px solid #d8dde6;
    border-bottom: none;
    padding: 8px 14px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #ffffff;
}
QPushButton {
    background: #2457a6;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 7px 12px;
}
QPushButton:hover {
    background: #1f4d93;
}
QPushButton:disabled {
    background: #9aa5b5;
}
QLineEdit, QSpinBox, QComboBox, QTableWidget, QPlainTextEdit {
    background: #ffffff;
    border: 1px solid #cfd6e2;
    border-radius: 3px;
    padding: 4px;
}
QHeaderView::section {
    background: #eef1f5;
    border: none;
    border-right: 1px solid #d8dde6;
    padding: 6px;
    font-weight: 600;
}
"""
