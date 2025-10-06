from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView, QStyledItemDelegate, QSpinBox
)
import sys


class SpinBoxDelegate(QStyledItemDelegate):
    """Delegate for integer-only column (Calle)."""
    def __init__(self, parent=None, minimum=1, maximum=9999):
        super().__init__(parent)
        self.minimum = minimum
        self.maximum = maximum

    def createEditor(self, parent, option, index):
        spinbox = QSpinBox(parent)
        spinbox.setMinimum(self.minimum)
        spinbox.setMaximum(self.maximum)
        return spinbox

    def setEditorData(self, editor, index):
        value = index.model().data(index, 0)
        if value and value.isdigit():
            editor.setValue(int(value))
        else:
            editor.setValue(self.minimum)

    def setModelData(self, editor, model, index):
        model.setData(index, str(editor.value()))


class ComboBoxDelegate(QStyledItemDelegate):
    """Delegate for restricted values (Posición, Barra)."""
    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.items = items

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(self.items)
        return combo

    def setEditorData(self, editor, index):
        value = index.model().data(index, 0)
        if value in self.items:
            editor.setCurrentText(value)
        else:
            editor.setCurrentIndex(0)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText())


class VoltageLevelConversionWizard(QWidget):
    """
    Voltage level conversion wizzard
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Substation Wizard")

        # Main layout
        main_layout = QVBoxLayout(self)

        # --- Combobox for type of park ---
        self.combo = QComboBox()
        self.combo.addItems(["Simple barra", "Doble barra", "Interruptor y medio", "Anillo"])
        main_layout.addWidget(self.combo)

        # --- Checkboxes ---
        self.checkbox1 = QCheckBox("Usar interruptores")
        self.checkbox1.setChecked(True)
        self.checkbox2 = QCheckBox("Mantener ratio original")
        self.checkbox3 = QCheckBox("Activar barra de transferencia")
        self.checkbox4 = QCheckBox("Otra opción...")

        for cb in [self.checkbox1, self.checkbox2, self.checkbox3, self.checkbox4]:
            main_layout.addWidget(cb)

        # --- Table for positions ---
        self.table = QTableWidget(3, 3)
        self.table.setHorizontalHeaderLabels(["Posición", "Calle", "Barra"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Allowed lists
        self.posiciones_list = ["Rama 1", "Rama 2", "Gen 1", "Gen 2", "Trafo 1"]
        self.barras_list = ["JPB1", "JPB2", "JPB3"]

        # Apply delegates
        self.table.setItemDelegateForColumn(0, ComboBoxDelegate(self.posiciones_list, self.table))  # Posición
        self.table.setItemDelegateForColumn(1, SpinBoxDelegate(self.table, minimum=1, maximum=1000))  # Calle
        self.table.setItemDelegateForColumn(2, ComboBoxDelegate(self.barras_list, self.table))  # Barra

        # Example data
        data = [
            ("Rama 1", "1", "JPB1"),
            ("Rama 2", "1", "JPB2"),
            ("Gen 1", "2", "JPB1"),
        ]

        for row, (pos, calle, barra) in enumerate(data):
            self.table.setItem(row, 0, QTableWidgetItem(pos))
            self.table.setItem(row, 1, QTableWidgetItem(calle))
            self.table.setItem(row, 2, QTableWidgetItem(barra))

        main_layout.addWidget(self.table)

        # --- Bottom buttons layout ---
        bottom_layout = QHBoxLayout()

        # Left side: add/remove buttons
        self.add_button = QPushButton("+")
        self.add_button.setFixedWidth(40)
        self.add_button.clicked.connect(self.add_row)

        self.remove_button = QPushButton("-")
        self.remove_button.setFixedWidth(40)
        self.remove_button.clicked.connect(self.remove_row)

        bottom_layout.addWidget(self.add_button)
        bottom_layout.addWidget(self.remove_button)
        bottom_layout.addStretch()  # spacer pushes "Do it" button to the right

        # Right side: Do it button
        self.do_button = QPushButton("Do it")
        bottom_layout.addWidget(self.do_button)

        main_layout.addLayout(bottom_layout)

    def add_row(self):
        """Insert an empty row with defaults."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(self.posiciones_list[0]))
        self.table.setItem(row, 1, QTableWidgetItem("1"))
        self.table.setItem(row, 2, QTableWidgetItem(self.barras_list[0]))

    def remove_row(self):
        """Remove the currently selected row."""
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoltageLevelConversionWizard()
    window.resize(600, 500)
    window.show()
    sys.exit(app.exec())
