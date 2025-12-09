# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QListWidget, QDialogButtonBox
)

import matplotlib.pyplot as plt


class RmsPlotDialog(QDialog):
    def __init__(self, devices_options: dict[str, str], results_table, uid2idx: dict[int, int], vars_glob_name2uid,
                 parent=None):
        """
        :param devices: dict {device: [Var]}
        """
        super().__init__(parent)
        self.setWindowTitle("Plot Variables")
        self.uid2idx = uid2idx
        self.vars_glob_name2uid = vars_glob_name2uid
        self.devices = devices_options
        self.results_table = results_table
        self.selected_vars = []

        layout = QVBoxLayout(self)

        # --- Device selector ---
        dev_layout = QHBoxLayout()
        dev_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(list(devices_options.keys()))
        self.device_combo.currentIndexChanged.connect(self.update_variables)
        dev_layout.addWidget(self.device_combo)
        layout.addLayout(dev_layout)

        # --- Variable selector ---
        var_layout = QHBoxLayout()
        var_layout.addWidget(QLabel("Variable:"))
        self.var_combo = QComboBox()
        var_layout.addWidget(self.var_combo)
        layout.addLayout(var_layout)

        # --- Add button ---
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self.add_variable)
        layout.addWidget(add_btn)

        # --- List of selected vars ---
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        # --- Buttons ---
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.plot_selected)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.update_variables(0)

    def update_variables(self, index):
        device = self.device_combo.currentText()
        self.var_combo.clear()
        self.var_combo.addItems(self.devices[device])

    def add_variable(self):
        var = self.var_combo.currentText()
        item_str = var
        if item_str not in self.selected_vars:
            self.selected_vars.append(self.vars_glob_name2uid[item_str])
            self.list_widget.addItem(item_str)

    def plot_selected(self):
        selected_col_idx = [self.uid2idx[uid] for uid in self.selected_vars]
        self.results_table.plot(selected_col_idx=selected_col_idx)
        plt.show()
