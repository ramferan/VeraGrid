# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from enum import Enum, auto
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import sys
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QGraphicsScene, QGraphicsView, QGraphicsItem,
                               QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsTextItem, QMenu, QGraphicsPathItem,
                               QDialog, QVBoxLayout, QComboBox, QDialogButtonBox, QSplitter, QLabel, QDoubleSpinBox,
                               QListView, QAbstractItemView, QPushButton, QListWidget, QInputDialog)
from PySide6.QtGui import (QPen, QBrush, QPainterPath, QAction, QPainter, QIcon, QStandardItemModel, QStandardItem,
                           QPixmap, QDropEvent, QDragEnterEvent, QDragMoveEvent)
from PySide6.QtCore import Qt, QPointF, QByteArray, QDataStream, QIODevice, QModelIndex, QMimeData
from VeraGridEngine.Utils.Symbolic.block import (
    Block,
    adder,
    constant,
    variable,
    gain,
    integrator,
    generic
)
from VeraGridEngine.Utils.Symbolic.symbolic import Var, make_symbolic, symbolic_to_string
from VeraGridEngine.Devices.Dynamic.dynamic_model_host import BlockDiagram, DynamicModelHost


def change_font_size(obj, font_size: int):
    """

    :param obj:
    :param font_size:
    :return:
    """
    font1 = obj.font()
    font1.setPointSize(font_size)
    obj.setFont(font1)


@dataclass
class BlockBridge:
    gui: "BlockItem"  # visual node
    outs: List[Var]  # exactly len(gui.outputs)
    ins: List[Var]  # exactly len(gui.inputs) – placeholders
    api_blocks: List[Block]  # usually length 1, but e.g. PI returns 4


class BlockType(Enum):
    GAIN = auto()
    SUM = auto()
    INTEGRATOR = auto()
    DERIVATIVE = auto()
    PRODUCT = auto()
    DIVIDE = auto()
    SQRT = auto()
    SQUARE = auto()
    ABS = auto()
    MIN = auto()
    MAX = auto()
    STEP = auto()
    CONSTANT = auto()
    VARIABLE = auto()
    SATURATION = auto()
    RELATIONAL = auto()
    LOGICAL = auto()
    SOURCE = auto()
    DRAIN = auto()
    GENERIC = auto()


class BlockTypeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Block Type")
        self.layout = QVBoxLayout(self)

        self.combo = QComboBox(self)
        for bt in BlockType:
            self.combo.addItem(bt.name, bt)
        self.layout.addWidget(self.combo)

        # 👇 Extra field for constants
        self.value_label = QLabel("Constant value:", self)
        self.value_spin = QDoubleSpinBox(self)
        self.value_spin.setRange(-1e6, 1e6)
        self.value_spin.setValue(0.0)
        self.layout.addWidget(self.value_label)
        self.layout.addWidget(self.value_spin)

        # Initially hidden
        self.value_label.hide()
        self.value_spin.hide()

        self.combo.currentIndexChanged.connect(self._on_block_changed)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

    def _on_block_changed(self, index):
        block_type = self.combo.itemData(index)
        if block_type == BlockType.CONSTANT:
            self.value_label.show()
            self.value_spin.show()
        else:
            self.value_label.hide()
            self.value_spin.hide()

    def selected_block_type(self) -> BlockType:
        return self.combo.currentData()

    def constant_value(self) -> float:
        return self.value_spin.value()


class PortItem(QGraphicsEllipseItem):
    """
    Port of a block
    """

    def __init__(self,
                 subsystem: Union[BlockItem, ModelHostItem],
                 is_input: bool,
                 index: int,  # number of inputs
                 total: int,
                 radius=6):
        """

        :param block:
        :param is_input:
        :param index:
        :param total:
        :param radius:
        """
        super().__init__(-radius, -radius, 2 * radius, 2 * radius, subsystem)
        self.setBrush(QBrush(Qt.GlobalColor.blue if is_input else Qt.GlobalColor.green))
        self.setPen(QPen(Qt.GlobalColor.black))
        self.setZValue(1)
        self.setAcceptHoverEvents(True)
        self.subsystem = subsystem
        self.is_input = is_input
        self.connection = None
        self.index = index
        self.total = total

        spacing = subsystem.rect().height() / (total + 1)
        y = spacing * (index + 1)
        x = 0 if is_input else subsystem.rect().width()
        self.setPos(x, y)

    def hoverEnterEvent(self, event):
        QApplication.setOverrideCursor(Qt.CursorShape.PointingHandCursor)

    def hoverLeaveEvent(self, event):
        QApplication.restoreOverrideCursor()

    def is_connected(self):
        return self.connection is not None


class ConnectionItem(QGraphicsPathItem):
    def __init__(self, source_port, target_port):
        super().__init__()
        self.setZValue(-1)
        self.source_port = source_port
        self.target_port = target_port
        self.source_port.connection = self
        self.target_port.connection = self
        self.setPen(QPen(Qt.GlobalColor.darkBlue, 2))
        self.setAcceptHoverEvents(True)

        self.update_path()

    def update_path(self):
        start = self.source_port.scenePos()
        end = self.target_port.scenePos()
        mid_x = (start.x() + end.x()) / 2
        c1 = QPointF(mid_x, start.y())
        c2 = QPointF(mid_x, end.y())
        path = QPainterPath(start)
        path.cubicTo(c1, c2, end)
        self.setPath(path)

    def hoverEnterEvent(self, event):
        QApplication.setOverrideCursor(Qt.CursorShape.PointingHandCursor)

    def hoverLeaveEvent(self, event):
        QApplication.restoreOverrideCursor()

    def contextMenuEvent(self, event):
        menu = QMenu()
        remove_action = QAction("Remove Connection", menu)
        menu.addAction(remove_action)
        if menu.exec(event.screenPos()) == remove_action:
            self.scene().removeItem(self)
            self.source_port.connection = None
            self.target_port.connection = None


class ResizeHandle(QGraphicsRectItem):
    def __init__(self, block, size=10):
        super().__init__(0, 0, size, size, block)
        self.setBrush(QBrush(Qt.GlobalColor.darkGray))
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self.setZValue(2)
        self.block = block
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges)
        self.setAcceptHoverEvents(True)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            if not self.block._resizing_from_handle:
                return super().itemChange(change, value)

            new_pos = value  # already QPointF
            min_width, min_height = 40, 30
            new_width = max(new_pos.x(), min_width)
            new_height = max(new_pos.y(), min_height)

            self.block.resize_block(new_width, new_height)

            return QPointF(new_width, new_height)
        return super().itemChange(change, value)


class BlockItem(QGraphicsRectItem):
    def __init__(self, block_sys: Block):
        """

        :param block_sys: Block
        """
        super().__init__(0, 0, 100, 60)

        # ------------------------
        # API
        # ------------------------
        self.subsys = block_sys

        # ---------------------------
        # Graphical stuff
        # ---------------------------
        self.setBrush(Qt.GlobalColor.lightGray)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges
        )
        self.setAcceptHoverEvents(True)
        self.setAcceptHoverEvents(True)

        self.name_item = QGraphicsTextItem(self.subsys.name, self)

        self.name_item.setPos(10, 5)

        n_inputs = len(self.subsys.in_vars)
        n_outputs = len(self.subsys.out_vars)

        self.inputs = [PortItem(self, True, i, n_inputs) for i in range(n_inputs)]
        self.outputs = [PortItem(self, False, i, n_outputs) for i in range(n_outputs)]

        self.resize_handle = ResizeHandle(self)

        # ✅ Avoid triggering overridden setRect during init
        super().setRect(0, 0, 100, 60)
        self.update_ports()
        self.update_handle_position()

        self._resizing_from_handle = False


    def mouseDoubleClickEvent(self, event):
        # --- Constant editing ---
        if self.subsys.name.lower().startswith("const") or self.subsys.name == "CONSTANT":
            dlg = QDialog()
            dlg.setWindowTitle("Edit Constant Value")
            layout = QVBoxLayout(dlg)

            spin = QDoubleSpinBox(dlg)
            spin.setRange(-1e6, 1e6)
            spin.setValue(self.subsys.value if hasattr(self.subsys, "value") else 0.0)
            layout.addWidget(QLabel("Constant value:"))
            layout.addWidget(spin)

            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            layout.addWidget(buttons)
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)

            if dlg.exec() == QDialog.Accepted:
                new_val = spin.value()
                self.subsys.value = new_val
                self.name_item.setPlainText(f"Const({new_val})")
            return

        super().mouseDoubleClickEvent(event)

    def resize_block(self, width, height):
        # Update geometry safely
        self.prepareGeometryChange()
        QGraphicsRectItem.setRect(self, 0, 0, width, height)
        self.update_ports()
        self.update_handle_position()

    def update_handle_position(self):
        rect = self.rect()
        self._resizing_from_handle = False
        self.resize_handle.setPos(rect.width(), rect.height())
        self._resizing_from_handle = True

    def _set_rect_internal(self, w, h):
        QGraphicsRectItem.setRect(self, 0, 0, w, h)
        self.update_ports()
        self.update_handle_position()

    def setRect(self, x, y, w, h):
        if not getattr(self, '_suppress_resize', False):
            self._set_rect_internal(w, h)

    def update_ports(self):
        for i, port in enumerate(self.inputs):
            spacing = self.rect().height() / (len(self.inputs) + 1)
            port.setPos(0, spacing * (i + 1))
        for i, port in enumerate(self.outputs):
            spacing = self.rect().height() / (len(self.outputs) + 1)
            port.setPos(self.rect().width(), spacing * (i + 1))
        self.update_handle_position()
        # Also update connections
        for port in self.inputs + self.outputs:
            if port.connection:
                port.connection.update_path()

    def hoverEnterEvent(self, event):
        QApplication.setOverrideCursor(Qt.CursorShape.OpenHandCursor)

    def hoverLeaveEvent(self, event):
        QApplication.restoreOverrideCursor()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            for port in self.inputs + self.outputs:
                if port.connection:
                    port.connection.update_path()
        return super().itemChange(change, value)

    def contextMenuEvent(self, event):
        menu = QMenu()

        delete_action = QAction("Remove Block", menu)
        menu.addAction(delete_action)

        edit_action = None
        if self.subsys.name == "generic":
            edit_action = QAction("Edit Block", menu)
            menu.addAction(edit_action)

        chosen = menu.exec(event.screenPos())

        if chosen == delete_action:
            for port in self.inputs + self.outputs:
                if port.connection:
                    self.scene().removeItem(port.connection)
                    if port.connection.source_port:
                        port.connection.source_port.connection = None
                    if port.connection.target_port:
                        port.connection.target_port.connection = None
            self.scene().removeItem(self)

        elif chosen == edit_action:
            self.open_generic_editor()

    def open_generic_editor(self):
        dlg = QDialog()
        dlg.setWindowTitle(f"Edit Generic Block ({self.subsys.uid})")
        dlg.resize(600, 400)
        layout = QVBoxLayout(dlg)

        # Section: Algebraic Variables
        alg_section = self.create_variable_section("Algebraic Variables", self.subsys.algebraic_vars)
        layout.addLayout(alg_section)

        # Section: State Variables
        state_section = self.create_variable_section("State Variables", self.subsys.state_vars)
        layout.addLayout(state_section)

        # Section: Algebraic Equations
        alg_eq_section = self.create_equation_section("Algebraic Equations", self.subsys.algebraic_eqs)
        layout.addLayout(alg_eq_section)

        # Section: State Equations
        state_eq_section = self.create_equation_section("State Equations", self.subsys.state_eqs)
        layout.addLayout(state_eq_section)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)

        dlg.exec()

    def create_variable_section(self, title, var_list):
        layout = QVBoxLayout()

        label = QLabel(title)
        layout.addWidget(label)

        list_widget = QListWidget()
        for v in var_list:

            list_widget.addItem(v.name)
        layout.addWidget(list_widget)

        add_btn = QPushButton("+")
        layout.addWidget(add_btn)

        def add_var():
            text, ok = QInputDialog.getText(None, f"Add {title}", "Variable name:")
            if ok and text:
                var_list.append(Var(text))
                print(self.subsys.algebraic_vars)
                list_widget.addItem(text)

        add_btn.clicked.connect(add_var)

        return layout

    def create_equation_section(self, title, eq_list):
        layout = QVBoxLayout()

        label = QLabel(title)
        layout.addWidget(label)

        list_widget = QListWidget()
        for eq in eq_list:
            text = symbolic_to_string(eq)
            list_widget.addItem(text)
        layout.addWidget(list_widget)

        add_btn = QPushButton("+")
        layout.addWidget(add_btn)

        def add_eq():
            text, ok = QInputDialog.getText(None, f"Add {title}", "Equation:")
            if ok and text:
                print(text)
                sym_expr = make_symbolic(text)
                print(type(sym_expr))
                eq_list.append(sym_expr)
                print(self.subsys.algebraic_eqs)
                list_widget.addItem(text)

        add_btn.clicked.connect(add_eq)

        return layout

class ModelHostItem(QGraphicsRectItem):
    def __init__(self, model_host_sys: DynamicModelHost):
        """

        :param block_sys: Block
        """
        super().__init__(0, 0, 100, 60)

        # ------------------------
        # API
        # ------------------------
        self.model_host = model_host_sys

        # ---------------------------
        # Graphical stuff
        # ---------------------------
        self.setBrush(Qt.GlobalColor.lightGray)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges
        )
        self.setAcceptHoverEvents(True)
        self.setAcceptHoverEvents(True)

        self.name_item = QGraphicsTextItem(self.model_host.model.name, self)

        self.name_item.setPos(10, 5)

        n_inputs = len(self.model_host.model.in_vars)
        n_outputs = len(self.model_host.model.out_vars)

        self.inputs = [PortItem(self, True, i, n_inputs) for i in range(n_inputs)]
        self.outputs = [PortItem(self, False, i, n_outputs) for i in range(n_outputs)]

        self.resize_handle = ResizeHandle(self)

        super().setRect(0, 0, 100, 60)
        self.update_ports()
        self.update_handle_position()

        self._resizing_from_handle = False


    def mouseDoubleClickEvent(self, event):
        # --- Generic block editing ---
        dlg = QDialog()
        dlg.setWindowTitle(f"Editing GENERIC Block ({self.model_host.model.uid})")
        dlg.resize(800, 600)
        layout = QVBoxLayout(dlg)

        # The block’s children become the "main_block" in the sub-editor
        sub_editor = BlockEditor(block=self.model_host.model, diagram=self.model_host.diagram)
        sub_editor.rebuild_scene_from_diagram()

        layout.addWidget(sub_editor)

        dlg.exec()
        return

        # fallback to default
        super().mouseDoubleClickEvent(event)

    def resize_block(self, width, height):
        # Update geometry safely
        self.prepareGeometryChange()
        QGraphicsRectItem.setRect(self, 0, 0, width, height)
        self.update_ports()
        self.update_handle_position()

    def update_handle_position(self):
        rect = self.rect()
        self._resizing_from_handle = False
        self.resize_handle.setPos(rect.width(), rect.height())
        self._resizing_from_handle = True

    def _set_rect_internal(self, w, h):
        QGraphicsRectItem.setRect(self, 0, 0, w, h)
        self.update_ports()
        self.update_handle_position()

    def setRect(self, x, y, w, h):
        if not getattr(self, '_suppress_resize', False):
            self._set_rect_internal(w, h)

    def update_ports(self):
        for i, port in enumerate(self.inputs):
            spacing = self.rect().height() / (len(self.inputs) + 1)
            port.setPos(0, spacing * (i + 1))
        for i, port in enumerate(self.outputs):
            spacing = self.rect().height() / (len(self.outputs) + 1)
            port.setPos(self.rect().width(), spacing * (i + 1))
        self.update_handle_position()
        # Also update connections
        for port in self.inputs + self.outputs:
            if port.connection:
                port.connection.update_path()

    def hoverEnterEvent(self, event):
        QApplication.setOverrideCursor(Qt.CursorShape.OpenHandCursor)

    def hoverLeaveEvent(self, event):
        QApplication.restoreOverrideCursor()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            for port in self.inputs + self.outputs:
                if port.connection:
                    port.connection.update_path()
        return super().itemChange(change, value)

    def contextMenuEvent(self, event):
        menu = QMenu()

        delete_action = QAction("Remove Block", menu)
        menu.addAction(delete_action)


        edit_action = QAction("Edit Block", menu)
        menu.addAction(edit_action)

        chosen = menu.exec(event.screenPos())

        if chosen == delete_action:
            for port in self.inputs + self.outputs:
                if port.connection:
                    self.scene().removeItem(port.connection)
                    if port.connection.source_port:
                        port.connection.source_port.connection = None
                    if port.connection.target_port:
                        port.connection.target_port.connection = None
            self.scene().removeItem(self)

        elif chosen == edit_action:
            self.open_generic_editor()

    def open_generic_editor(self):
        dlg = QDialog()
        dlg.setWindowTitle(f"Edit Generic Block ({self.model_host.model.uid})")
        dlg.resize(600, 400)
        layout = QVBoxLayout(dlg)

        # Section: Algebraic Variables
        alg_section = self.create_variable_section("Algebraic Variables", self.model_host.model.algebraic_vars)
        layout.addLayout(alg_section)

        # Section: State Variables
        state_section = self.create_variable_section("State Variables", self.model_host.model.state_vars)
        layout.addLayout(state_section)

        # Section: Algebraic Equations
        alg_eq_section = self.create_equation_section("Algebraic Equations", self.model_host.model.algebraic_eqs)
        layout.addLayout(alg_eq_section)

        # Section: State Equations
        state_eq_section = self.create_equation_section("State Equations", self.model_host.model.state_eqs)
        layout.addLayout(state_eq_section)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)

        dlg.exec()

    def create_variable_section(self, title, var_list):
        layout = QVBoxLayout()

        label = QLabel(title)
        layout.addWidget(label)

        list_widget = QListWidget()
        for v in var_list:

            list_widget.addItem(v.name)
        layout.addWidget(list_widget)

        add_btn = QPushButton("+")
        layout.addWidget(add_btn)

        def add_var():
            text, ok = QInputDialog.getText(None, f"Add {title}", "Variable name:")
            if ok and text:
                var_list.append(Var(text))
                print(self.model_host.model.algebraic_vars)
                list_widget.addItem(text)

        add_btn.clicked.connect(add_var)

        return layout

    def create_equation_section(self, title, eq_list):
        layout = QVBoxLayout()

        label = QLabel(title)
        layout.addWidget(label)

        list_widget = QListWidget()
        for eq in eq_list:
            text = symbolic_to_string(eq)
            list_widget.addItem(text)
        layout.addWidget(list_widget)

        add_btn = QPushButton("+")
        layout.addWidget(add_btn)

        def add_eq():
            text, ok = QInputDialog.getText(None, f"Add {title}", "Equation:")
            if ok and text:
                print(text)
                sym_expr = make_symbolic(text)
                print(type(sym_expr))
                eq_list.append(sym_expr)
                print(self.model_host.model.algebraic_eqs)
                list_widget.addItem(text)

        add_btn.clicked.connect(add_eq)

        return layout


class GraphicsView(QGraphicsView):
    """
    GraphicsView
    """

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # self.setDragMode(QGraphicsView.DragMode.NoDrag)
        # self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

        # self.setMouseTracking(True)
        # self.setInteractive(True)

        self._panning = False
        self._pan_start = QPointF()

    def wheelEvent(self, event):
        """

        :param event:
        :return:
        """
        zoom_in = event.angleDelta().y() > 0
        zoom_factor = 1.15 if zoom_in else 1 / 1.15
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        """

        :param event:
        :return:
        """
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """

        :param event:
        :return:
        """
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """

        :param event:
        :return:
        """
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)



def create_block_of_type(block_type: BlockType, ins: int, outs: int, const_value: Optional[float] = None) -> Block:
    """
    Create a Block appropriate for block_type. Use placeholder Vars for inputs/outputs
    so that Block.in_vars and Block.out_vars are not empty.
    """

    def placeholders(n, base):
        return [Var(f"{base}{i}") for i in range(n)]

    if block_type == BlockType.CONSTANT:
        value = const_value if const_value is not None else 0.0
        y, blk = constant(value, name="const")
        blk.name = "const"
        blk.in_vars = []
        blk.out_vars = [y]
        return blk

    if block_type == BlockType.VARIABLE:
        y, blk = variable(name="variable")
        blk.name = "variable"
        blk.in_vars = []
        blk.out_vars = [y]
        return blk

    # GAIN (single input -> single output)
    if block_type == BlockType.GAIN:
        u = Var("gain_u")  # placeholder input var
        y, blk = gain(1.0, u, name="gain_out")
        if not getattr(blk, "in_vars", None):
            blk.in_vars = [u]
        if not getattr(blk, "out_vars", None):
            blk.out_vars = [y]
        return blk

    # SUM / ADDER (N inputs)
    if block_type == BlockType.SUM or block_type == BlockType.PRODUCT:
        # for SUM use adder; for PRODUCT you may later implement product()
        inputs = placeholders(ins, "sum_in_")
        y, blk = adder(inputs, name="sum_out")
        blk.name = "sum"
        if not getattr(blk, "in_vars", None):
            blk.in_vars = inputs
        if not getattr(blk, "out_vars", None):
            blk.out_vars = [y]
        return blk

    # INTEGRATOR (1 input -> 1 state output)
    if block_type == BlockType.INTEGRATOR:
        u = Var("int_u")
        x, blk = integrator(u, name="x")
        if not getattr(blk, "in_vars", None):
            blk.in_vars = [u]
        if not getattr(blk, "out_vars", None):
            blk.out_vars = [x]
        return blk

    # SOURCE: a block with only an output (like a constant/source)
    if block_type == BlockType.SOURCE:
        y, blk = constant(0.0, name="source_out")
        if not getattr(blk, "out_vars", None):
            blk.out_vars = [y]
        blk.in_vars = []
        return blk

    # DRAIN: a sink with inputs but no outputs
    if block_type == BlockType.DRAIN:
        ins_vars = placeholders(ins, "drain_in_")
        blk = Block(name="DRAIN")
        blk.in_vars = ins_vars
        blk.out_vars = []
        return blk

    if block_type == BlockType.GENERIC:
        blk = generic()
        blk.name = "generic"
        return blk

    in_vars = placeholders(ins, f"{block_type.name.lower()}_in_")
    out_vars = [Var(f"{block_type.name.lower()}_out{i}") for i in range(outs)]
    blk = Block(name=block_type.name)
    blk.in_vars = in_vars
    blk.out_vars = out_vars
    return blk


class DiagramScene(QGraphicsScene):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.temp_line = None
        self.source_port = None

        self._main_block = Block()

    def get_main_block(self):
        return self._main_block


    def mousePressEvent(self, event):
        for item in self.items(event.scenePos()):
            if isinstance(item, PortItem) and not item.is_input and not item.is_connected():
                self.source_port = item
                path = QPainterPath(item.scenePos())
                self.temp_line = self.addPath(path, QPen(Qt.PenStyle.DashLine))
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """

        :param event:
        :return:
        """
        if self.temp_line:
            start = self.source_port.scenePos()
            end = event.scenePos()
            mid_x = (start.x() + end.x()) / 2
            c1 = QPointF(mid_x, start.y())
            c2 = QPointF(mid_x, end.y())
            path = QPainterPath(start)
            path.cubicTo(c1, c2, end)
            self.temp_line.setPath(path)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """

        :param event:
        :return:
        """
        if self.temp_line:
            # FIX: scan items under mouse for a valid input Port
            for item in self.items(event.scenePos()):
                if isinstance(item, PortItem) and item.is_input and not item.is_connected():
                    dst_port: PortItem = item
                    connection = ConnectionItem(self.source_port, dst_port)

                    dst_var = self.source_port.subsystem.subsys.out_vars[self.source_port.index]

                    dst_port.subsystem.subsys.in_vars[dst_port.index] = dst_var

                    self.addItem(connection)

                    color = connection.pen().color().name()
                    # save branches in diagram
                    self.editor.diagram.add_branch(self.source_port.subsystem.subsys.uid, dst_port.subsystem.subsys.uid, self.source_port.index, dst_port.index, color)
                    break

            self.removeItem(self.temp_line)
            self.temp_line = None
            self.source_port = None
        else:
            super().mouseReleaseEvent(event)


class DynamicLibraryModel(QStandardItemModel):
    """
    Items model to host the draggable icons
    This is the list of draggable items
    """

    def __init__(self) -> None:
        """
        Items model to host the draggable icons
        """
        QStandardItemModel.__init__(self)

        self.setColumnCount(1)

        self.mime_dict: Dict[object, BlockType] = dict()

        for bt in BlockType:
            self.add(name=bt.name, icon_name="dyn")
            t = self.to_bytes_array(bt.name)
            self.mime_dict[t] = bt

    def get_type(self, t) -> BlockType | None:
        return self.mime_dict.get(t, None)

    def add(self, name: str, icon_name: str):
        """
        Add element to the library
        :param name: Name of the element
        :param icon_name: Icon name, the path is taken care of
        :return:
        """
        _icon = QIcon()
        _icon.addPixmap(QPixmap(f":/Icons/icons/{icon_name}.png"))
        _item = QStandardItem(_icon, name)
        _item.setToolTip(f"Drag & drop {name} into the schematic")
        self.appendRow(_item)

    @staticmethod
    def to_bytes_array(val: str) -> QByteArray:
        """
        Convert string to QByteArray
        :param val: string
        :return: QByteArray
        """
        data = QByteArray()
        stream = QDataStream(data, QIODevice.WriteOnly)
        stream.writeQString(val)
        return data

    def mimeData(self, idxs: List[QModelIndex]) -> QMimeData:
        """

        @param idxs:
        @return:
        """
        mimedata = QMimeData()
        for idx in idxs:
            if idx.isValid():
                txt = self.data(idx, Qt.ItemDataRole.DisplayRole)

                data = QByteArray()
                stream = QDataStream(data, QIODevice.WriteOnly)
                stream.writeQString(txt)

                mimedata.setData('component/name', data)
        return mimedata

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """

        :param index:
        :return:
        """
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDragEnabled


class BlockEditor(QSplitter):
    """
    BlockEditor
    """

    def __init__(self,
                 block: Block,
                 diagram: BlockDiagram,
                 parent=None):
        super().__init__(parent)

        self.main_block = block
        self.diagram = diagram

        # --------------------------------------------------------------------------------------------------------------
        # Widget creation
        # --------------------------------------------------------------------------------------------------------------
        # Widget layout and child widgets:
        self.horizontal_layout = QHBoxLayout(self)

        # Actual libraryView object
        self.library_view = QListView(self)
        self.library_view.setViewMode(self.library_view.ViewMode.ListMode)
        self.library_view.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.library_model = DynamicLibraryModel()
        self.library_view.setModel(self.library_model)
        change_font_size(self.library_view, 9)

        self.scene = DiagramScene(self)
        self.view = GraphicsView(self.scene)

        self.view.dragEnterEvent = self.graphicsDragEnterEvent
        self.view.dragMoveEvent = self.graphicsDragMoveEvent
        self.view.dropEvent = self.graphicsDropEvent

        self.addWidget(self.library_view)
        self.addWidget(self.view)

        # self.block_system = self.scene.get_main_block()

        self.setStretchFactor(0, 1)
        self.setStretchFactor(1, 1000)

        self.resize(800, 600)

    def graphicsDragEnterEvent(self, event: QDragEnterEvent) -> None:
        """

        @param event:
        @return:
        """
        if event.mimeData().hasFormat('component/name'):
            event.accept()

    def graphicsDragMoveEvent(self, event: QDragMoveEvent) -> None:
        """
        Move element
        @param event:
        @return:
        """
        if event.mimeData().hasFormat('component/name'):
            event.accept()

    def graphicsDropEvent(self, event: QDropEvent) -> None:
        """
        Create an element
        @param event:
        @return:
        """
        if event.mimeData().hasFormat('component/name'):
            obj_type = event.mimeData().data('component/name')

            point0 = self.view.mapToScene(int(event.position().x()), int(event.position().y()))
            x0 = point0.x()
            y0 = point0.y()

            tpe = self.library_model.get_type(obj_type)

            if tpe == BlockType.GENERIC:

                model_host: DynamicModelHost = DynamicModelHost()

                if model_host is not None:
                    self.main_block.add(model_host.model)

                    item = ModelHostItem(model_host)

                    self.scene.addItem(item)
                    item.setPos(QPointF(x0, y0))
                    # save nodes in diagram
                    self.diagram.add_node(
                        x=x0,
                        y=y0,
                        device_uid=model_host.model.uid,
                        tpe=tpe.name,
                        subdiagram= model_host.diagram
                    )

            else:

                blk: Block = create_block_of_type(block_type=tpe,
                                              ins=2,
                                              outs=1,
                                              const_value=3)

                if blk is not None:
                    self.main_block.add(blk)
                    item = BlockItem(blk)

                    self.scene.addItem(item)
                    item.setPos(QPointF(x0, y0))
                    # save nodes in diagram
                    self.diagram.add_node(
                        x=x0,
                        y=y0,
                        device_uid=blk.uid,
                        tpe=tpe.name
                    )

    def rebuild_scene_from_diagram(self):
        """Rebuilds the graphical scene from saved diagram data"""
        self.scene.clear()

        uid_to_blockitem = {}

        # Recreate nodes
        for uid, node in self.diagram.node_data.items():
            block_type = BlockType[node.tpe]
            if block_type == BlockType.GENERIC:
                model_host = DynamicModelHost()
                model_host.diagram = node.sub_diagram
                item = ModelHostItem(model_host)

                self.scene.addItem(item)
                item.setPos(QPointF(node.x, node.y))


            else:
                ins = 2 if block_type in {BlockType.SUM, BlockType.PRODUCT, BlockType.MIN, BlockType.MAX} else 1
                outs = 1
                if block_type == BlockType.SOURCE:
                    ins = 0
                elif block_type == BlockType.DRAIN:
                    outs = 0

                blk = create_block_of_type(block_type, ins=ins, outs=outs)
                blk.uid = uid

                block_item = BlockItem(blk)
                self.scene.addItem(block_item)
                block_item.setPos(node.x, node.y)

                uid_to_blockitem[uid] = block_item


        # Recreate connections
        for con in self.diagram.con_data:
            src_item = uid_to_blockitem.get(con.from_uid)
            dst_item = uid_to_blockitem.get(con.to_uid)
            if not src_item or not dst_item:
                continue

            try:
                src_port = src_item.outputs[con.port_number_from]
                dst_port = dst_item.inputs[con.port_number_to]
            except IndexError:
                continue  # invalid port number

            connection = ConnectionItem(src_port, dst_port)
            self.scene.addItem(connection)

        self.block_system = self.scene.get_main_block()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BlockEditor(
        block=Block(),
        diagram=BlockDiagram()
    )
    window.show()
    sys.exit(app.exec())
