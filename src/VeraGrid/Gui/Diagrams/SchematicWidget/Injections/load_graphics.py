# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.  
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations
from typing import TYPE_CHECKING

from PySide6 import QtWidgets
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPen, QIcon, QPixmap, QPolygonF
from PySide6.QtWidgets import QMenu
from VeraGrid.Gui.gui_functions import add_menu_entry
from VeraGrid.Gui.Diagrams.generic_graphics import ACTIVE, DEACTIVATED, OTHER, Polygon, Square
from VeraGrid.Gui.Diagrams.SchematicWidget.Injections.injections_template_graphics import InjectionTemplateGraphicItem
from VeraGrid.Gui.Diagrams.Editors.RmsModelEditor.rms_model_editor_engine import RmsModelEditorGUI
from VeraGrid.Gui.messages import yes_no_question
from VeraGridEngine.Devices.Injections.load import Load

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from VeraGrid.Gui.Diagrams.SchematicWidget.schematic_widget import SchematicWidget


class LoadGraphicItem(InjectionTemplateGraphicItem):

    def __init__(self, parent, api_obj: Load, editor: SchematicWidget):
        """

        :param parent:
        :param api_obj:
        :param editor:
        """
        InjectionTemplateGraphicItem.__init__(self,
                                              parent=parent,
                                              api_obj=api_obj,
                                              editor=editor,
                                              device_type_name='load',
                                              w=20,
                                              h=20)

        # triangle
        self.set_glyph(glyph=Polygon(self,
                                     polygon=QPolygonF([QPointF(0, 0),
                                                        QPointF(self.w, 0),
                                                        QPointF(self.w / 2, self.h)]),
                                     update_nexus_fcn=self.update_nexus)
                       )

    @property
    def api_object(self) -> Load:
        return self._api_object

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent):
        """
        Display context menu
        @param event:
        @return:
        """
        if self.api_object is not None:
            menu = QMenu()
            menu.addSection("Load")


            add_menu_entry(menu=menu,
                        text="Rms Editor",
                        function_ptr=self.edit_rms,
                        icon_path=":/Icons/icons/edit.svg")

            menu.exec(event.screenPos())
        else:
            pass

    def edit_rms(self):
        """
        Open the appropriate editor dialogue
        :return:
        """
        dlg = RmsModelEditorGUI(self.api_object, parent=self.editor)
        dlg.show()
