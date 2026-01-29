import pytest
import os
import sys

# Ajustar path si es necesario según la estructura de VeraGrid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture
def empty_grid():
    """Fixture para crear un objeto grid vacío de VeraGrid/GridCal."""
    # Nota: Importar aquí para evitar problemas de path en discovery inicial
    # from src.GridCal.Engine import MultiCircuit
    # return MultiCircuit()
    pass
