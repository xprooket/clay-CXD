"""
Pytest configuration and fixtures for CXD Classifier tests.

This module provides common fixtures and configuration for all tests,
including test classifiers, sample data, and golden datasets.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Generator

# Import CXD components (will be implemented in actual modules)
# from cxd_classifier import CXDConfig, MetaCXDClassifier
# from cxd_classifier.testing import GoldenDataset
# from cxd_classifier.core.types import CXDFunction

@pytest.fixture(scope="session")
def temp_cache_dir() -> Generator[Path, None, None]:
    """Create a temporary cache directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="cxd_test_cache_")
    cache_path = Path(temp_dir)
    yield cache_path
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def test_config(temp_cache_dir: Path):
    """Create a test configuration."""
    # Will be implemented when CXDConfig is available
    # return CXDConfig(
    #     cache_dir=str(temp_cache_dir),
    #     canonical_examples_path="./tests/data/test_examples.yaml",
    #     log_level="DEBUG",
    #     enable_cache_persistence=False,
    #     batch_size=4,
    # )
    pass

@pytest.fixture
def test_classifier(test_config):
    """Create a test classifier instance."""
    # Will be implemented when MetaCXDClassifier is available
    # return MetaCXDClassifier(config=test_config)
    pass

@pytest.fixture
def sample_texts() -> List[str]:
    """Sample texts for testing."""
    return [
        "Buscar información en la base de datos",
        "Esta conversación se relaciona con el trabajo anterior", 
        "Procesar y analizar los datos disponibles",
        "Filtrar resultados según criterios específicos",
        "Conectar con el contexto de la discusión previa",
        "Transformar datos en formato utilizable",
        "No pude encontrar memorias relevantes",
        "¿Qué información necesitas sobre el proyecto?",
        "Generar un resumen de los resultados obtenidos",
        "El contexto de esta pregunta no está claro"
    ]

@pytest.fixture
def expected_control_texts() -> List[str]:
    """Texts expected to be classified as CONTROL."""
    return [
        "Buscar información en la base de datos",
        "Filtrar resultados según criterios específicos", 
        "¿Qué información necesitas sobre el proyecto?",
        "Encontrar documentos relevantes",
        "Seleccionar la mejor opción disponible",
    ]

@pytest.fixture
def expected_context_texts() -> List[str]:
    """Texts expected to be classified as CONTEXT."""
    return [
        "Esta conversación se relaciona con el trabajo anterior",
        "Conectar con el contexto de la discusión previa", 
        "El contexto de esta pregunta no está claro",
        "Hacer referencia a nuestra charla anterior",
        "Vincular con el proyecto discutido ayer",
    ]

@pytest.fixture
def expected_data_texts() -> List[str]:
    """Texts expected to be classified as DATA."""
    return [
        "Procesar y analizar los datos disponibles",
        "Transformar datos en formato utilizable",
        "Generar un resumen de los resultados obtenidos",
        "Calcular estadísticas del dataset",
        "Extraer patrones de la información",
    ]

@pytest.fixture
def golden_dataset_dict() -> Dict:
    """Golden dataset as dictionary for testing."""
    return {
        "version": "1.0",
        "description": "Test golden dataset for CXD Classifier",
        "examples": [
            {
                "text": "Buscar información en la base de datos",
                "expected_function": "CONTROL",
                "expected_confidence": 0.8,
                "tags": ["search", "database"],
            },
            {
                "text": "Esta conversación se relaciona con el trabajo anterior",
                "expected_function": "CONTEXT", 
                "expected_confidence": 0.9,
                "tags": ["relation", "previous"],
            },
            {
                "text": "Procesar y analizar los datos disponibles",
                "expected_function": "DATA",
                "expected_confidence": 0.8,
                "tags": ["process", "analyze"],
            },
        ]
    }

@pytest.fixture
def performance_test_texts() -> List[str]:
    """Large set of texts for performance testing."""
    base_texts = [
        "Search for relevant information",
        "This relates to our previous work", 
        "Process the available data",
        "Filter the results carefully",
        "Connect with prior context",
        "Transform the input format",
    ]
    
    # Repeat and vary texts for performance testing
    performance_texts = []
    for i in range(100):
        for text in base_texts:
            performance_texts.append(f"{text} - variant {i}")
    
    return performance_texts

# Pytest markers for test organization
pytest_marks = {
    "unit": pytest.mark.unit,
    "integration": pytest.mark.integration, 
    "performance": pytest.mark.performance,
    "slow": pytest.mark.slow,
    "gpu": pytest.mark.gpu,
}

# Test configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance benchmark"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (may take several seconds)"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU support"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add 'unit' marker to all tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add 'integration' marker to all tests in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add 'performance' marker to all tests in performance/ directory
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Mark tests with 'benchmark' in name as performance tests
        if "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        # Mark tests with 'slow' in name as slow tests
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)
