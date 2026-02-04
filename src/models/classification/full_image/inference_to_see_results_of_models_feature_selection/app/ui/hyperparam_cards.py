"""
Hyperparameter card widgets for modern search space UI.

Provides card-based controls for editing hyperparameter ranges and choices.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QSpinBox, QPushButton, QLineEdit, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from typing import Dict, List, Optional, Any


# Card styling
CARD_STYLE = """
QFrame {
    border: 1px solid #ccc;
    border-radius: 8px;
    background-color: #fafafa;
    padding: 10px;
}
"""

CARD_STYLE_ERROR = """
QFrame {
    border: 2px solid #e74c3c;
    border-radius: 8px;
    background-color: #fafafa;
    padding: 10px;
}
"""

ERROR_LABEL_STYLE = "color: #e74c3c; font-size: 10px; font-weight: bold;"
TITLE_STYLE = "font-weight: bold; font-size: 13px; color: #2c3e50;"
SUBTITLE_STYLE = "font-size: 10px; color: #7f8c8d; font-style: italic;"
PREVIEW_STYLE = "font-size: 10px; color: #34495e; background-color: #ecf0f1; padding: 4px; border-radius: 3px;"


class HyperParamCard(QFrame):
    """
    Base class for hyperparameter cards.
    
    Signals:
        value_changed: Emitted when parameter value changes
    """
    
    value_changed = pyqtSignal()
    
    def __init__(self, param_name: str, param_type: str, display_name: str = None, parent=None):
        """
        Initialize hyperparameter card.
        
        Args:
            param_name: Internal parameter name
            param_type: Type string ('float', 'int', 'categorical')
            display_name: Human-readable display name
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.param_name = param_name
        self.param_type = param_type
        self.display_name = display_name or param_name.replace('_', ' ').title()
        
        self._setup_ui()
        self.setStyleSheet(CARD_STYLE)
        
    def _setup_ui(self):
        """Setup base UI structure."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Title
        self.title_label = QLabel(self.display_name)
        self.title_label.setStyleSheet(TITLE_STYLE)
        layout.addWidget(self.title_label)
        
        # Subtitle (type)
        self.subtitle_label = QLabel(f"Type: {self.param_type}")
        self.subtitle_label.setStyleSheet(SUBTITLE_STYLE)
        layout.addWidget(self.subtitle_label)
        
        # Controls container (to be filled by subclasses)
        self.controls_layout = QVBoxLayout()
        self.controls_layout.setSpacing(6)
        layout.addLayout(self.controls_layout)
        
        # Preview label
        self.preview_label = QLabel()
        self.preview_label.setStyleSheet(PREVIEW_STYLE)
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)
        
        # Error label (hidden by default)
        self.error_label = QLabel()
        self.error_label.setStyleSheet(ERROR_LABEL_STYLE)
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)
        
    def get_spec(self) -> Dict[str, Any]:
        """Get parameter specification. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def set_spec(self, spec: Dict[str, Any]):
        """Set parameter specification. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def is_valid(self) -> tuple[bool, str]:
        """
        Check if current configuration is valid.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        raise NotImplementedError
    
    def update_preview(self):
        """Update preview text. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def _show_error(self, message: str):
        """Show error state."""
        self.setStyleSheet(CARD_STYLE_ERROR)
        self.error_label.setText(f"⚠ {message}")
        self.error_label.show()
    
    def _clear_error(self):
        """Clear error state."""
        self.setStyleSheet(CARD_STYLE)
        self.error_label.hide()
    
    def _on_value_changed(self):
        """Handle value change - validate and emit signal."""
        is_valid, error_msg = self.is_valid()
        if is_valid:
            self._clear_error()
        else:
            self._show_error(error_msg)
        
        self.update_preview()
        self.value_changed.emit()


class FloatRangeCard(HyperParamCard):
    """Card for float range parameters."""
    
    def __init__(self, param_name: str, display_name: str = None, 
                 min_value: float = 0.0, max_value: float = 1.0, 
                 step: float = 0.01, decimals: int = 3, parent=None):
        """
        Initialize float range card.
        
        Args:
            param_name: Parameter name
            display_name: Display name
            min_value: Default minimum value
            max_value: Default maximum value
            step: Step size for spinboxes
            decimals: Number of decimal places
            parent: Parent widget
        """
        self._min_value = min_value
        self._max_value = max_value
        self._step = step
        self._decimals = decimals
        
        super().__init__(param_name, 'float', display_name, parent)
        
    def _setup_ui(self):
        """Setup float range UI."""
        super()._setup_ui()
        
        # Min value
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min:"))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setDecimals(self._decimals)
        self.min_spin.setSingleStep(self._step)
        self.min_spin.setRange(-1e6, 1e6)
        self.min_spin.setValue(self._min_value)
        self.min_spin.valueChanged.connect(self._on_value_changed)
        min_layout.addWidget(self.min_spin, stretch=1)
        self.controls_layout.addLayout(min_layout)
        
        # Max value
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max:"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setDecimals(self._decimals)
        self.max_spin.setSingleStep(self._step)
        self.max_spin.setRange(-1e6, 1e6)
        self.max_spin.setValue(self._max_value)
        self.max_spin.valueChanged.connect(self._on_value_changed)
        max_layout.addWidget(self.max_spin, stretch=1)
        self.controls_layout.addLayout(max_layout)
        
        self.update_preview()
    
    def get_spec(self) -> Dict[str, Any]:
        """Get float range specification."""
        return {
            'name': self.param_name,
            'type': 'float',
            'min_value': self.min_spin.value(),
            'max_value': self.max_spin.value()
        }
    
    def set_spec(self, spec: Dict[str, Any]):
        """Set float range specification."""
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        
        self.min_spin.setValue(spec.get('min_value', self._min_value))
        self.max_spin.setValue(spec.get('max_value', self._max_value))
        
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)
        
        self.update_preview()
    
    def is_valid(self) -> tuple[bool, str]:
        """Check if range is valid."""
        min_val = self.min_spin.value()
        max_val = self.max_spin.value()
        
        if min_val >= max_val:
            return False, f"Min ({min_val}) must be less than Max ({max_val})"
        
        return True, ""
    
    def update_preview(self):
        """Update preview text."""
        min_val = self.min_spin.value()
        max_val = self.max_spin.value()
        self.preview_label.setText(f"Current range: [{min_val:.{self._decimals}f}, {max_val:.{self._decimals}f}]")


class IntRangeCard(HyperParamCard):
    """Card for integer range parameters."""
    
    def __init__(self, param_name: str, display_name: str = None,
                 min_value: int = 0, max_value: int = 100, 
                 step: int = 1, parent=None):
        """
        Initialize integer range card.
        
        Args:
            param_name: Parameter name
            display_name: Display name
            min_value: Default minimum value
            max_value: Default maximum value
            step: Step size
            parent: Parent widget
        """
        self._min_value = min_value
        self._max_value = max_value
        self._step = step
        
        super().__init__(param_name, 'int', display_name, parent)
    
    def _setup_ui(self):
        """Setup integer range UI."""
        super()._setup_ui()
        
        # Min value
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min:"))
        self.min_spin = QSpinBox()
        self.min_spin.setSingleStep(self._step)
        self.min_spin.setRange(-1000000, 1000000)
        self.min_spin.setValue(self._min_value)
        self.min_spin.valueChanged.connect(self._on_value_changed)
        min_layout.addWidget(self.min_spin, stretch=1)
        self.controls_layout.addLayout(min_layout)
        
        # Max value
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max:"))
        self.max_spin = QSpinBox()
        self.max_spin.setSingleStep(self._step)
        self.max_spin.setRange(-1000000, 1000000)
        self.max_spin.setValue(self._max_value)
        self.max_spin.valueChanged.connect(self._on_value_changed)
        max_layout.addWidget(self.max_spin, stretch=1)
        self.controls_layout.addLayout(max_layout)
        
        self.update_preview()
    
    def get_spec(self) -> Dict[str, Any]:
        """Get integer range specification."""
        return {
            'name': self.param_name,
            'type': 'int',
            'min_value': self.min_spin.value(),
            'max_value': self.max_spin.value()
        }
    
    def set_spec(self, spec: Dict[str, Any]):
        """Set integer range specification."""
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        
        self.min_spin.setValue(spec.get('min_value', self._min_value))
        self.max_spin.setValue(spec.get('max_value', self._max_value))
        
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)
        
        self.update_preview()
    
    def is_valid(self) -> tuple[bool, str]:
        """Check if range is valid."""
        min_val = self.min_spin.value()
        max_val = self.max_spin.value()
        
        if min_val >= max_val:
            return False, f"Min ({min_val}) must be less than Max ({max_val})"
        
        return True, ""
    
    def update_preview(self):
        """Update preview text."""
        min_val = self.min_spin.value()
        max_val = self.max_spin.value()
        self.preview_label.setText(f"Current range: [{min_val}, {max_val}]")


class CategoricalCard(HyperParamCard):
    """Card for categorical parameters with chip-based selection."""
    
    def __init__(self, param_name: str, display_name: str = None,
                 choices: List[int] = None, parent=None):
        """
        Initialize categorical card.
        
        Args:
            param_name: Parameter name
            display_name: Display name
            choices: List of integer choices
            parent: Parent widget
        """
        self._default_choices = choices or []
        self.choices = list(self._default_choices)
        
        super().__init__(param_name, 'categorical', display_name, parent)
    
    def _setup_ui(self):
        """Setup categorical UI with chips."""
        super()._setup_ui()
        
        # Chips container
        self.chips_layout = QHBoxLayout()
        self.chips_layout.setSpacing(6)
        self.controls_layout.addLayout(self.chips_layout)
        
        # Add new choice controls
        add_layout = QHBoxLayout()
        add_layout.addWidget(QLabel("Add choice:"))
        self.add_input = QLineEdit()
        self.add_input.setPlaceholderText("Enter integer...")
        self.add_input.setMaximumWidth(100)
        add_layout.addWidget(self.add_input)
        
        self.add_btn = QPushButton("+ Add")
        self.add_btn.clicked.connect(self._add_choice)
        add_layout.addWidget(self.add_btn)
        add_layout.addStretch()
        
        self.controls_layout.addLayout(add_layout)
        
        # Build chips for default choices
        self._rebuild_chips()
        
        self.update_preview()
    
    def _rebuild_chips(self):
        """Rebuild all choice chips."""
        # Clear existing chips
        while self.chips_layout.count():
            item = self.chips_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Sort choices for consistent display
        self.choices.sort()
        
        # Create chip for each choice
        for choice in self.choices:
            chip = self._create_chip(choice)
            self.chips_layout.addWidget(chip)
        
        self.chips_layout.addStretch()
    
    def _create_chip(self, value: int) -> QPushButton:
        """Create a chip button for a choice."""
        chip = QPushButton(str(value))
        chip.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 12px;
                padding: 4px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #e74c3c;
            }
        """)
        chip.setToolTip(f"Click to remove {value}")
        chip.clicked.connect(lambda: self._remove_choice(value))
        chip.setMaximumWidth(60)
        return chip
    
    def _add_choice(self):
        """Add a new choice from input."""
        text = self.add_input.text().strip()
        
        try:
            value = int(text)
            
            if value in self.choices:
                self._show_error(f"Choice {value} already exists")
                return
            
            self.choices.append(value)
            self._rebuild_chips()
            self.add_input.clear()
            self._on_value_changed()
            
        except ValueError:
            self._show_error("Please enter a valid integer")
    
    def _remove_choice(self, value: int):
        """Remove a choice."""
        if value in self.choices:
            self.choices.remove(value)
            self._rebuild_chips()
            self._on_value_changed()
    
    def get_spec(self) -> Dict[str, Any]:
        """Get categorical specification."""
        return {
            'name': self.param_name,
            'type': 'categorical',
            'choices': sorted(self.choices)
        }
    
    def set_spec(self, spec: Dict[str, Any]):
        """Set categorical specification."""
        self.choices = list(spec.get('choices', self._default_choices))
        self._rebuild_chips()
        self.update_preview()
    
    def is_valid(self) -> tuple[bool, str]:
        """Check if choices are valid."""
        if len(self.choices) == 0:
            return False, "At least one choice is required"
        
        return True, ""
    
    def update_preview(self):
        """Update preview text."""
        choices_str = ', '.join(map(str, sorted(self.choices)))
        self.preview_label.setText(f"Current choices: [{choices_str}]")
