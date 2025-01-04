import typing
from dataclasses import fields
from enum import Enum
from typing import Dict

from model import Model, model_types

import streamlit as st

import logging

logger = logging.getLogger('Streamlit_logger')


def create_model_configurator(selected_model_type: str) -> typing.Optional[Model]:
    """Generates a Streamlit form for configuring a given dataclass."""
    model_class = model_types[selected_model_type]
    st.write(f"## Configure {model_class.name} model")
    config = {}
    for field in fields(model_class):
        field_name = field.name
        if field_name == "name":
            continue
        field_type = field.type
        default_value = field.default
        logger.info(f'type is {field_type} {field_type is None}')
        if field_type is str:
            config[field_name] = st.text_input(field_name.replace("_", " ").title(), value=default_value)
        elif field_type is float:
            config[field_name] = st.number_input(field_name.replace("_", " ").title(), value=default_value, step=0.1)
        elif field_type is int:
            config[field_name] = st.number_input(field_name.replace("_", " ").title(), value=default_value, step=1)
        elif field_type is bool:
            config[field_name] = st.checkbox(field_name.replace("_", " ").title(), value=default_value)
        elif field_type is Dict[float, float]:  # Handle Class Weights
            use_class_weights = st.checkbox("Use Class Weights")
            class_weight = {1: 1}
            if use_class_weights:
                num_classes = st.number_input("Number of curr_model parameters", min_value=1, step=1)
                class_weight = {}
                for i in range(num_classes):
                    class_name = st.text_input(f"Parameter {i + 1} name", f"{i + 1}")
                    weight = st.number_input(f"Weight for {class_name}", min_value=0.0, step=0.1, value=1.0)
                    class_weight[class_name] = weight
            config[field_name] = class_weight
        elif isinstance(field_type, Enum):
            config[field_name] = field_type
        elif issubclass(field_type, Enum):
            enum_options = [enum_member.value for enum_member in field_type]
            config[field_name] = field_type(st.selectbox(field_name.replace("_", " ").title(), enum_options,
                                                         index=enum_options.index(default_value.value)))
        else:
            st.write(f"Unsupported field type: {field_type} for field: {field_name}")
            config[field_name] = None

    try:
        model_class.__init__(selected_model_type)
        for elem in config:
            model_class.__setattr__(elem, config[elem])
        return model_class
    except Exception as e:
        st.error(f"Error creating curr_model instance: {e}")
        return None
