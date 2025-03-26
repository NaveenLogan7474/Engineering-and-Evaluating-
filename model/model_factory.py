"""Model factory module to create models with mixins dynamically."""

from typing import List, Type, Any
import sys
import builtins

# Dictionary to store all created classes, accessible from anywhere
_DYNAMIC_CLASSES = {}

def create_model(base_class, mixins=None, *args, **kwargs):
    """Create a model with the specified mixins.
    
    Args:
        base_class: The base model class to extend
        mixins: List of mixin classes to apply
        *args, **kwargs: Arguments to pass to the model constructor
        
    Returns:
        An instance of the created model class
    """
    if mixins is None:
        mixins = []
    
    # Combine mixins with base class
    if mixins:
        # Create a class name that uniquely identifies this combination
        class_name = f"{''.join(m.__name__.replace('Mixin', '') for m in mixins)}{base_class.__name__}"
        
        # Check if we already defined this class
        if class_name in _DYNAMIC_CLASSES:
            combined_class = _DYNAMIC_CLASSES[class_name]
        else:
            # Define the new class with the mixins
            combined_class = type(
                class_name,
                tuple(mixins) + (base_class,),
                {}
            )
            
            # Store the class in our registry
            _DYNAMIC_CLASSES[class_name] = combined_class
            
            # Register it in builtins to make it available for pickling
            setattr(builtins, class_name, combined_class)
        
        return combined_class(*args, **kwargs)
    
    return base_class(*args, **kwargs)


def standard_model(base_class, *args, **kwargs):
    """Create a standard model with all common mixins.
    
    This is a convenience function to create a model with the standard
    set of mixins: SerializationMixin, MetricsMixin, ValidationMixin, LoggingMixin
    
    Args:
        base_class: The base model class to extend
        *args, **kwargs: Arguments to pass to the model constructor
        
    Returns:
        An instance of the created model with standard mixins
    """
    from mixins import SerializationMixin, MetricsMixin, ValidationMixin, LoggingMixin
    
    standard_mixins = [SerializationMixin, MetricsMixin, ValidationMixin, LoggingMixin]
    return create_model(base_class, standard_mixins, *args, **kwargs) 