def safe_uint32(value):
    """
    Converts negative integers to their unsigned equivalent within the uint32 range.
    """
    if isinstance(value, int) and value < 0:
        return value & 0xFFFFFFFF  # Wrap negative values into uint32 range
    return value