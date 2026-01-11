import struct
import binascii


def embedding_to_hex(embedding):
    """
    Convert a list of floats (sentence embedding) to a hexadecimal string.

    Args:
        embedding (list of float): The sentence embedding vector.

    Returns:
        str: Hexadecimal representation of the embedding.
    """
    return "".join(
        binascii.hexlify(struct.pack("f", val)).decode("utf-8") for val in embedding
    )


def hex_to_embedding(hex_str):
    """
    Convert a hexadecimal string back to a list of floats (sentence embedding).

    Args:
        hex_str (str): Hexadecimal representation of an embedding.

    Returns:
        list of float: The original sentence embedding vector.
    """
    # Each float is 4 bytes = 8 hex characters
    return [
        struct.unpack("f", binascii.unhexlify(hex_str[i : i + 8]))[0]
        for i in range(0, len(hex_str), 8)
    ]
