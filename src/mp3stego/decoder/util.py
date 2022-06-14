BYTE_LENGTH = 8

H0 = {3, 6, 8, 11, 12, 15, 17, 19, 21, 23, 24, 26, 28, 30}


def char_to_int(four_bytes: list) -> int:
    """
    Puts four bytes into a single four byte integer type.

    :param four_bytes: the bytes
    :type four_bytes: list

    :return: the int value
    :rtype int
    """
    num = 0x00
    for i in range(4):
        num = (num << 7) + four_bytes[i]
    return int(num)


def get_bits(buffer: list, start_bit: int, slice_len: int):
    """
    Assumes that end_bit is greater than start_bit and that the result is less than 32 bits, length of an unsigned type.

    :param buffer: buffer of bytes
    :param start_bit: the starting bit
    :param slice_len: the length

    :return: the bits from buffer[start_bit] to buffer[start_bit + slice_len]. transform the bytes from buffer into bits
    """
    # exclude the last bit of the slice
    end_bit = start_bit + slice_len - 1

    start_byte = start_bit >> 3
    end_byte = end_bit >> 3

    buff_copy = buffer.copy()

    buff_len = len(buffer)
    if end_byte >= buff_len:
        # pad with zeros
        buff_copy.extend([0 for _ in range(end_byte - buff_len + 1)])

    bits = []
    for idx in range(start_byte, end_byte + 1):
        num = buff_copy[idx]
        out = [1 if num & (1 << (BYTE_LENGTH - 1 - n)) else 0 for n in range(BYTE_LENGTH)]
        bits.extend(out)

    # update to relative positions in the bits array
    start_bit %= 8
    end_bit = start_bit + slice_len - 1

    bit_slice = bits[start_bit:end_bit + 1]

    result = 0

    bit_slice.reverse()

    for i in range(len(bit_slice)):
        result += (2 ** i) * bit_slice[i]

    return result


def bit_from_huffman_tables(all_huffman_tables):
    """
    calc the bits from the huffman tables, according to the steganography

    :param all_huffman_tables: list of huffman tables used.

    :return: string contains bits
    """

    s = ""
    for x in all_huffman_tables[-1]:
        if x == 0:
            continue
        s += "0" if x in H0 else "1"
    return s
