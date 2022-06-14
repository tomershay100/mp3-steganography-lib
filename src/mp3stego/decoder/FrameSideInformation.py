from mp3stego.decoder.FrameHeader import *
from mp3stego.decoder.util import *


class FrameSideInformation:
    """
    The frame side information class, contains all the information of teh side information of a frame in mp3 file.
    Side information contains info relevant to decode main data.
    """

    def __init__(self):
        self.__main_data_begin: int = 0
        self.__scfsi: np.ndarray = np.zeros((2, 4))

        # Side Info for the two granules. Allocate space for two granules and two channels.
        self.__part2_3_length: np.ndarray = np.zeros((2, 2))
        self.__part2_length: np.ndarray = np.zeros((2, 2))
        self.__big_value: np.ndarray = np.zeros((2, 2))
        self.__global_gain: np.ndarray = np.zeros((2, 2))
        self.__scale_fac_compress: np.ndarray = np.zeros((2, 2))
        self.__slen1: np.ndarray = np.zeros((2, 2))
        self.__slen2: np.ndarray = np.zeros((2, 2))
        self.__window_switching: np.ndarray = np.zeros((2, 2))
        self.__block_type: np.ndarray = np.zeros((2, 2))
        self.__mixed_block_flag: np.ndarray = np.zeros((2, 2))
        self.__switch_point_l: np.ndarray = np.zeros((2, 2))
        self.__switch_point_s: np.ndarray = np.zeros((2, 2))
        self.__table_select: np.ndarray = np.zeros((2, 2, 3))
        self.__sub_block_gain: np.ndarray = np.zeros((2, 2, 3))
        self.__region0_count: np.ndarray = np.zeros((2, 2))
        self.__region1_count: np.ndarray = np.zeros((2, 2))
        self.__pre_flag: np.ndarray = np.zeros((2, 2))
        self.__scale_fac_scale: np.ndarray = np.zeros((2, 2))
        self.__count1table_select: np.ndarray = np.zeros((2, 2))

        self.__scale_fac_l: np.ndarray = np.zeros((2, 2, 22))
        self.__scale_fac_s: np.ndarray = np.zeros((2, 2, 3, 13))

    def set_side_info(self, buffer: list, header: FrameHeader):
        """
        The side information contains information on how to decode the main_data.

        :param buffer: buffer that contains the mp3 file, from the first byte of the side info.
        :param header: The frame header.
        """

        offset = 0

        # Get main data begin pointer from buffer
        self.__main_data_begin = get_bits(buffer, 0, 9)
        offset += 9
        # Skip private bits
        offset += 5 if header.channel_mode == ChannelMode.Mono else 3

        # Scale factor selection info:
        # If scfsi[scfsi_band] == 1, then scale factors for 1st granule are reused in the 2nd granule.
        # Else, each granule has its own scale factors.
        # scfsi_band indicates what group of scaling factors are reused (1-4)
        for ch in range(header.channels):
            for scfsi_band in range(4):
                self.__scfsi[ch][scfsi_band] = get_bits(buffer, offset, 1) != 0
                offset += 1

        for gr in range(2):
            for ch in range(header.channels):
                # Length of scaling factors and main data in bits.
                self.__part2_3_length[gr][ch] = get_bits(buffer, offset, 12)
                offset += 12
                # Number of values is each big_region.
                self.__big_value[gr][ch] = get_bits(buffer, offset, 9)
                offset += 9
                # Quantizer step size.
                self.__global_gain[gr][ch] = get_bits(buffer, offset, 8)
                offset += 8
                # Used to determine the values of slen1 and slen2.
                self.__scale_fac_compress[gr][ch] = get_bits(buffer, offset, 4)
                offset += 4
                # Number of bits given to a range of scale factors.
                # - Normal blocks: slen1 0 - 10, slen2 11-20
                # - Short blocks: Short blocks && mixed_block_flag == 1: slen1 0 - 5, slen2 6-11
                # - Short blocks && mixed_block_flag == 0:
                self.__slen1[gr][ch] = slen[int(self.__scale_fac_compress[gr][ch])][0]
                self.__slen2[gr][ch] = slen[int(self.__scale_fac_compress[gr][ch])][1]
                # If set, a not normal window is being used.
                self.__window_switching[gr][ch] = get_bits(buffer, offset, 1) == 1
                offset += 1

                if self.__window_switching[gr][ch]:
                    # Window type for the granule: 0=reserved, 1=start block, 2=3 short blocks, 3=end block
                    self.__block_type[gr][ch] = get_bits(buffer, offset, 2)
                    offset += 2
                    # Number of scale factor bands before window switching.
                    self.__mixed_block_flag[gr][ch] = get_bits(buffer, offset, 1) == 1
                    offset += 1
                    if self.__mixed_block_flag[gr][ch]:
                        self.__switch_point_l[gr][ch] = 8
                        self.__switch_point_s[gr][ch] = 3

                    # These are set by default if window_switching is on.
                    self.__region0_count[gr][ch] = 8 if self.__block_type[gr][ch] == 2 else 7
                    # No third region
                    self.__region1_count[gr][ch] = 20 - self.__region0_count[gr][ch]

                    for region in range(2):
                        # Huffman table number for a big region
                        self.__table_select[gr][ch][region] = get_bits(buffer, offset, 5)
                        offset += 5
                    for window in range(3):
                        self.__sub_block_gain[gr][ch][window] = get_bits(buffer, offset, 3)
                        offset += 3

                else:
                    # Set by default if window_switching not set.
                    self.__block_type[gr][ch] = 0
                    self.__mixed_block_flag[gr][ch] = False

                    for region in range(3):
                        self.__table_select[gr][ch][region] = get_bits(buffer, offset, 5)
                        offset += 5

                    # Number of scale factor bands in the first big value region.
                    self.__region0_count[gr][ch] = get_bits(buffer, offset, 4)
                    offset += 4
                    # Number of scale factor bands in the third big value region.
                    self.__region1_count[gr][ch] = get_bits(buffer, offset, 3)
                    offset += 3
                    # scale factor bands is 12*3 = 36

                # if set, adds values from a table to the scaling factor
                self.__pre_flag[gr][ch] = get_bits(buffer, offset, 1)
                offset += 1
                # Determines the step size.
                self.__scale_fac_scale[gr][ch] = get_bits(buffer, offset, 1)
                offset += 1
                # Table that determines which count1 table is used.
                self.__count1table_select[gr][ch] = get_bits(buffer, offset, 1)
                offset += 1

    @property
    def main_data_begin(self):
        return self.__main_data_begin

    @property
    def scfsi(self):
        return self.__scfsi

    @property
    def part2_3_length(self):
        return self.__part2_3_length

    @property
    def part2_length(self):
        return self.__part2_length

    @property
    def big_value(self):
        return self.__big_value

    @property
    def global_gain(self):
        return self.__global_gain

    @property
    def scale_fac_compress(self):
        return self.__scale_fac_compress

    @property
    def slen1(self):
        return self.__slen1

    @property
    def slen2(self):
        return self.__slen2

    @property
    def window_switching(self):
        return self.__window_switching

    @property
    def block_type(self):
        return self.__block_type

    @property
    def mixed_block_flag(self):
        return self.__mixed_block_flag

    @property
    def switch_point_l(self):
        return self.__switch_point_l

    @property
    def switch_point_s(self):
        return self.__switch_point_s

    @property
    def table_select(self):
        return self.__table_select

    @property
    def sub_block_gain(self):
        return self.__sub_block_gain

    @property
    def region0_count(self):
        return self.__region0_count

    @property
    def region1_count(self):
        return self.__region1_count

    @property
    def pre_flag(self):
        return self.__pre_flag

    @property
    def scale_fac_scale(self):
        return self.__scale_fac_scale

    @property
    def count1table_select(self):
        return self.__count1table_select

    @property
    def scale_fac_l(self):
        return self.__scale_fac_l

    @property
    def scale_fac_s(self):
        return self.__scale_fac_s
