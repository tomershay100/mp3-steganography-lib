import math
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from mp3stego.encoder import tables
from mp3stego.encoder import util
from mp3stego.encoder.WAV_Reader import WavReader


@dataclass
class Subband:
    off: []
    fl: []
    x: []

    def __init__(self):
        self.off = np.zeros(util.MAX_CHANNELS, dtype=np.int32)
        self.fl = np.zeros((util.SBLIMIT, 64), dtype=np.int32)
        self.x = np.zeros((util.MAX_CHANNELS, util.HAN_SIZE), dtype=np.int32)


@dataclass
class MDCT:
    cos_l: []

    def __init__(self):
        self.cos_l = np.zeros((18, 36), dtype=np.int32)


@dataclass
class MPEG:
    version: int = 0
    layer: int = 0
    granules_per_frame: int = 0
    mode: int = 0  # Stereo mode
    bitrate: int = 0
    emphasis: int = 0  # De-emphasis
    padding: int = 0
    bits_per_frame: int = 0
    bits_per_slot: int = 0
    frac_slots_per_frame: float = 0.0
    slot_lag: float = 0.0
    whole_slots_per_frame: int = 0
    mean_bits: int = 0
    bitrate_index: int = 0
    samplerate_index: int = 0
    crc: int = 0
    ext: int = 0
    mode_ext: int = 0
    copyright: int = 0
    original: int = 0


@dataclass
class BitstreamStruct:
    data: []  # Processed data
    data_size: int = 0  # Total data size
    data_position: int = 0  # Data position
    cache: int = 0  # bit stream cache
    cache_bits: int = 0  # free bits in cache

    def __init__(self, data_size, data_position, cache, cache_bits):
        self.data = np.zeros(data_size, dtype=np.uint8)
        self.data_size = data_size
        self.data_position = data_position
        self.cache = cache
        self.cache_bits = cache_bits


@dataclass
class GrInfo:
    table_select: []
    slen: []
    part2_3_length: int = 0
    big_values: int = 0
    count1: int = 0
    global_gain: int = 0
    scalefac_compress: int = 0
    region0_count: int = 0
    region1_count: int = 0
    preflag: int = 0
    scalefac_scale: int = 0
    count1table_select: int = 0
    part2_length: int = 0
    sfb_lmax: int = 0
    address1: int = 0
    address2: int = 0
    address3: int = 0
    quantizerStepSize: int = 0

    def __init__(self):
        self.table_select = np.zeros(3, dtype=np.int32)
        self.slen = np.zeros(4, dtype=np.int32)


@dataclass
class CH:
    tt: GrInfo

    def __init__(self):
        self.tt = GrInfo()


@dataclass
class GR:
    ch: []

    def __init__(self):
        self.ch = [CH() for _ in range(util.MAX_CHANNELS)]


@dataclass
class SideInfo:
    scfsi: []
    gr: []
    private_bits: int = 0
    resv_drain: int = 0

    def __init__(self):
        self.gr = [GR() for _ in range(util.MAX_GRANULES)]
        self.scfsi = np.zeros((util.MAX_CHANNELS, 4), dtype=np.int32)


@dataclass
class ScaleFactor:
    l: []  # [cb]
    s: []  # [window][cb]

    def __init__(self):
        self.l = np.zeros((util.MAX_GRANULES, util.MAX_CHANNELS, 22), dtype=np.int32)  # [cb]
        self.s = np.zeros((util.MAX_GRANULES, util.MAX_CHANNELS, 13, 3), dtype=np.int32)  # [window][cb]


@dataclass
class L3Loop:
    xr: np.ndarray  # a pointer of the magnitudes of the spectral values
    xrsq: []  # xr squared
    xrabs: []  # xr absolute
    xrmax: int  # maximum of xrabs array
    en_tot: []  # gr
    en: []
    xm: []
    xrmaxl: []
    steptab: []  # 2**(-x/4)  for x = -127..0
    steptabi: []  # 2**(-x/4)  for x = -127..0
    int2idx: []  # x**(3/4)   for x = 0..9999

    def __init__(self):
        self.xr = np.zeros(1)
        self.xrsq = np.zeros(util.GRANULE_SIZE, dtype=np.int32)
        self.xrabs = np.zeros(util.GRANULE_SIZE, dtype=np.int32)
        self.en_tot = np.zeros(util.MAX_GRANULES, dtype=np.int32)
        self.en = np.zeros((util.MAX_GRANULES, 21), dtype=np.int32)
        self.xm = np.zeros((util.MAX_GRANULES, 21), dtype=np.int32)
        self.xrmaxl = np.zeros(util.MAX_GRANULES, dtype=np.int32)
        self.steptab = np.zeros(128, dtype=np.double)
        self.steptabi = np.zeros(128, dtype=np.int32)
        self.int2idx = np.zeros(10000, dtype=np.int32)


def count1_bitcount(ix, cod_info):
    """
    Determines the number of bits to encode the quadruples.
    :param ix: self.__l3_enc[ch][gr], vector of quantized values ix(0..575)
    :param cod_info: self.__side_info.gr[gr].ch[ch].tt
    :return:
    """
    i = cod_info.big_values << 1
    sum0 = 0
    sum1 = 0

    for k in range(cod_info.count1):
        v = ix[i]
        w = ix[i + 1]
        x = ix[i + 2]
        y = ix[i + 3]

        p = v + (w << 1) + (x << 2) + (y << 3)

        signbits = 0
        signbits += (v != 0)
        signbits += (w != 0)
        signbits += (x != 0)
        signbits += (y != 0)

        sum0 += signbits
        sum1 += signbits

        sum0 += tables.huffman_table[32].hlen[p]
        sum1 += tables.huffman_table[33].hlen[p]

        i += 4

    if sum0 < sum1:
        cod_info.count1table_select = 0
        return sum0
    else:
        cod_info.count1table_select = 1
        return sum1


# Count the number of bits necessary to code the subregion.
def count_bit(ix, start, end, table):
    """
    :param ix: self.__l3_enc[ch][gr], vector of quantized values ix(0..575)
    :param start:
    :param end:
    :param table:
    :return:
    """
    if table == 0:
        return 0

    huf_table = tables.huffman_table[table]
    h_sum = 0
    ylen = huf_table.ylen
    linbits = huf_table.linbits

    if table > 15:  # ESC-table is used
        for i in range(start, end, 2):
            x = ix[i]
            y = ix[i + 1]
            if x > 14:
                x = 15
                h_sum += linbits
            if y > 14:
                y = 15
                h_sum += linbits

            h_sum += huf_table.hlen[(x * ylen) + y]
            if x:
                h_sum += 1
            if y:
                h_sum += 1

    else:  # No ESC-words
        for i in range(start, end, 2):
            x = ix[i]
            y = ix[i + 1]

            h_sum += huf_table.hlen[(x * ylen) + y]

            if x != 0:
                h_sum += 1
            if y != 0:
                h_sum += 1

    return h_sum


def calc_runlen(ix, cod_info):
    """
    Calculation of rzero, count1, big_values (partitions ix into big values, quadruples and zeros).
    :param ix: self.__l3_enc[ch][gr], vector of quantized values ix(0..575)
    :param cod_info: self.__side_info.gr[gr].ch[ch].tt
    """
    rzero = 0
    i = util.GRANULE_SIZE

    while i > 1:
        if ix[i - 1] == 0 and ix[i - 2] == 0:
            rzero += 1
            i -= 2
        else:
            break

    cod_info.count1 = 0
    while i > 3:
        if ix[i - 1] <= 1 and ix[i - 2] <= 1 and ix[i - 3] <= 1 and ix[i - 4] <= 1:
            cod_info.count1 += 1
            i -= 4
        else:
            break

    cod_info.big_values = np.right_shift(i, 1)


def bigv_bitcount(ix, cod_info):
    """
    Count the number of bits necessary to code the bigvalues region.
    :param ix: self.__l3_enc[ch][gr], vector of quantized values ix(0..575)
    :param cod_info: self.__side_info.gr[gr].ch[ch].tt
    :return:
    """
    bits = 0

    table = cod_info.table_select[0]
    if table:
        bits += count_bit(ix, 0, cod_info.address1, table)
    table = cod_info.table_select[1]
    if table:
        bits += count_bit(ix, cod_info.address1, cod_info.address2, table)
    table = cod_info.table_select[2]
    if table:
        bits += count_bit(ix, cod_info.address2, cod_info.address3, table)

    return bits


# Tables 0 and 14 are not used.
idx_to_transform_idx = {(1, 1): 1, (1, 0): 3,
                        (2, 1): 2, (2, 0): 3,
                        (3, 1): 2, (3, 0): 3,
                        (5, 1): 5, (5, 0): 6,
                        (6, 1): 5, (6, 0): 6,
                        (7, 1): 7, (7, 0): 8,
                        (8, 1): 7, (8, 0): 8,
                        (9, 1): 9, (9, 0): 8,
                        (10, 1): 10, (10, 0): 11,
                        (11, 1): 10, (11, 0): 11,
                        (12, 1): 10, (12, 0): 12,
                        (13, 1): 13, (13, 0): 15,
                        (15, 1): 13, (15, 0): 15,
                        (16, 1): 16, (16, 0): 17,
                        (17, 1): 18, (17, 0): 17,
                        (18, 1): 18, (18, 0): 19,
                        (19, 1): 20, (19, 0): 19,
                        (20, 1): 20, (20, 0): 21,
                        (21, 1): 22, (21, 0): 21,
                        (22, 1): 22, (22, 0): 23,
                        (23, 1): 31, (23, 0): 23,
                        (24, 1): 25, (24, 0): 24,
                        (25, 1): 25, (25, 0): 26,
                        (26, 1): 27, (26, 0): 26,
                        (27, 1): 27, (27, 0): 28,
                        (28, 1): 29, (28, 0): 28,
                        (29, 1): 29, (29, 0): 30,
                        (30, 1): 31, (30, 0): 30,
                        (31, 1): 31, (31, 0): 23,
                        }


class MP3Encoder:
    """
    Class for encoding wav file into mp3 file.

    :param wav_file: a WavReader that contains the wav file data.
    :type wav_file: WavReader
    :param hide_str: if is not empty, hides the string inside the output mp3 file.
    :type hide_str: str
    """

    def __init__(self, wav_file: WavReader, hide_str: str = ""):
        self.__wav_file: WavReader = wav_file
        # Compute default encoding values.
        self.__ratio: np.ndarray = np.zeros((util.MAX_GRANULES, util.MAX_CHANNELS, 21), dtype=np.double)
        self.__scalefactor: ScaleFactor = ScaleFactor()
        self.__pe: np.ndarray = np.zeros((util.MAX_CHANNELS, util.MAX_GRANULES), dtype=np.double)
        self.__l3_enc: np.ndarray = np.zeros((util.MAX_CHANNELS, util.MAX_GRANULES, util.GRANULE_SIZE), dtype=np.int32)
        self.__l3_sb_sample: np.ndarray = np.zeros((util.MAX_CHANNELS, util.MAX_GRANULES + 1, 18, util.SBLIMIT),
                                                   dtype=np.int32)
        self.__mdct_freq: np.ndarray = np.zeros((util.MAX_CHANNELS, util.MAX_GRANULES, util.GRANULE_SIZE),
                                                dtype=np.int32)
        self.__l3loop: L3Loop = L3Loop()
        self.__mdct: MDCT = MDCT()
        self.__subband: Subband = Subband()
        self.__side_info: SideInfo = SideInfo()
        self.__mpeg: MPEG = MPEG()

        self.__subband_initialise()
        self.__mdct_initialise()
        self.__loop_initialise()

        self.__mpeg.mode = wav_file.mpeg_mode
        self.__mpeg.bitrate = wav_file.bitrate
        self.__mpeg.emphasis = wav_file.emphasis
        self.__mpeg.copyright = wav_file.copyright
        self.__mpeg.original = wav_file.original

        #  Set default values.
        self.__resv_max: int = 0
        self.__resv_size: int = 0
        self.__mpeg.layer = 1  # Only Layer III currently implemented.
        self.__mpeg.crc = 0
        self.__mpeg.ext = 0
        self.__mpeg.mode_ext = 0
        self.__mpeg.bits_per_slot = 8

        self.__mpeg.samplerate_index = util.find_samplerate_index(wav_file.samplerate)
        self.__mpeg.version = util.find_mpeg_version(self.__mpeg.samplerate_index)
        self.__mpeg.bitrate_index = util.find_bitrate_index(self.__mpeg.bitrate, self.__mpeg.version)
        self.__mpeg.granules_per_frame = util.GRANULES_PER_FRAME[self.__mpeg.version]

        # Figure average number of 'slots' per frame.
        avg_slots_per_frame = (np.double(self.__mpeg.granules_per_frame) * util.GRANULES_SIZE / (np.double(
            wav_file.samplerate))) * (1000 * np.double(self.__mpeg.bitrate) / np.double(self.__mpeg.bits_per_slot))

        self.__mpeg.whole_slots_per_frame = int(avg_slots_per_frame)

        self.__mpeg.frac_slots_per_frame = avg_slots_per_frame - np.double(self.__mpeg.whole_slots_per_frame)
        self.__mpeg.slot_lag = - self.__mpeg.frac_slots_per_frame

        if self.__mpeg.frac_slots_per_frame == 0:
            self.__mpeg.padding = 0

        self.__bitstream: BitstreamStruct = BitstreamStruct(util.BUFFER_SIZE, 0, 0, 32)

        # determine the mean bitrate for main data
        if self.__mpeg.granules_per_frame == 2:  # MPEG 1
            self.__side_info_len: int = 8 * ((4 + 17) if wav_file.num_of_channels == 1 else (4 + 32))
        else:  # MPEG 2
            self.__side_info_len: int = 8 * ((4 + 9) if wav_file.num_of_channels == 1 else (4 + 17))

        self.__out_buffer: bytearray = bytearray('', "utf-8")

        self.__hide_str: str = hide_str
        self.__hide_str_offset: int = 0

    def __subband_initialise(self):
        for i in range(util.MAX_CHANNELS - 1, -1, -1):
            self.__subband.off[i] = 0
            self.__subband.x[i][:] = 0

        for i in range(util.SBLIMIT - 1, -1, -1):
            for j in range(64 - 1, -1, -1):
                filter = 1e9 * math.cos(np.double(((2 * i + 1) * (16 - j) * util.PI64)))
                if filter >= 0:
                    filter = math.modf(filter + 0.5)[1]
                else:
                    filter = math.modf(filter - 0.5)[1]
                # scale and convert to fixed point before storing
                self.__subband.fl[i][j] = np.int32(filter * 0x7fffffff * 1e-9)

    def __mdct_initialise(self):
        # prepare the mdct coefficients
        for m in range(18 - 1, -1, -1):
            for k in range(36 - 1, -1, -1):
                # combine window and mdct coefficients into a single table
                # scale and convert to fixed point before storing
                self.__mdct.cos_l[m][k] = np.int32(math.sin(util.PI36 * (k + 0.5)) * math.cos(
                    (util.PI / 72) * (2 * k + 19) * (2 * m + 1)) * 0x7fffffff)

    # Calculates the look up tables used by the iteration loop.
    def __loop_initialise(self):
        # quantize: stepsize conversion, fourth root of 2 table.
        # The table is inverted (negative power) from the equation given
        # in the spec because it is quicker to do x*y than x/y.
        # The 0.5 is for rounding.
        for i in range(128 - 1, -1, -1):
            self.__l3loop.steptab[i] = 2.0 ** (np.double(127 - i) / 4)
            if self.__l3loop.steptab[i] * 2 > 0x7fffffff:  # MAXINT = 2**31 = 2**(124/4)
                self.__l3loop.steptabi[i] = 0x7fffffff
            else:
                # The table is multiplied by 2 to give an extra bit of accuracy.
                # In quantize, the long multiply does not shift it's result left one
                # bit to compensate.
                self.__l3loop.steptabi[i] = np.int32(self.__l3loop.steptab[i] * 2 + 0.5)

        # quantize: vector conversion, three quarter power table.
        # The 0.5 is for rounding, the .0946 comes from the spec.
        for i in range(10000 - 1, -1, -1):
            self.__l3loop.int2idx[i] = np.int32(math.sqrt(math.sqrt(np.double(i)) * np.double(i)) - 0.0946 + 0.5)

    def print_info(self):
        """
        Print some info about the file about to be created
        """
        version_names = ["2.5", "reserved", "II", "I"]
        mode_names = ["stereo", "joint-stereo", "dual-channel", "mono"]
        demp_names = ["none", "50/15us", "", "CITT"]

        print(f"MPEG-{version_names[self.__mpeg.version]} layer III, {mode_names[self.__mpeg.mode]}"
              f" Psychoacoustic Model: Shine")
        print(f"Bitrate: {self.__mpeg.bitrate} kbps ", end='')
        print(f"De-emphasis: {demp_names[self.__mpeg.emphasis]}\t{'Original' if self.__mpeg.original else ''}\t"
              f"{'(C)' if self.__mpeg.copyright else ''}")
        print(f"Encoding \"{self.__wav_file.file_path}\" to \"{self.__wav_file.file_path[:-3]}mp3\"\n")

    def encode(self):
        """
        Encoding the wav file into mp3 file, frame by frame and saves the output bytes of the mp3 file.
        Also, writes the mp3 file.
        """
        samples_per_pass = self.__samples_per_pass() * self.__wav_file.num_of_channels

        # All the magic happens here
        total_sample_count = self.__wav_file.num_of_samples * self.__wav_file.num_of_channels
        count = total_sample_count // samples_per_pass

        for _ in tqdm(range(count), desc='encoding'):
            written, data = self.__encode_buffer_internal()
            self.__out_buffer += bytearray(data[:written])

        last = total_sample_count % samples_per_pass
        if last != 0:
            written, data = self.__encode_buffer_internal()
            self.__out_buffer += bytearray(data[:written])

        # Flush and write remaining data.
        written, data = self.__flush()
        self.__out_buffer += bytearray(data[:written])

    def __samples_per_pass(self):
        return self.__mpeg.granules_per_frame * util.GRANULE_SIZE

    def __encode_buffer_internal(self):
        if self.__mpeg.frac_slots_per_frame:
            self.__mpeg.padding = (1 if self.__mpeg.slot_lag <= (self.__mpeg.frac_slots_per_frame - 1.0) else 0)
            self.__mpeg.slot_lag += self.__mpeg.padding - self.__mpeg.frac_slots_per_frame

        self.__mpeg.bits_per_frame = 8 * (self.__mpeg.whole_slots_per_frame + self.__mpeg.padding)
        self.__mpeg.mean_bits = int(
            (self.__mpeg.bits_per_frame - self.__side_info_len) / self.__mpeg.granules_per_frame)

        # apply mdct to the polyphase output
        self.__mdct_sub()

        # bit and noise allocation
        self.__iteration_loop()

        # write the frame to the bitstream
        self.__format_bitstream()

        written = self.__bitstream.data_position
        self.__bitstream.data_position = 0

        return written, self.__bitstream.data

    def __mdct_sub(self):
        # note. we wish to access the array 'config.mdct_freq[2][2][576]' as
        # [2][2][32][18]. (32*18=576),
        self.__mdct_freq = self.__mdct_freq.reshape((2, 2, 32, 18))
        mdct_in = np.zeros(36, dtype=np.int32)

        for ch in range(self.__wav_file.num_of_channels - 1, -1, -1):
            for gr in range(self.__mpeg.granules_per_frame):
                # set up pointer to the part of config.mdct_freq we're using
                # mdct_enc = self.__mdct_freq[ch][gr]

                # polyphase filtering
                for k in range(0, 18, 2):
                    self.__l3_sb_sample[ch, gr + 1, k, :] = self.__window_filter_subband(
                        self.__l3_sb_sample[ch, gr + 1, k, :], ch)
                    self.__l3_sb_sample[ch, gr + 1, k + 1, :] = self.__window_filter_subband(
                        self.__l3_sb_sample[ch, gr + 1, k + 1, :], ch)

                    # Compensate for inversion in the analysis filter
                    # (every odd index of band AND k)
                    for band in range(1, 32, 2):
                        self.__l3_sb_sample[ch][gr + 1][k + 1][band] *= -1

                # Perform imdct of 18 previous subband samples + 18 current subband samples
                for band in range(32):
                    for k in range(18 - 1, -1, -1):
                        mdct_in[k] = self.__l3_sb_sample[ch][gr][k][band]
                        mdct_in[k + 18] = self.__l3_sb_sample[ch][gr + 1][k][band]

                    # Calculation of the MDCT
                    # In the case of long blocks ( block_type 0,1,3 ) there are
                    # 36 coefficients in the time domain and 18 in the frequency domain.
                    for k in range(18 - 1, -1, -1):
                        vm = util.mul(mdct_in[35], self.__mdct.cos_l[k][35])
                        for j in range(35, 0, -7):
                            vm += util.mul(mdct_in[j - 1], self.__mdct.cos_l[k][j - 1])
                            vm += util.mul(mdct_in[j - 2], self.__mdct.cos_l[k][j - 2])
                            vm += util.mul(mdct_in[j - 3], self.__mdct.cos_l[k][j - 3])
                            vm += util.mul(mdct_in[j - 4], self.__mdct.cos_l[k][j - 4])
                            vm += util.mul(mdct_in[j - 5], self.__mdct.cos_l[k][j - 5])
                            vm += util.mul(mdct_in[j - 6], self.__mdct.cos_l[k][j - 6])
                            vm += util.mul(mdct_in[j - 7], self.__mdct.cos_l[k][j - 7])
                        self.__mdct_freq[ch][gr][band][k] = vm

                    # Perform aliasing reduction butterfly
                    if band != 0:
                        self.__mdct_freq[ch][gr][band][0], self.__mdct_freq[ch][gr][band - 1][17 - 0] = util.cmuls(
                            self.__mdct_freq[ch][gr][band][0], self.__mdct_freq[ch][gr][band - 1][17 - 0],
                            tables.MDCT_CS0, tables.MDCT_CA0)
                        self.__mdct_freq[ch][gr][band][1], self.__mdct_freq[ch][gr][band - 1][17 - 1] = util.cmuls(
                            self.__mdct_freq[ch][gr][band][1], self.__mdct_freq[ch][gr][band - 1][17 - 1],
                            tables.MDCT_CS1, tables.MDCT_CA1)
                        self.__mdct_freq[ch][gr][band][2], self.__mdct_freq[ch][gr][band - 1][17 - 2] = util.cmuls(
                            self.__mdct_freq[ch][gr][band][2], self.__mdct_freq[ch][gr][band - 1][17 - 2],
                            tables.MDCT_CS2, tables.MDCT_CA2)
                        self.__mdct_freq[ch][gr][band][3], self.__mdct_freq[ch][gr][band - 1][17 - 3] = util.cmuls(
                            self.__mdct_freq[ch][gr][band][3], self.__mdct_freq[ch][gr][band - 1][17 - 3],
                            tables.MDCT_CS3, tables.MDCT_CA3)
                        self.__mdct_freq[ch][gr][band][4], self.__mdct_freq[ch][gr][band - 1][17 - 4] = util.cmuls(
                            self.__mdct_freq[ch][gr][band][4], self.__mdct_freq[ch][gr][band - 1][17 - 4],
                            tables.MDCT_CS4, tables.MDCT_CA4)
                        self.__mdct_freq[ch][gr][band][5], self.__mdct_freq[ch][gr][band - 1][17 - 5] = util.cmuls(
                            self.__mdct_freq[ch][gr][band][5], self.__mdct_freq[ch][gr][band - 1][17 - 5],
                            tables.MDCT_CS5, tables.MDCT_CA5)
                        self.__mdct_freq[ch][gr][band][6], self.__mdct_freq[ch][gr][band - 1][17 - 6] = util.cmuls(
                            self.__mdct_freq[ch][gr][band][6], self.__mdct_freq[ch][gr][band - 1][17 - 6],
                            tables.MDCT_CS6, tables.MDCT_CA6)
                        self.__mdct_freq[ch][gr][band][7], self.__mdct_freq[ch][gr][band - 1][17 - 7] = util.cmuls(
                            self.__mdct_freq[ch][gr][band][7], self.__mdct_freq[ch][gr][band - 1][17 - 7],
                            tables.MDCT_CS7, tables.MDCT_CA7)

            # Save latest granule's subband samples to be used in the next mdct call
            self.__l3_sb_sample[ch, 0, :, :] = self.__l3_sb_sample[ch, self.__mpeg.granules_per_frame, :, :]

        self.__mdct_freq = self.__mdct_freq.reshape((util.MAX_CHANNELS, util.MAX_GRANULES, util.GRANULE_SIZE))

    def __window_filter_subband(self, s, ch):
        """
        :param s: self.__l3_sb_sample[ch, gr + 1, k, :]
        :param ch: the channel
        :return:
        """
        y = np.zeros(64, dtype=np.int32)
        # replace 32 oldest samples with 32 new samples
        for i in range(32 - 1, -1, -1):
            self.__subband.x[ch][i + self.__subband.off[ch]] = np.int32(
                self.__wav_file.buffer[self.__wav_file.get_buffer_pos(ch)]) << 16
            self.__wav_file.set_buffer_pos(ch, 2)

        for i in range(64 - 1, -1, -1):
            s_value = util.mul(self.__subband.x[ch][(self.__subband.off[ch] + i + (0 << 6)) & (util.HAN_SIZE - 1)],
                               tables.enwindow[i + (0 << 6)])
            s_value += util.mul(self.__subband.x[ch][(self.__subband.off[ch] + i + (1 << 6)) & (util.HAN_SIZE - 1)],
                                tables.enwindow[i + (1 << 6)])
            s_value += util.mul(self.__subband.x[ch][(self.__subband.off[ch] + i + (2 << 6)) & (util.HAN_SIZE - 1)],
                                tables.enwindow[i + (2 << 6)])
            s_value += util.mul(self.__subband.x[ch][(self.__subband.off[ch] + i + (3 << 6)) & (util.HAN_SIZE - 1)],
                                tables.enwindow[i + (3 << 6)])
            s_value += util.mul(self.__subband.x[ch][(self.__subband.off[ch] + i + (4 << 6)) & (util.HAN_SIZE - 1)],
                                tables.enwindow[i + (4 << 6)])
            s_value += util.mul(self.__subband.x[ch][(self.__subband.off[ch] + i + (5 << 6)) & (util.HAN_SIZE - 1)],
                                tables.enwindow[i + (5 << 6)])
            s_value += util.mul(self.__subband.x[ch][(self.__subband.off[ch] + i + (6 << 6)) & (util.HAN_SIZE - 1)],
                                tables.enwindow[i + (6 << 6)])
            s_value += util.mul(self.__subband.x[ch][(self.__subband.off[ch] + i + (7 << 6)) & (util.HAN_SIZE - 1)],
                                tables.enwindow[i + (7 << 6)])

            y[i] = s_value

        self.__subband.off[ch] = (self.__subband.off[ch] + 480) & (util.HAN_SIZE - 1)  # offset is modulo (HAN_SIZE)

        for i in range(util.SBLIMIT - 1, -1, -1):
            s_value = util.mul(self.__subband.fl[i][63], y[63])
            for j in range(63, 0, -7):
                s_value += util.mul(self.__subband.fl[i][j - 1], y[j - 1])
                s_value += util.mul(self.__subband.fl[i][j - 2], y[j - 2])
                s_value += util.mul(self.__subband.fl[i][j - 3], y[j - 3])
                s_value += util.mul(self.__subband.fl[i][j - 4], y[j - 4])
                s_value += util.mul(self.__subband.fl[i][j - 5], y[j - 5])
                s_value += util.mul(self.__subband.fl[i][j - 6], y[j - 6])
                s_value += util.mul(self.__subband.fl[i][j - 7], y[j - 7])
            s[i] = s_value

        return s

    def __iteration_loop(self):
        """
        bit and noise allocation
        """
        l3_xmin = np.zeros((util.MAX_GRANULES, util.MAX_CHANNELS, 21), dtype=np.double)
        for ch in range(self.__wav_file.num_of_channels):
            for gr in range(self.__mpeg.granules_per_frame):
                # setup pointers
                ix = self.__l3_enc[ch][gr]
                self.__l3loop.xr = self.__mdct_freq[ch][gr]

                # Precalculate the square, abs, and maximum, for us later on.
                self.__l3loop.xrmax = 0
                for i in range(util.GRANULE_SIZE - 1, -1, -1):
                    self.__l3loop.xrsq[i] = util.mulsr(self.__l3loop.xr[i], self.__l3loop.xr[i])
                    self.__l3loop.xrabs[i] = util.labs(self.__l3loop.xr[i])
                    if self.__l3loop.xrabs[i] > self.__l3loop.xrmax:
                        self.__l3loop.xrmax = self.__l3loop.xrabs[i]

                cod_info = self.__side_info.gr[gr].ch[ch].tt
                cod_info.sfb_lmax = util.SFB_LMAX - 1  # gr_deco

                l3_xmin[gr, ch, 0:cod_info.sfb_lmax] = 0

                if self.__mpeg.version == util.MPEG_VERSIONS["MPEG_I"]:
                    self.__calc_scfsi(l3_xmin, ch, gr)

                # calculation of number of available bit( per granule )
                max_bits = self.__max_reservoir_bits(ch, gr)

                # reset of iteration variables
                self.__scalefactor.l[gr][ch] = 0
                self.__scalefactor.s[gr][ch] = 0
                cod_info.slen[:] = 0
                cod_info.part2_3_length = 0
                cod_info.big_values = 0
                cod_info.count1 = 0
                cod_info.scalefac_compress = 0
                cod_info.table_select[0] = 0
                cod_info.table_select[1] = 0
                cod_info.table_select[2] = 0
                cod_info.region0_count = 0
                cod_info.region1_count = 0
                cod_info.part2_length = 0
                cod_info.preflag = 0
                cod_info.scalefac_scale = 0
                cod_info.count1table_select = 0

                # all spectral values zero
                if self.__l3loop.xrmax:
                    cod_info.part2_3_length = self.__shine_outer_loop(max_bits, ix, gr, ch)
                    self.__hide_str_offset += int(cod_info.table_select[0] > 0) + int(
                        cod_info.table_select[1] > 0) + int(cod_info.table_select[2] > 0)

                # Re-adjust the size of the reservoir to reflect the granule's usage.
                self.__resv_size += (self.__mpeg.mean_bits / self.__wav_file.num_of_channels) - cod_info.part2_3_length
                cod_info.global_gain = cod_info.quantizerStepSize + 210

        self.__resv_frame_end()

    def __calc_scfsi(self, l3_xmin, ch, gr):
        """
        calculation of the scalefactor select information (scfsi)
        :param l3_xmin:
        :param ch: the channel
        :param gr: the granule
        """
        l3_side = self.__side_info

        # This is the scfsi_band table from 2.4.2.7 of the IS
        scfsi_band_long = [0, 6, 11, 16, 21]
        condition = 0

        scalefac_band_long = util.scale_fact_band_index[self.__mpeg.samplerate_index]

        self.__l3loop.xrmaxl[gr] = self.__l3loop.xrmax
        scfsi_set = 0

        # the total energy of the granule
        temp = 0
        for i in range(util.GRANULE_SIZE - 1, -1, -1):
            temp += np.right_shift(self.__l3loop.xrsq[i], 10)  # a bit of scaling to avoid overflow

        if temp:
            self.__l3loop.en_tot[gr] = np.log(np.double(temp * 4.768371584e-7)) / util.LN2
        else:
            self.__l3loop.en_tot[gr] = 0

        # The energy of each scalefactor band, en
        # The allowed distortion of each scalefactor band, xm
        for sfb in range(21 - 1, -1, -1):
            start = scalefac_band_long[sfb]
            end = scalefac_band_long[sfb + 1]

            temp = 0
            for i in range(start, end):
                temp += np.right_shift(self.__l3loop.xrsq[i], 10)
            if temp:
                self.__l3loop.en[gr][sfb] = np.log(np.double(temp * 4.768371584e-7)) / util.LN2
            else:
                self.__l3loop.en[gr][sfb] = 0

            if l3_xmin[gr][ch][sfb]:
                self.__l3loop.xm[gr][sfb] = np.log(l3_xmin[gr][ch][sfb]) / util.LN2
            else:
                self.__l3loop.xm[gr][sfb] = 0

        if gr == 1:
            for gr2 in range(2 - 1, -1, -1):
                if self.__l3loop.xrmaxl[gr2]:
                    condition += 1
                condition += 1

            if abs(self.__l3loop.en_tot[0] - self.__l3loop.en_tot[1]) < util.en_tot_krit:
                condition += 1
            tp = 0
            for sfb in range(21 - 1, -1, -1):
                tp += abs(self.__l3loop.en[0][sfb] - self.__l3loop.en[1][sfb])
            if tp < util.en_dif_krit:
                condition += 1

            if condition == 6:
                for scfsi_band in range(4):
                    sum0, sum1 = 0, 0
                    l3_side.scfsi[ch][scfsi_band] = 0
                    start = scfsi_band_long[scfsi_band]
                    end = scfsi_band_long[scfsi_band + 1]
                    for sfb in range(start, end):
                        sum0 += abs(self.__l3loop.en[0][sfb] - self.__l3loop.en[1][sfb])
                        sum1 += abs(self.__l3loop.xm[0][sfb] - self.__l3loop.xm[1][sfb])

                    if sum0 < util.en_scfsi_band_krit and sum1 < util.xm_scfsi_band_krit:
                        l3_side.scfsi[ch][scfsi_band] = 1
                        scfsi_set |= np.int32(np.left_shift(np.int32(1), scfsi_band))
                    else:
                        l3_side.scfsi[ch][scfsi_band] = 0

            else:
                l3_side.scfsi[ch, :] = 0

    def __max_reservoir_bits(self, ch, gr):
        """
        Called at the beginning of each granule to get the max bit
        allowance for the current granule based on reservoir size and perceptual entropy.
        :param ch: the channel
        :param gr: the granule
        """
        pe = self.__pe[ch][gr]

        mean_bits = self.__mpeg.mean_bits

        mean_bits //= self.__wav_file.num_of_channels
        max_bits = mean_bits

        if max_bits > 4095:
            max_bits = 4095
        if not self.__resv_max:
            return max_bits

        more_bits = int(pe[0] * 3.1 - mean_bits)
        add_bits = 0
        if more_bits > 100:
            frac = int((self.__resv_size * 6) / 10)
            if frac < more_bits:
                add_bits = frac
            else:
                add_bits = more_bits

        over_bits = int(self.__resv_size - ((self.__resv_max << 3) / 10) - add_bits)
        if over_bits > 0:
            add_bits += over_bits

        max_bits += add_bits
        if max_bits > 4095:
            max_bits = 4095

        return max_bits

    def __shine_outer_loop(self, max_bits, ix, gr, ch):
        """
        Function: The outer iteration loop controls the masking conditions of all scalefactorbands.
        It computes the best scalefac and global gain. This module calls the inner iteration loop.
        :param max_bits:
        :param ix: self.__l3_enc[ch][gr], vector of quantized values ix(0..575)
        :param gr: the granule
        :param ch: the channel
        """
        side_info = self.__side_info
        cod_info = side_info.gr[gr].ch[ch].tt

        cod_info.quantizerStepSize = self.__bin_search_step_size(max_bits, ix, cod_info)

        cod_info.part2_length = self.__part2_length(gr, ch)
        huff_bits = max_bits - cod_info.part2_length

        bits = self.__inner_loop(ix, huff_bits, cod_info)
        cod_info.part2_3_length = cod_info.part2_length + bits

        return cod_info.part2_3_length

    def __bin_search_step_size(self, desired_rate, ix, cod_info):
        """
        Successive approximation approach to obtaining a initial quantizer step size. When BIN_SEARCH is defined, the
        outer_loop function precedes the call to the function inner_loop with a call to bin_search gain defined below,
        which returns a good starting quantizer_step_size.
        :param desired_rate:
        :param ix: self.__l3_enc[ch][gr], vector of quantized values ix(0..575)
        :param cod_info: self.__l3_sb_sample[ch, gr + 1, k, :], ch
        """
        next = -120
        count = 120

        condition = True
        while condition:
            half = count // 2

            if self.__quantize(ix, next + half) > 8192:
                bit = 100000
            else:
                calc_runlen(ix, cod_info)  # rzero, count1, big_values
                bit = count1_bitcount(ix, cod_info)  # count1_table selection
                self.__subdivide(cod_info)  # bigvalues sfb division
                self.__bigv_tab_select(ix, cod_info)  # codebook selection
                bit += bigv_bitcount(ix, cod_info)  # bitcount

            if bit < desired_rate:
                count = half
            else:
                next += half
                count -= half

            # End of loop body
            condition = count > 1

        return next

    def __subdivide(self, cod_info):
        """
        Presumable subdivides the bigvalue region which will use separate Huffman tables.
        :param cod_info: self.__l3_sb_sample[ch, gr + 1, k, :], ch
        """
        if cod_info.big_values == 0:  # No big_values region
            cod_info.region0_count = 0
            cod_info.region1_count = 0
        else:
            temp_scale_fact_band_index = np.array(util.scale_fact_band_index)
            scalefac_band_long = self.__mpeg.samplerate_index * temp_scale_fact_band_index.shape[1]
            temp_scale_fact_band_index = temp_scale_fact_band_index.flatten()[scalefac_band_long:]
            bigvalues_region = 2 * cod_info.big_values

            # Calculate scfb_anz
            scfb_anz = 0
            while temp_scale_fact_band_index[scfb_anz] < bigvalues_region:
                scfb_anz += 1

            thiscount = tables.subdv_table[scfb_anz][0]
            while thiscount > 0:
                if temp_scale_fact_band_index[thiscount + 1] <= bigvalues_region:
                    break
                thiscount -= 1
            cod_info.region0_count = thiscount
            cod_info.address1 = temp_scale_fact_band_index[thiscount + 1]

            temp_scale_fact_band_index = temp_scale_fact_band_index[thiscount + 1:]

            thiscount = tables.subdv_table[scfb_anz][1]
            while thiscount > 0:
                if temp_scale_fact_band_index[thiscount + 1] <= bigvalues_region:
                    break
                thiscount -= 1
            cod_info.region1_count = thiscount
            cod_info.address2 = temp_scale_fact_band_index[thiscount + 1]

            cod_info.address3 = bigvalues_region

    def __quantize(self, ix, stepsize):
        """
        :param ix: self.__l3_enc[ch][gr], vector of quantized values ix(0..575)
        :param stepsize:
        :return:
        """
        ix_max = 0
        scalei = self.__l3loop.steptabi[stepsize + 127]  # 2**(-stepsize/4)

        # A quick check to see if ixmax will be less than 8192
        # This speeds up the early calls to bin_search_StepSize
        if util.mulr(self.__l3loop.xrmax, scalei) > 165140:  # 8192**(4/3)
            ix_max = 16384  # No point in continuing, step size not big enough
        else:
            for i in range(util.GRANULE_SIZE):
                # This calculation is very sensitive. The multiply must round
                # it's result or bad things happen to the quality.

                ln = util.mulr(util.labs(self.__l3loop.xr[i]), scalei)

                if ln < 10000:  # ln < 10000 catches most values
                    ix[i] = self.__l3loop.int2idx[ln]  # Quick lookup method
                else:
                    # Outside table range so have to do it using floats
                    scale = self.__l3loop.steptab[stepsize + 127]  # 2**(-stepsize/4)
                    dbl = np.double(self.__l3loop.xrabs[i]) * scale * 4.656612875e-10  # 0x7fffffff
                    ix[i] = int(np.sqrt(np.sqrt(dbl) * dbl))  # dbl**(3/4)

                # calculate ixmax while we're here. note: ix cannot be negative
                if ix_max < ix[i]:
                    ix_max = ix[i]

        return ix_max

    def __part2_length(self, gr, ch):
        """
        calculates the number of bits needed to encode the scalefacs in the main data block.
        :param gr: the granule
        :param ch: the channel
        """
        gi = self.__side_info.gr[gr].ch[ch].tt
        bits = 0

        slen1 = tables.slen1_tab[gi.scalefac_compress]
        slen2 = tables.slen2_tab[gi.scalefac_compress]

        if gr == 0 or self.__side_info.scfsi[ch][0] == 0:
            bits += 6 * slen1
        if gr == 0 or self.__side_info.scfsi[ch][1] == 0:
            bits += 5 * slen1
        if gr == 0 or self.__side_info.scfsi[ch][2] == 0:
            bits += 5 * slen2
        if gr == 0 or self.__side_info.scfsi[ch][3] == 0:
            bits += 5 * slen2

        return bits

    def __inner_loop(self, ix, max_bits, cod_info):
        """
        The code selects the best quantizer_step_size for a particular set of scalefacs.
        :param ix: self.__l3_enc[ch][gr], vector of quantized values ix(0..575)
        :param max_bits:
        :param cod_info: self.__side_info.gr[gr].ch[ch].tt
        :return:
        """
        bits = 0

        if max_bits < 0:
            cod_info.quantizerStepSize -= 1

        condition = True
        while condition:
            while self.__quantize(ix, cod_info.quantizerStepSize + 1) > 8192:  # within table range?
                cod_info.quantizerStepSize += 1
            cod_info.quantizerStepSize += 1

            calc_runlen(ix, cod_info)  # rzero,count1,big_values
            bits = count1_bitcount(ix, cod_info)  # count1_table selection
            self.__subdivide(cod_info)  # bigvalues sfb division
            self.__bigv_tab_select(ix, cod_info)  # codebook selection
            bits += bigv_bitcount(ix, cod_info)  # bit count

            condition = (bits > max_bits)

        return bits

    def __resv_frame_end(self):
        """
        Called after all granules in a frame have been allocated. Makes sure that the reservoir size is within limits,
        possibly by adding stuffing bits. Note that stuffing bits are added by increasing a granule's part2_3_length.
        The bitstream formatter will detect this and write the appropriate stuffing bits to the bitstream.
        """
        l3_side = self.__side_info

        ancillary_pad = 0

        # just in case mean_bits is odd this is necessary
        if self.__wav_file.num_of_channels == 2 and (self.__mpeg.mean_bits & 1):
            self.__resv_size += 1

        over_bits = self.__resv_size - self.__resv_max
        if over_bits < 0:
            over_bits = 0

        self.__resv_size -= over_bits
        stuffing_bits = over_bits + ancillary_pad

        # we must be byte aligned
        over_bits = self.__resv_size % 8
        if over_bits:
            stuffing_bits += over_bits
            self.__resv_size -= over_bits

        if stuffing_bits:

            gi = l3_side.gr[0].ch[0].tt

            if gi.part2_3_length + stuffing_bits < 4095:
                # plan a: put all into the first granule
                gi.part2_3_length += stuffing_bits
            else:
                # plan b: distribute throughout the granules
                for gr in range(self.__mpeg.granules_per_frame):
                    for ch in range(self.__wav_file.num_of_channels):
                        gi = l3_side.gr[gr].ch[ch].tt
                        if not stuffing_bits:
                            break
                        extra_bits = 4095 - gi.part2_3_length
                        bits_this_gr = extra_bits if extra_bits < stuffing_bits else stuffing_bits
                        gi.part2_3_length += bits_this_gr
                        stuffing_bits -= bits_this_gr

                # If any stuffing bits remain, we elect to spill them into ancillary data.
                # The bitstream formatter will do this if l3side.resv_drain is set
                l3_side.resv_drain = stuffing_bits

    def __bigv_tab_select(self, ix, cod_info):
        """
         Select huffman code tables for bigvalues regions
        :param ix: self.__l3_enc[ch][gr], vector of quantized values ix(0..575)
        :param cod_info: self.__side_info.gr[gr].ch[ch].tt
        """
        idx = self.__hide_str_offset

        cod_info.table_select[0] = 0 if cod_info.address1 <= 0 else self.__new_choose_table(ix, 0, cod_info.address1,
                                                                                            self.__hide_str_offset)
        if cod_info.table_select[0] > 0:
            idx += 1

        cod_info.table_select[1] = 0 if cod_info.address2 <= cod_info.address1 \
            else self.__new_choose_table(ix, cod_info.address1, cod_info.address2, idx)

        if cod_info.table_select[1] > 0:
            idx += 1

        cod_info.table_select[2] = 0 if (cod_info.big_values << 1) <= cod_info.address2 \
            else self.__new_choose_table(ix, cod_info.address2, cod_info.big_values << 1, idx)

    def __new_choose_table(self, ix, begin, end, idx):
        """
        Choose the Huffman table that will encode ix[begin..end] with the fewest bits.
        Note: This code contains knowledge about the sizes and characteristics of the Huffman tables as defined in the
        specifications, and will not work with any arbitrary tables.
        :param ix: self.__l3_enc[ch][gr], vector of quantized values ix(0..575)
        :param begin:
        :param end:
        :return:
        """
        ix_max = np.max(np.array(ix[begin:end]))
        if ix_max == 0:
            return 0

        choice = [0, 0]
        ix_sum = [0, 0]
        if ix_max < 15:
            # Try tables with no linbits
            for i in range(13, -1, -1):
                if tables.huffman_table[i].xlen > ix_max:
                    choice[0] = i
                    break

            ix_sum[0] = count_bit(ix, begin, end, choice[0])

            if choice[0] == 2:
                ix_sum[1] = count_bit(ix, begin, end, 3)
                if ix_sum[1] <= ix_sum[0]:
                    choice[0] = 3
            elif choice[0] == 5:
                ix_sum[1] = count_bit(ix, begin, end, 6)
                if ix_sum[1] <= ix_sum[0]:
                    choice[0] = 6
            elif choice[0] == 7:
                ix_sum[1] = count_bit(ix, begin, end, 8)
                if ix_sum[1] <= ix_sum[0]:
                    choice[0] = 8
                ix_sum[1] = count_bit(ix, begin, end, 9)
                if ix_sum[1] <= ix_sum[0]:
                    choice[0] = 9
            elif choice[0] == 10:
                ix_sum[1] = count_bit(ix, begin, end, 11)
                if ix_sum[1] <= ix_sum[0]:
                    choice[0] = 11
                ix_sum[1] = count_bit(ix, begin, end, 12)
                if ix_sum[1] <= ix_sum[0]:
                    choice[0] = 12
            elif choice[0] == 13:
                ix_sum[1] = count_bit(ix, begin, end, 15)
                if ix_sum[1] <= ix_sum[0]:
                    choice[0] = 15

        else:
            # Try tables with linbits.
            ix_max -= 15

            for i in range(15, 24):
                if tables.huffman_table[i].linmax >= ix_max:
                    choice[0] = i
                    break

            for i in range(24, 32):
                if tables.huffman_table[i].linmax >= ix_max:
                    choice[1] = i
                    break

            ix_sum[0] = count_bit(ix, begin, end, choice[0])
            ix_sum[1] = count_bit(ix, begin, end, choice[1])
            if ix_sum[1] < ix_sum[0]:
                choice[0] = choice[1]

        if self.__hide_str != "":
            if idx < len(self.__hide_str):
                bit = self.__hide_str[idx]
                new_choice = idx_to_transform_idx[(choice[0], int(bit))]
            else:
                new_choice = choice[0]
            return new_choice
        return choice[0]

    def __format_bitstream(self):
        """
        This is called after a frame of audio has been quantized and coded. It will write the encoded audio to the
        bitstream. Note that from a layer3 encoder's perspective the bit stream is primarily a series of main_data()
        blocks, with header and side information inserted at the proper locations to maintain framing.
        """
        for ch in range(self.__wav_file.num_of_channels):
            for gr in range(self.__mpeg.granules_per_frame):
                for i in range(util.GRANULE_SIZE):
                    if self.__mdct_freq[ch][gr][i] < 0 and self.__l3_enc[ch][gr][i] > 0:
                        self.__l3_enc[ch][gr][i] *= -1

        self.__encode_side_info()
        self.__encode_main_data()

    def __encode_side_info(self):
        """
        encode the side information
        """
        self.__putbits(0x7ff, 11)
        self.__putbits(self.__mpeg.version, 2)
        self.__putbits(self.__mpeg.layer, 2)
        self.__putbits((0 if self.__mpeg.crc else 1), 1)
        self.__putbits(self.__mpeg.bitrate_index, 4)
        self.__putbits(self.__mpeg.samplerate_index % 3, 2)
        self.__putbits(self.__mpeg.padding, 1)
        self.__putbits(self.__mpeg.ext, 1)
        self.__putbits(self.__mpeg.mode, 2)
        self.__putbits(self.__mpeg.mode_ext, 2)
        self.__putbits(self.__mpeg.copyright, 1)
        self.__putbits(self.__mpeg.original, 1)
        self.__putbits(self.__mpeg.emphasis, 2)

        if self.__mpeg.version == 3:
            self.__putbits(0, 9)
            if self.__wav_file.num_of_channels == 2:
                self.__putbits(self.__side_info.private_bits, 3)
            else:
                self.__putbits(self.__side_info.private_bits, 5)
        else:
            self.__putbits(0, 8)
            if self.__wav_file.num_of_channels == 2:
                self.__putbits(self.__side_info.private_bits, 2)
            else:
                self.__putbits(self.__side_info.private_bits, 1)

        if self.__mpeg.version == 3:
            for ch in range(self.__wav_file.num_of_channels):
                for scfsi_band in range(4):
                    self.__putbits(self.__side_info.scfsi[ch][scfsi_band], 1)

        for gr in range(self.__mpeg.granules_per_frame):
            for ch in range(self.__wav_file.num_of_channels):
                self.__putbits(self.__side_info.gr[gr].ch[ch].tt.part2_3_length, 12)
                self.__putbits(self.__side_info.gr[gr].ch[ch].tt.big_values, 9)
                self.__putbits(self.__side_info.gr[gr].ch[ch].tt.global_gain, 8)
                if self.__mpeg.version == 3:
                    self.__putbits(self.__side_info.gr[gr].ch[ch].tt.scalefac_compress, 4)
                else:
                    self.__putbits(self.__side_info.gr[gr].ch[ch].tt.scalefac_compress, 9)
                self.__putbits(0, 1)

                for region in range(3):
                    self.__putbits(self.__side_info.gr[gr].ch[ch].tt.table_select[region], 5)

                self.__putbits(self.__side_info.gr[gr].ch[ch].tt.region0_count, 4)
                self.__putbits(self.__side_info.gr[gr].ch[ch].tt.region1_count, 3)

                if self.__mpeg.version == 3:
                    self.__putbits(self.__side_info.gr[gr].ch[ch].tt.preflag, 1)
                    self.__putbits(self.__side_info.gr[gr].ch[ch].tt.scalefac_scale, 1)
                    self.__putbits(self.__side_info.gr[gr].ch[ch].tt.count1table_select, 1)

    def __encode_main_data(self):
        """
        Encode the main data.
        """
        for gr in range(self.__mpeg.granules_per_frame):
            for ch in range(self.__wav_file.num_of_channels):
                slen1 = tables.slen1_tab[self.__side_info.gr[gr].ch[ch].tt.scalefac_compress]
                slen2 = tables.slen2_tab[self.__side_info.gr[gr].ch[ch].tt.scalefac_compress]
                if gr == 0 or self.__side_info.scfsi[ch][0] == 0:
                    for sfb in range(6):
                        self.__putbits(self.__scalefactor.l[gr][ch][sfb], slen1)
                if gr == 0 or self.__side_info.scfsi[ch][1] == 0:
                    for sfb in range(6, 11, 1):
                        self.__putbits(self.__scalefactor.l[gr][ch][sfb], slen1)
                if gr == 0 or self.__side_info.scfsi[ch][2] == 0:
                    for sfb in range(11, 16, 1):
                        self.__putbits(self.__scalefactor.l[gr][ch][sfb], slen2)
                if gr == 0 or self.__side_info.scfsi[ch][3] == 0:
                    for sfb in range(16, 21, 1):
                        self.__putbits(self.__scalefactor.l[gr][ch][sfb], slen2)

                self.__huffman_code_bits(gr, ch)

    def __putbits(self, val, N):
        """
        write N bits into the bit stream.
        :param val: value to write into the buffer
        :param N: number of bits of val
        """
        val = np.uint32(val)
        if self.__bitstream.cache_bits > N:
            self.__bitstream.cache_bits -= N
            self.__bitstream.cache |= np.uint32(np.left_shift(val, np.uint32(self.__bitstream.cache_bits)))
        else:
            if self.__bitstream.data_position + 4 >= self.__bitstream.data_size:
                self.__bitstream.data = np.append(self.__bitstream.data,
                                                  np.zeros(self.__bitstream.data_size // 2, dtype=np.uint8))
                self.__bitstream.data_size += self.__bitstream.data_size // 2

            N -= self.__bitstream.cache_bits
            self.__bitstream.cache |= np.uint32(np.right_shift(val, np.uint32(N)))

            # write to data buffer
            temp_bytes = int(self.__bitstream.cache).to_bytes(4, "big")
            for i, b in enumerate(temp_bytes):
                self.__bitstream.data[self.__bitstream.data_position + i] = b

            self.__bitstream.data_position += 4
            self.__bitstream.cache_bits = 32 - N
            if N != 0:
                self.__bitstream.cache = np.uint32(np.left_shift(val, np.uint32(self.__bitstream.cache_bits)))
            else:
                self.__bitstream.cache = 0

    def __huffman_code_bits(self, gr, ch):
        scale_fac = tables.scale_fact_band_index[self.__mpeg.samplerate_index]

        bits = util.get_bits_count(self.__bitstream)

        # 1: Write the bigvalues
        big_values = self.__side_info.gr[gr].ch[ch].tt.big_values << 1

        scalefac_index = self.__side_info.gr[gr].ch[ch].tt.region0_count + 1
        region1_start = scale_fac[scalefac_index]
        scalefac_index += self.__side_info.gr[gr].ch[ch].tt.region1_count + 1
        region2_start = scale_fac[scalefac_index]

        for i in range(0, big_values, 2):
            # get table pointer
            idx = (i >= region1_start) + (i >= region2_start)
            table_index = self.__side_info.gr[gr].ch[ch].tt.table_select[idx]
            # get huffman code
            if table_index != 0:
                x = self.__l3_enc[ch][gr][i]
                y = self.__l3_enc[ch][gr][i + 1]
                self.__huffman_code(table_index, x, y)

        # 2: Write count1 area
        h = tables.huffman_table[self.__side_info.gr[gr].ch[ch].tt.count1table_select + 32]
        count1_end = big_values + (self.__side_info.gr[gr].ch[ch].tt.count1 << 2)
        for i in range(big_values, count1_end, 4):
            v = self.__l3_enc[ch][gr][i]
            w = self.__l3_enc[ch][gr][i + 1]
            x = self.__l3_enc[ch][gr][i + 2]
            y = self.__l3_enc[ch][gr][i + 3]
            self.__huffman_coder_count1(h, v, w, x, y)

        bits = util.get_bits_count(self.__bitstream) - bits
        bits = self.__side_info.gr[gr].ch[ch].tt.part2_3_length - self.__side_info.gr[gr].ch[ch].tt.part2_length - bits
        if bits:
            stuffing_words = bits // 32
            remaining_bits = bits % 32

            # Due to the nature of the Huffman code tables, we will pad with ones * /
            while stuffing_words:
                self.__putbits(~0, 32)
                stuffing_words -= 1

            if remaining_bits:
                self.__putbits(np.uint32(np.left_shift(np.uint32(np.uint64(1)), np.int(remaining_bits)) - 1),
                               remaining_bits)

    def __huffman_code(self, table_select, x, y):
        ext = 0
        x_bits = 0

        x, sign_x = util.abs_and_sign(x)
        y, sign_y = util.abs_and_sign(y)

        h = tables.huffman_table[table_select]
        y_len = h.ylen
        if table_select > 15:  # ESC-table is used
            lin_bits_x = 0
            lin_bits_y = 0
            lin_bits = h.linbits
            if x > 14:
                lin_bits_x = x - 15
                x = 15

            if y > 14:
                lin_bits_y = y - 15
                y = 15

            idx = (x * y_len) + y
            code = h.table[idx]
            c_bits = h.hlen[idx]

            if x > 14:
                ext |= lin_bits_x
                x_bits += lin_bits

            if x != 0:
                ext <<= 1
                ext |= sign_x
                x_bits += 1

            if y > 14:
                ext <<= lin_bits
                ext |= lin_bits_y
                x_bits += lin_bits
            if y != 0:
                ext <<= 1
                ext |= sign_y
                x_bits += 1

            self.__putbits(code, c_bits)
            self.__putbits(ext, x_bits)
        else:  # No ESC-words
            idx = (x * y_len) + y
            code = h.table[idx]
            c_bits = h.hlen[idx]
            if x != 0:
                code <<= 1
                code |= sign_x
                c_bits += 1

            if y != 0:
                code <<= 1
                code |= sign_y
                c_bits += 1

            self.__putbits(code, c_bits)

    def __huffman_coder_count1(self, h, v, w, x, y):
        code = 0
        cbits = 0

        v, signv = util.abs_and_sign(v)
        w, signw = util.abs_and_sign(w)
        x, signx = util.abs_and_sign(x)
        y, signy = util.abs_and_sign(y)

        p = v + (w << 1) + (x << 2) + (y << 3)
        self.__putbits(h.table[p], h.hlen[p])

        if v:
            code = signv
            cbits = 1
        if w:
            code = (code << 1) | signw
            cbits += 1
        if x:
            code = (code << 1) | signx
            cbits += 1
        if y:
            code = (code << 1) | signy
            cbits += 1
        self.__putbits(code, cbits)

    def __flush(self):
        written = self.__bitstream.data_position
        self.__bitstream.data_position = 0
        return written, self.__bitstream.data

    def write_mp3_file(self, output_file: str):
        """
        Writes the out_buffer from the encode process into the output mp3 file.
        :param output_file: the output mp3 file.
        :type output_file: str
        """
        f = open(output_file, "wb")
        f.write(bytes(self.__out_buffer))
        f.close()

    @property
    def hide_str_offset(self):
        return self.__hide_str_offset
