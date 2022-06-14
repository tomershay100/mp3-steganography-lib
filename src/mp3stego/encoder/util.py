import numpy as np
from numba import njit

MAX_CHANNELS = 2
MAX_GRANULES = 2
GRANULE_SIZE = 576

SB_LIMIT = 32
HAN_SIZE = 512  # for loop unrolling, require that HAN_SIZE%8==0
PI = 3.14159265358979
PI36 = 0.087266462599717
PI64 = 0.049087385212
LN2 = 0.69314718

BUFFER_SIZE = 4096

SFB_LMAX = 22

en_tot_krit = 10
en_dif_krit = 100
en_scfsi_band_krit = 10
xm_scfsi_band_krit = 10

BIT_RATES = [
    # MPEG version:
    # 2.5, reserved, II, I
    (-1, -1, -1, -1),
    (8, -1, 8, 32),
    (16, -1, 16, 40),
    (24, -1, 24, 48),
    (32, -1, 32, 56),
    (40, -1, 40, 64),
    (48, -1, 48, 80),
    (56, -1, 56, 96),
    (64, -1, 64, 112),
    (-1, -1, 80, 128),
    (-1, -1, 96, 160),
    (-1, -1, 112, 192),
    (-1, -1, 128, 224),
    (-1, -1, 144, 256),
    (-1, -1, 160, 320),
    (-1, -1, -1, -1)
]
SAMPLE_RATES = [
    44100, 48000, 32000,  # MPEG - I
    22050, 24000, 16000,  # MPEG - II
    11025, 12000, 8000  # MPEG - 2.5
]

MODES = {
    "STEREO": 0,
    "JOINT STEREO": 1,
    "DUAL CHANNEL": 2,
    "MONO": 3
}
MPEG_VERSIONS = {
    "MPEG_I": 3,
    "MPEG_II": 2,
    "MPEG_25": 0
}

GRANULES_PER_FRAME = [
    1,  # MPEG 2.5
    -1,  # Reserved
    1,  # MPEG II
    2  # MPEG I
]
GRANULES_SIZE = 576

scale_fact_band_index = [  # MPEG-I
    #  Table B.8.b: 44.1 kHz
    [0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 52, 62, 74, 90, 110, 134, 162, 196, 238, 288, 342, 418, 576],
    #  Table B.8.c: 48 kHz
    [0, 4, 8, 12, 16, 20, 24, 30, 36, 42, 50, 60, 72, 88, 106, 128, 156, 190, 230, 276, 330, 384, 576],
    #  Table B.8.a: 32 kHz
    [0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 54, 66, 82, 102, 126, 156, 194, 240, 296, 364, 448, 550, 576],
    #  MPEG-II
    #  Table B.2.b: 22.05 kHz
    [0, 6, 12, 18, 24, 30, 36, 44, 54, 66, 80, 96, 116, 140, 168, 200, 238, 284, 336, 396, 464, 522, 576],
    #  Table B.2.c: 24 kHz
    [0, 6, 12, 18, 24, 30, 36, 44, 54, 66, 80, 96, 114, 136, 162, 194, 232, 278, 330, 394, 464, 540, 576],
    #  Table B.2.a: 16 kHz
    [0, 6, 12, 18, 24, 30, 36, 44, 45, 66, 80, 96, 116, 140, 168, 200, 238, 248, 336, 396, 464, 522, 576],

    #  MPEG-2.5
    #  11.025 kHz
    [0, 6, 12, 18, 24, 30, 36, 44, 54, 66, 80, 96, 116, 140, 168, 200, 238, 284, 336, 396, 464, 522, 576],
    #  12 kHz
    [0, 6, 12, 18, 24, 30, 36, 44, 54, 66, 80, 96, 116, 140, 168, 200, 238, 284, 336, 396, 464, 522, 576],
    #  MPEG-2.5 8 kHz
    [0, 12, 24, 36, 48, 60, 72, 88, 108, 132, 160, 192, 232, 280, 336, 400, 476, 566, 568, 570, 572, 574, 576]]


def find_bitrate_index(bitrate, mpeg_version):
    for i in range(16):
        if bitrate == BIT_RATES[i][mpeg_version]:
            return i

    return -1


def find_samplerate_index(samplerate):
    for i in range(9):
        if samplerate == SAMPLE_RATES[i]:
            return i

    return -1


def find_mpeg_version(samplerate_index):
    # Pick mpeg version according to samplerate index.
    if samplerate_index < 3:
        # First 3 samplerates are for MPEG-I
        return MPEG_VERSIONS["MPEG_I"]
    elif samplerate_index < 6:
        # Then it's MPEG-II
        return MPEG_VERSIONS["MPEG_II"]
    else:
        # Finally, MPEG-2.5
        return MPEG_VERSIONS["MPEG_25"]


@njit(fastmath=True)
def mulsr(a, b):
    a = np.int64(a)
    b = np.int64(b)
    return np.int32((np.right_shift(((a * b) + np.int64(1073741824)), 31)))


@njit(fastmath=True)
def mulr(a, b):
    a = np.int64(a)
    b = np.int64(b)
    return np.int32((np.right_shift((a * b) + np.int64(2147483648), 32)))


@njit(fastmath=True)
def mul(a, b):
    a = np.int64(a)
    b = np.int64(b)
    tmp = np.right_shift((a * b), 32)
    return np.int32(tmp)


@njit(fastmath=True)
def cmuls(are, aim, bre, bim):
    are = np.int64(are)
    aim = np.int64(aim)
    bre = np.int64(bre)
    bim = np.int64(bim)

    tre = np.int32(np.right_shift(are * bre - aim * bim, 31))
    dim = np.int32(np.right_shift(are * bim + aim * bre, 31))
    dre = tre
    return dre, dim


@njit(fastmath=True)
def labs(a):
    return np.abs(np.long(a))


def get_bits_count(bitstream):
    return bitstream.data_position * 8 + 32 - bitstream.cache_bits


@njit(fastmath=True)
def abs_and_sign(x):
    if x > 0:
        return x, 0
    x *= -1
    return x, 1
