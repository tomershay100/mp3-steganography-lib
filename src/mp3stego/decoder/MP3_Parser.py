from scipy.io.wavfile import write
from tqdm import tqdm

from mp3stego.decoder.Frame import *

HEADER_SIZE = 4


class MP3Parser:
    """
    Class for parsing mp3 files into wav file.

    :param file_data: buffer for the file hexadecimal data.
    :type file_data: list
    :param offset: offset for the file to begin after the id3.
    :type offset: int
    :param wav_file_path: the output .wav file path
    :type wav_file_path: str
    """

    def __init__(self, file_data: list, offset: int, wav_file_path: str):
        # Declarations
        self.__curr_frame: Frame = Frame()
        self.__valid: bool = False
        # List of integers that contain the file (without ID3) data
        self.__file_data: list = []
        self.__buffer: list = []
        self.__pcm_data: np.array = np.array([])
        self.__file_length: int = 0
        # self.__file_path = file_path
        self.__wav_file_path: str = wav_file_path

        # cut the id3 from hex_data
        self.__buffer: list = file_data[offset:]

        if self.__buffer[0] == 0xFF and self.__buffer[1] >= 0xE0:
            self.__valid: bool = True
            self.__file_data: list = file_data
            self.__file_length: int = len(file_data)
            self.__offset: int = offset
            self.__init_curr_header()
            self.__curr_frame.set_frame_size()
        else:
            self.__valid: bool = False

        self.output_bits: str = ""

    def __init_curr_header(self):
        if self.__buffer[0] == 0xFF and self.__buffer[1] >= 0xE0:
            self.__curr_frame.init_header_params(self.__buffer)
        else:
            self.__valid = False

    def __init_curr_frame(self):
        self.__curr_frame.init_frame_params(self.__buffer, self.__file_data, self.__offset)

    def parse_file(self) -> int:
        """
        decoding the mp3 file, frame by frame and saves the final pcm data.

        :return: the number of parsed frames
        :rtype: int
        """
        pcm_data = []
        num_of_parsed_frames = 0

        pbar = tqdm(total=self.__file_length + 1 - HEADER_SIZE, desc='decoding')
        while self.__valid and self.__file_length > self.__offset + HEADER_SIZE:
            self.__init_curr_header()
            if self.__valid:
                self.__init_curr_frame()
                # get all bits from the huffman tables
                self.output_bits += util.bit_from_huffman_tables(self.__curr_frame.all_huffman_tables)
                num_of_parsed_frames += 1
                self.__offset += self.__curr_frame.frame_size
                self.__buffer = self.__file_data[self.__offset:]
                # print(f'Parsed: {num_of_parsed_frames}')

            pcm_data.extend(list(self.__curr_frame.pcm.copy()))
            pbar.update(self.__curr_frame.frame_size)

        pbar.close()
        self.__pcm_data = np.array(pcm_data)

        return num_of_parsed_frames

    def write_to_wav(self):
        """
        Convert PCM to WAV (from 32-bit floating-point to 16-bit PCM by mult by 32767)
        """
        write(self.__wav_file_path, self.__curr_frame.sampling_rate, (self.__pcm_data * 32767).astype(np.int16))

    def get_bitrate(self) -> int:
        """
        :return: the bitrate of the mp3 file (and the output wav file)
        :rtype: int
        """
        return self.__curr_frame.get_bitrate()
