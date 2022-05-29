import os
import sys
import time

from mp3stego.encoder.MP3_Encoder import MP3Encoder
from mp3stego.encoder.WAV_Reader import WavReader


class Encoder:
    """
    Class for wrapping the MP3Encoder class, for creating mp3 file from wav file.

    :param file_path: the wav file path.
    :type file_path: str
    :param output_file_path: the mp3 output file path.
    :type output_file_path: str
    :param bitrate: the bitrate of the input wav file
    :type bitrate: int
    :param hide_str: if is not empty, hides the string inside the output mp3 file.
    :type hide_str: str
    """

    def __init__(self, file_path, output_file_path, bitrate: int = 320, hide_str: str = ''):
        self.__file_path = file_path
        self.__output_file_path = output_file_path

        if not os.path.exists(self.__file_path):
            sys.exit('File not found.')

        self.__wav_file = WavReader(self.__file_path, bitrate)
        self.__hide_str = hide_str
        self.__encoder = MP3Encoder(self.__wav_file, hide_str=hide_str)

    def encode(self, quiet: bool = True):
        """
        Encoding the input wav file into mp3 file. can also hide string in the file.

        :param quiet: if False, print some information about the decoding process
        :type quiet: bool

        """
        if not quiet:
            self.__encoder.print_info()
        self.__encoder.encode()

        self.__encoder.write_mp3_file(self.__output_file_path)
        if not quiet:
            if self.__encoder.hide_str_offset < len(self.__hide_str) - 1:
                print("File too short for this message length, your message has been trimmed.")
            print(f"MP3 file created on {self.__output_file_path}")


if __name__ == "__main__":
    start = time.time()

    if len(sys.argv) > 2:
        sys.exit('Unexpected number of arguments.')
    if len(sys.argv) < 2:
        sys.exit('No directory specified.')
    file_path = sys.argv[1]

    e = Encoder(file_path, file_path)
    e.encode()

    print(f'Execution time: {int(time.time() - start)} seconds')
