import os
import sys

import bitarray

from mp3stego import Decoder
from mp3stego import Encoder


def str_to_binary_str(string: str) -> str:
    """
    Convert string into string of bits.

    :param string: the string
    :type string: str

    :return: the string of bits.
    :rtype: str
    """
    binary_str = bitarray.bitarray()
    binary_str.frombytes(string.encode('utf-8'))
    binary_str = [str(ch) for ch in binary_str]
    binary_str = "".join(binary_str)
    return binary_str


class Steganography:
    """
    This class is the Façade of the module. It allows the users a simple way to encode and decode mp3 and wav files,
    hide messages in mp3 files, reveal hidden string and clear mp3 files from any string.

    :param quiet: if False, prints information about the processes and the files.
    :type quiet: bool
    """

    def __init__(self, quiet: bool = True):
        self.quiet = quiet
        self.__encoder: Encoder
        self.__decoder: Decoder
        self.__bitrate: int

    def __encode(self, wav_file_path: str, output_file_path: str, bitrate: int = 320, quiet: bool = True,
                 hide: bool = False, massage: str = "") -> bool:
        binary_str = ""
        if hide:
            massage = str(len(massage)) + "#" + massage
            binary_str = str_to_binary_str(massage)

        self.__encoder = Encoder(wav_file_path, output_file_path, bitrate=bitrate, hide_str=binary_str)
        return self.__encoder.encode(quiet=quiet)

    def __decode(self, input_file_path: str, wav_file_path: str, quiet: bool = True, reveal: bool = False,
                 txt_file_path: str = ""):
        self.__decoder = Decoder(input_file_path, wav_file_path)
        self.__bitrate = self.__decoder.decode(quiet, reveal=reveal, txt_file_path=txt_file_path)

    def __delete_wav_file(self, quiet: bool = True):
        self.__decoder.delete_wav_file()
        if not quiet:
            print("Wav file has been deleted.")

    @staticmethod
    def __file_existence(file: str):
        if not os.path.exists(file):
            sys.exit(f'File {file} not found.')

    def __check_for_decoder(self, input_file_path: str, wav_file_path: str = ""):
        self.__file_existence(input_file_path)
        if wav_file_path == '':
            wav_file_path = input_file_path[:-4] + ".wav"
        if input_file_path[-4:] != '.mp3' or wav_file_path[-4:] != '.wav':
            sys.exit("input_file_path must be mp3 file, wav_file_path must be wav file.")
        return wav_file_path

    def __check_for_encoder(self, wav_file_path: str, output_file_path: str):
        self.__file_existence(wav_file_path)
        if output_file_path[-4:] != '.mp3' or wav_file_path[-4:] != '.wav':
            sys.exit("wav_file_path must be wav file, output_file_path must be mp3 file.")

    def encode_wav_to_mp3(self, wav_file_path: str, output_file_path: str, bitrate: int = 320):
        """
        Allow encoding wav file into mp3 file.

        :param wav_file_path: the wav file path.
        :type wav_file_path: str
        :param output_file_path: the output mp3 file desired path.
        :type output_file_path: str
        :param bitrate: the bitrate of the wav file.
        :type bitrate: int
        """
        if not self.quiet:
            print(f"\n##################\nStart Encoding {wav_file_path} to  {output_file_path}.")
        self.__check_for_encoder(wav_file_path, output_file_path)
        self.__encode(wav_file_path, output_file_path, hide=False, bitrate=bitrate, quiet=self.quiet)
        if not self.quiet:
            print(f"\nFinished Encoding.\n##################")

    def decode_mp3_to_wav(self, input_file_path: str, wav_file_path: str = "") -> int:
        """
        Allow decoding mp3 file into wav file.

        :param input_file_path: the input mp3 file path.
        :type input_file_path: str
        :param wav_file_path: the output wav file desired path.
        :type wav_file_path: str

        :return: the bitrate of the mp3 (and wav) file.
        :rtype: int
        """
        if not self.quiet:
            print(f"\n##################\nStart Decoding {input_file_path} to  {wav_file_path}.")
        wav_file_path = self.__check_for_decoder(input_file_path, wav_file_path)
        self.__decode(input_file_path, wav_file_path, reveal=False, quiet=self.quiet)
        if not self.quiet:
            print(f"\nFinished Decoding.\n##################")
        return self.__bitrate

    def reveal_massage(self, input_file_path: str, txt_file_path: str):
        """
        Allow revealing string from mp3 file. The function writes the string into txt file.

        :param input_file_path: the input mp3 file path.
        :type input_file_path: str
        :param txt_file_path: the output txt file desired path.
        :type txt_file_path: str
        """
        if not self.quiet:
            print(f"\n##################\nStart Revealing hidden message in {input_file_path} to  {txt_file_path}.")
        wav_file_path = self.__check_for_decoder(input_file_path, "")
        if txt_file_path[-4:] != '.txt':
            sys.exit("txt_file_path must be txt file.")
        self.__decode(input_file_path, wav_file_path, reveal=True, quiet=self.quiet, txt_file_path=txt_file_path)
        self.__delete_wav_file(quiet=self.quiet)
        if not self.quiet:
            print(f"\nFinished Revealing.\n##################")

    def hide_message(self, input_file_path: str, output_file_path: str, message: str) -> bool:
        """
        Allow hiding string in mp3 file. The function creates the new mp3 file with the string hidden in it.

        :param input_file_path: the input mp3 file path.
        :type input_file_path: str
        :param output_file_path: the output mp3 desired path.
        :type output_file_path: str
        :param message: the message to hide in the mp3 file.
        :type message: str

        :return True if the message is too long to be inserted into the mp3 file.
        :rtype bool
        """
        if not self.quiet:
            print(f"\n##################\nStart Hiding {message} in {output_file_path}.")
        wav_file_path = self.__check_for_decoder(input_file_path, "")
        self.__decode(input_file_path, wav_file_path, reveal=False, quiet=self.quiet)

        self.__check_for_encoder(wav_file_path, output_file_path)
        too_long = self.__encode(wav_file_path, output_file_path, hide=True, bitrate=self.__bitrate, quiet=self.quiet,
                                 massage=message)
        self.__delete_wav_file(quiet=self.quiet)
        if not self.quiet:
            print(f"\nFinished Hiding.\n##################")
        return too_long

    def clear_file(self, input_file_path: str, output_file_path: str):
        """
        Allow clearing mp3 file from hidden string in it. The function creates new mp3 file without the hidden string.

        :param input_file_path: the input mp3 file path.
        :type input_file_path: str
        :param output_file_path: the output mp3 desired path.
        :type output_file_path: str
        """
        if not self.quiet:
            print(f"\n##################\nStart Cleaning {input_file_path} into {output_file_path}.")
        wav_file_path = self.__check_for_decoder(input_file_path, "")
        self.__decode(input_file_path, wav_file_path, reveal=False, quiet=self.quiet)

        self.__check_for_encoder(wav_file_path, output_file_path)
        self.__encode(wav_file_path, output_file_path, hide=False, bitrate=self.__bitrate, quiet=self.quiet)
        self.__delete_wav_file(quiet=self.quiet)
        if not self.quiet:
            print(f"\nFinished Cleaning.\n##################")
