import os
import sys

import bitarray

from mp3stego import Decoder
from mp3stego import Encoder


# TODO add more information prints when quiet is off
# TODO add more comments
# TODO change function and variables names
class Steganography:
    def __init__(self, quiet=True):
        self.quiet = quiet

    @staticmethod
    def __encode(wav_file_path, output_file_path, bitrate=320, quiet=True, hide=False, massage=""):
        e = Encoder(wav_file_path, output_file_path)
        binary_str = ""
        if hide:
            massage = str(len(massage)) + "#" + massage
            binary_str = bitarray.bitarray()
            binary_str.frombytes(massage.encode('utf-8'))
            binary_str = [str(ch) for ch in binary_str]
            binary_str = "".join(binary_str)

        e.encode(bitrate, quiet, hide_str=binary_str)

    @staticmethod
    def __decode(input_file_path, wav_file_path, quiet=True, reveal=False, txt_file_path=""):
        d = Decoder(input_file_path, wav_file_path)
        bitrate = d.decode(quiet, reveal=reveal, txt_file_path=txt_file_path)
        return bitrate, d

    @staticmethod
    def __delete_wav_file(d: Decoder, quiet=True):
        d.delete_wav_file()
        if not quiet:
            print("Wav file has been deleted.")

    @staticmethod
    def __file_existence(file):
        if not os.path.exists(file):
            sys.exit('File not found.')

    def __check_for_decoder(self, input_file_path, wav_file_path=""):
        self.__file_existence(input_file_path)
        if wav_file_path == '':
            wav_file_path = input_file_path[:-4] + ".wav"
        if input_file_path[-4:] != '.mp3' or wav_file_path[-4:] != '.wav':
            sys.exit("input_file_path must be mp3 file, wav_file_path must be wav file.")
        return wav_file_path

    def __check_for_encoder(self, wav_file_path, output_file_path):
        self.__file_existence(wav_file_path)
        if output_file_path[-4:] != '.mp3' or wav_file_path[-4:] != '.wav':
            sys.exit("wav_file_path must be wav file, output_file_path must be mp3 file.")

    def encode_wav_to_mp3(self, wav_file_path: str, output_file_path: str, bitrate: int = 320):
        if not self.quiet:
            print(f"\n##################\nStart Encoding {wav_file_path} to  {output_file_path}.")
        self.__check_for_encoder(wav_file_path, output_file_path)
        self.__encode(wav_file_path, output_file_path, hide=False, bitrate=bitrate, quiet=self.quiet)
        if not self.quiet:
            print(f"\nFinished Encoding.\n##################")

    def decode_mp3_to_wav(self, input_file_path: str, wav_file_path: str = ""):
        if not self.quiet:
            print(f"\n##################\nStart Decoding {input_file_path} to  {wav_file_path}.")
        wav_file_path = self.__check_for_decoder(input_file_path, wav_file_path)
        bitrate, _ = self.__decode(input_file_path, wav_file_path, reveal=False, quiet=self.quiet)
        if not self.quiet:
            print(f"\nFinished Decoding.\n##################")
        return bitrate

    def reveal_massage(self, input_file_path: str, txt_file_path: str):
        if not self.quiet:
            print(f"\n##################\nStart Revealing hidden message in {input_file_path} to  {txt_file_path}.")
        wav_file_path = self.__check_for_decoder(input_file_path, "")
        if txt_file_path[-4:] != '.txt':
            sys.exit("txt_file_path must be txt file.")
        _, d = self.__decode(input_file_path, wav_file_path, reveal=True, quiet=self.quiet, txt_file_path=txt_file_path)
        self.__delete_wav_file(d)
        if not self.quiet:
            print(f"\nFinished Revealing.\n##################")

    def hide_message(self, input_file_path: str, output_file_path: str, message: str, wav_file_path: str = "",
                     delete_wav: bool = True):
        if not self.quiet:
            print(f"\n##################\nStart Hiding {message} in {output_file_path}.")
        wav_file_path = self.__check_for_decoder(input_file_path, wav_file_path)
        bitrate, d = self.__decode(input_file_path, wav_file_path, reveal=False, quiet=self.quiet)

        self.__check_for_encoder(wav_file_path, output_file_path)
        self.__encode(wav_file_path, output_file_path, hide=True, bitrate=bitrate, quiet=self.quiet, massage=message)
        if delete_wav:
            self.__delete_wav_file(d)
        if not self.quiet:
            print(f"\nFinished Hiding.\n##################")

    def clear_file(self, input_file_path: str, output_file_path: str):
        if not self.quiet:
            print(f"\n##################\nStart Cleaning {input_file_path} into {output_file_path}.")
        wav_file_path = self.__check_for_decoder(input_file_path, "")
        bitrate, d = self.__decode(input_file_path, wav_file_path, reveal=False, quiet=self.quiet)

        self.__check_for_encoder(wav_file_path, output_file_path)
        self.__encode(wav_file_path, output_file_path, hide=False, bitrate=bitrate, quiet=self.quiet)
        self.__delete_wav_file(d)
        if not self.quiet:
            print(f"\nFinished Cleaning.\n##################")
