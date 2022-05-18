import argparse
import os
import sys
import time

from mp3stego.decoder.main import decode, delete_wav_file
from mp3stego.encoder.main import encode


class Stego:
    def __init__(self, input_file_path, output_file_path='.'):
        self.__input_file_path = input_file_path

        if not os.path.exists(self.__input_file_path):
            sys.exit('Input file not found.')

        self.__output_file_path = output_file_path
        if self.__output_file_path == '.':
            self.__output_file_path = self.__input_file_path[:-4] + "_out" + self.__input_file_path[-4:]

        if self.__input_file_path[-4:] != '.mp3' or self.__output_file_path[-4:] != '.mp3':
            sys.exit("Input and output files must be mp3 files.")

        self.__stego = False

    @property
    def stego(self):
        return self.__stego

    @stego.setter
    def stego(self, stego: bool):
        self.__stego = stego

    def run(self, data: str = "", quiet: bool = True):
        start_time = time.time()
        if not quiet:
            print("Start parsing the file")
        bitrate, wav_file_path = decode(self.__input_file_path)
        parse_time = time.time()
        if not quiet:
            print(f"File decoding time took {int(parse_time - start_time)} seconds")
        encode(self.__input_file_path[:-3] + 'wav', self.__output_file_path, bitrate, quiet)
        # TODO create child class from encode, and if data is not empty, use the new class with steganography option
        end_time = time.time()
        if not quiet:
            print(f"File encoding time took {int(end_time - parse_time)} seconds")
        delete_wav_file(wav_file_path)


if __name__ == "__main__":
    # get arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-infile", dest="mp3_input_file", default=".", type=str, help="the mp3 input file path")
    arg_parser.add_argument("-outfile", dest="mp3_output_file", default=".", type=str, help="the mp3 out file path")
    args = arg_parser.parse_args()

    input_file = args.mp3_input_file
    output_file = args.mp3_output_file

    stego = Stego(os.path.abspath(input_file), os.path.abspath(output_file) if output_file != '.' else output_file)
    stego.run()
