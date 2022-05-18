import argparse
import os
import sys
import time

from mp3stego.decoder.main import decode, delete_wav_file
from mp3stego.encoder.main import encode


def main(infile: str, outfile: str):
    start_time = time.time()
    print("Start parsing the file")
    bitrate, wav_file_path = decode(infile)
    parse_time = time.time()
    print(f"File decoding time took {int(parse_time - start_time)} seconds")
    encode(infile[:-3] + 'wav', outfile, bitrate)
    end_time = time.time()
    print(f"File encoding time took {int(end_time - parse_time)} seconds")
    delete_wav_file(wav_file_path)


if __name__ == "__main__":
    # get arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-infile", dest="mp3_input_file", default=".", type=str, help="the mp3 input file path")
    arg_parser.add_argument("-outfile", dest="mp3_output_file", default=".", type=str, help="the mp3 out file path")
    args = arg_parser.parse_args()

    input_file = args.mp3_input_file
    if input_file == '.':
        arg_parser.print_help()
        sys.exit(0)
    output_file = args.mp3_output_file if args.mp3_output_file != '.' else (input_file[:-4] + "_out" + input_file[-4:])
    if input_file[-4:] != '.mp3' or output_file[-4:] != '.mp3':
        sys.exit("Input and output files must be mp3 files.")
    if not os.path.exists(input_file):
        sys.exit('Input file not found.')

    main(os.path.abspath(input_file), os.path.abspath(output_file))
