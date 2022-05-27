import os
import sys
import time

from mp3stego.decoder.ID3_Parser import ID3
from mp3stego.decoder.MP3_Parser import MP3Parser


class Decoder:
    """
    Class for wrapping the MP3Parser class, for creating wav file from mp3 file.

    :param file_path: the mp3 file path.
    :type file_path: str
    :param output_file_path: the wav output file path.
    :type output_file_path: str
    """

    def __init__(self, file_path, output_file_path):
        self.__file_path = file_path
        self.__output_file_path = output_file_path

    def parse_metadata(self, id3_parser: ID3):
        with open('METADATA.txt', 'w') as metadata:
            metadata.write(f'METADATA FOR FILE: {self.__file_path}\n')
            metadata.write('################################\n\n\n')
            metadata.write(f'ID3 Version: {id3_parser.version}\n')
            if len(id3_parser.id3_flags) > 0:
                metadata.write('ID3 Flags:\n')
                for flag in id3_parser.id3_flags:
                    metadata.write(f'- {flag}\n')
                metadata.write('\n')

            metadata.write('\nID3 Frames:\n')
            for i, frame in enumerate(id3_parser.id3_frames):
                metadata.write(f'Frame number: {i}\n')
                metadata.write(f'Frame ID: {frame.id}\n')
                metadata.write(f'Content: {frame.content}\n')
                if len(frame.frame_flags) > 0:
                    metadata.write('Frame Flags:\n')
                    for flag in frame.frame_flags:
                        metadata.write(f'- {flag}\n')
                metadata.write('\n')

    def decode(self, quiet=True, reveal=False, txt_file_path=""):
        if not os.path.exists(self.__file_path):
            sys.exit('File not found.')

        with open(self.__file_path, 'rb') as f:
            hex_data = [c for c in f.read()]

        id3_decoder = ID3(hex_data)
        if id3_decoder.is_valid:  # add quiet
            self.parse_metadata(id3_decoder)
            offset = id3_decoder.offset

        else:
            offset = 0

        parser = MP3Parser(hex_data, offset, self.__output_file_path)
        start = time.time()
        num_of_parsed_frames = parser.parse_file()
        parsing_time = time.time() - start
        if not quiet:
            print('\nParsed', num_of_parsed_frames, 'frames in', parsing_time, 'seconds.')

        parser.write_to_wav()
        if not quiet:
            print(f"Wav file created on {self.__output_file_path}")

        if reveal:
            if txt_file_path[-4:] != '.txt':
                sys.exit("txt_file_path must be txt file.")

            output_str = ''.join(chr(int(''.join(x), 2)) for x in zip(*[iter(parser.output_bits)] * 8))
            message_len_str = ''
            for idx, ch in enumerate(output_str):
                if ch == '#':
                    break
                message_len_str += ch
            try:
                message_len = int(message_len_str)
            except:
                message_len = 0
                message_len_str = ""

            if (len(message_len_str) + 1 + message_len) > len(output_str):
                output_str = output_str[len(message_len_str) + 1:]
            else:
                output_str = output_str[len(message_len_str) + 1: len(message_len_str) + 1 + message_len]
            f = open(txt_file_path, 'wb')
            f.write(bytes(output_str, 'utf-8'))
            f.close()

        return parser.get_bitrate() // 1000

    def delete_wav_file(self):
        if os.path.exists(self.__output_file_path):
            os.remove(self.__output_file_path)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Unexpected number of arguments.")
        exit(-1)
    if len(sys.argv) < 2:
        print("No directory specified.")
        exit(-1)
    file_path = sys.argv[1]

    d = Decoder(file_path, file_path[:-3] + 'wav')
    d.decode()
