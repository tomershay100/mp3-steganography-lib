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

    def __init__(self, file_path: str, output_file_path: str):
        self.__file_path: str = file_path
        self.__output_file_path: str = output_file_path

        if not os.path.exists(self.__file_path):
            sys.exit(f'File {self.__file_path} not found.')

        with open(self.__file_path, 'rb') as f:
            self.__hex_data: list = [c for c in f.read()]

        self.__id3_decoder: ID3 = ID3(self.__hex_data)
        if self.__id3_decoder.is_valid:
            offset = self.__id3_decoder.offset
        else:
            offset = 0

        self.__parser: MP3Parser = MP3Parser(self.__hex_data, offset, self.__output_file_path)

    def __parse_metadata(self, id3_parser: ID3):
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

    def decode(self, quiet: bool = True, reveal: bool = False, txt_file_path: str = "") -> int:
        """
        Decoding the input mp3 file into wav file. Parse also the metadata.

        :param quiet: if False, print some information about the decoding process
        :type quiet: bool
        :param reveal: if True, reveals the hidden string in the mp3 file.
        :type reveal: bool
        :param txt_file_path: if reveal is True, saves the string into this txt file path.
        :type txt_file_path: str

        :return: the bitrate of the mp3 file (also the bitrate of the output wav file)
        :rtype: int
        """
        if not quiet and self.__id3_decoder.is_valid:
            self.__parse_metadata(self.__id3_decoder)

        start = time.time()
        num_of_parsed_frames = self.__parser.parse_file()
        parsing_time = time.time() - start
        if not quiet:
            print('\nParsed', num_of_parsed_frames, 'frames in', parsing_time, 'seconds.')

        self.__parser.write_to_wav()
        if not quiet:
            print(f"Wav file created on {self.__output_file_path}")

        if reveal:
            if txt_file_path[-4:] != '.txt':
                sys.exit("txt_file_path must be txt file.")

            output_str = ''.join(chr(int(''.join(x), 2)) for x in zip(*[iter(self.__parser.output_bits)] * 8))
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

        return self.__parser.get_bitrate() // 1000

    def delete_wav_file(self):
        """
        Deletes the output wav file
        """
        if os.path.exists(self.__output_file_path):
            os.remove(self.__output_file_path)
