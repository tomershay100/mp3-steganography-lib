import os
import sys
import time

from mp3stego.encoder.MP3_Encoder import MP3Encoder
from mp3stego.encoder.WAV_Reader import WavReader


def encode(file_path, output_file_path, bitrate=320, quiet=True):
    if not os.path.exists(file_path):
        sys.exit('File not found.')

    wav_file = WavReader(file_path, bitrate)
    encoder = MP3Encoder(wav_file)
    if not quiet:
        encoder.print_info()
    encoder.encode()

    encoder.write_mp3_file(output_file_path)


if __name__ == "__main__":
    start = time.time()

    if len(sys.argv) > 2:
        sys.exit('Unexpected number of arguments.')
    if len(sys.argv) < 2:
        sys.exit('No directory specified.')
    file_path = sys.argv[1]

    encode(file_path, file_path)

    print(f'Execution time: {int(time.time() - start)} seconds')
