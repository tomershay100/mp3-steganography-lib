import os
import unittest

from tests import Steganography


class SteganographyTestCase(unittest.TestCase):
    def setUp(self):
        self.stego = Steganography(quiet=True)

    def test_decoder_encoder(self):
        """
        Test the encoding and decoding process.
        """
        input_path = os.path.abspath("test.mp3")
        output_wav_path = os.getcwd() + "/out.wav"
        output_mp3_path = os.getcwd() + "/out.mp3"
        bitrate = self.stego.decode_mp3_to_wav(input_path, output_wav_path)
        self.assertEqual(bitrate, 320)
        self.stego.encode_wav_to_mp3(output_wav_path, output_mp3_path, bitrate)

    def test_hiding(self):
        """
        Test the hiding when the message is short enough.
        """
        too_long = self.stego.hide_message('test.mp3', 'out.mp3', message='ddd')
        self.assertEqual(too_long, False)

    def test_too_long_hiding(self):
        """
        Test the hiding when the message is too long.
        """
        too_long = self.stego.hide_message('test.mp3', 'out1.mp3', message='ddd' * 100)
        self.assertEqual(too_long, True)

    def test_reveal_hiding(self):
        """
        Test the revealing message from mp3 file.
        """
        self.stego.hide_message('test.mp3', 'out.mp3', message='ddd')
        self.stego.reveal_massage('out.mp3', "reveal.txt")

        with open("reveal.txt") as f:
            message = f.read()

        self.assertEqual(message, 'ddd')

    def test_reveal_cleared(self):
        """
        Test the revealing message from cleared mp3 file.
        """
        self.stego.hide_message('test.mp3', 'out.mp3', message='ddd')
        self.stego.clear_file('out.mp3', "cleared.mp3")
        self.stego.reveal_massage('cleared.mp3', "reveal-cleared.txt")

        with open("reveal-cleared.txt") as f:
            message = f.read()

        self.assertEqual(message, '')


if __name__ == '__main__':
    unittest.main()
