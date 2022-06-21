import unittest

from tests import Steganography


class SteganographyTestCase(unittest.TestCase):
    def setUp(self):
        self.stego = Steganography(quiet=True)
        self.input_path = "tests/test.mp3"
        self.output_mp3_path = "tests/out.mp3"
        self.output_wav_path = "tests/out.wav"
        self.reveal_txt_path = "tests/reveal.txt"
        self.cleared_mp3_path = "tests/cleared.mp3"

    def test_decoder_encoder(self):
        """
        Test the encoding and decoding process.
        """
        bitrate = self.stego.decode_mp3_to_wav(self.input_path, self.output_wav_path)
        self.assertEqual(bitrate, 320)
        self.stego.encode_wav_to_mp3(self.output_wav_path, self.output_mp3_path, bitrate)

    def test_hiding(self):
        """
        Test the hiding when the message is short enough.
        """
        too_long = self.stego.hide_message(self.input_path, self.output_mp3_path, message='ddd')
        self.assertEqual(too_long, False)

    def test_too_long_hiding(self):
        """
        Test the hiding when the message is too long.
        """
        too_long = self.stego.hide_message(self.input_path, self.output_mp3_path, message='ddd' * 100)
        self.assertEqual(too_long, True)

    def test_reveal_hiding(self):
        """
        Test the revealing message from mp3 file.
        """
        self.stego.hide_message(self.input_path, self.output_mp3_path, message='ddd')
        self.stego.reveal_massage(self.output_mp3_path, self.reveal_txt_path)

        with open(self.reveal_txt_path) as f:
            message = f.read()

        self.assertEqual(message, 'ddd')

    def test_reveal_cleared(self):
        """
        Test the revealing message from cleared mp3 file.
        """
        self.stego.hide_message(self.input_path, self.output_mp3_path, message='ddd')
        self.stego.clear_file(self.output_mp3_path, self.cleared_mp3_path)
        self.stego.reveal_massage(self.cleared_mp3_path, self.reveal_txt_path)

        with open(self.reveal_txt_path) as f:
            message = f.read()

        self.assertEqual(message, '')


if __name__ == '__main__':
    unittest.main()
