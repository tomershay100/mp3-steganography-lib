import unittest

from tests import Steganography


class SteganographyTestCase(unittest.TestCase):
    def setUp(self):
        self.stego = Steganography(quiet=True)

    def test_test(self):
        with open('readme.txt', 'w') as f:
            f.write('Create a new text file!')
    # def test_decoder_encoder(self):
    #     """
    #     Test the encoding and decoding process.
    #     """
    #     input_path = os.path.abspath("test.mp3")
    #     output_wav_path = os.getcwd() + "/out.wav"
    #     output_mp3_path = os.getcwd() + "/out.mp3"
    #     bitrate = self.stego.decode_mp3_to_wav(input_path, output_wav_path)
    #     self.assertEqual(bitrate, 320)
    #     self.stego.encode_wav_to_mp3(output_wav_path, output_mp3_path, bitrate)
    #
    # def test_hiding(self):
    #     """
    #     Test the hiding when the message is short enough.
    #     """
    #     input_path = os.path.abspath("test.mp3")
    #     output_mp3_path = os.getcwd() + "/out1.mp3"
    #     too_long = self.stego.hide_message(input_path, output_mp3_path, message='ddd')
    #     self.assertEqual(too_long, False)
    #
    # def test_too_long_hiding(self):
    #     """
    #     Test the hiding when the message is too long.
    #     """
    #     input_path = os.path.abspath("test.mp3")
    #     output_mp3_path = os.getcwd() + "/out2.mp3"
    #     too_long = self.stego.hide_message(input_path, output_mp3_path, message='ddd' * 100)
    #     self.assertEqual(too_long, True)
    #
    # def test_reveal_hiding(self):
    #     """
    #     Test the revealing message from mp3 file.
    #     """
    #     input_path = os.path.abspath("test.mp3")
    #     output_mp3_path = os.getcwd() + "/out3.mp3"
    #     reveal_txt_path = os.getcwd() + "/reveal.txt"
    #     self.stego.hide_message(input_path, output_mp3_path, message='ddd')
    #     self.stego.reveal_massage(output_mp3_path, reveal_txt_path)
    #
    #     with open(reveal_txt_path) as f:
    #         message = f.read()
    #
    #     self.assertEqual(message, 'ddd')
    #
    # def test_reveal_cleared(self):
    #     """
    #     Test the revealing message from cleared mp3 file.
    #     """
    #     input_path = os.path.abspath("test.mp3")
    #     output_mp3_path = os.getcwd() + "/out4.mp3"
    #     reveal_txt_path = os.getcwd() + "/reveal-cleared.txt"
    #     cleared_mp3_path = os.getcwd() + "/cleared.mp3"
    #     self.stego.hide_message(input_path, output_mp3_path, message='ddd')
    #     self.stego.clear_file(output_mp3_path, cleared_mp3_path)
    #     self.stego.reveal_massage(cleared_mp3_path, reveal_txt_path)
    #
    #     with open(reveal_txt_path) as f:
    #         message = f.read()
    #
    #     self.assertEqual(message, '')


if __name__ == '__main__':
    unittest.main()
