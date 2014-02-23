from __future__ import print_function

import unittest

from chess_social.bayes_community_detection import *

class CommunityDetectorTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_create_community_detector(self):
        detector = CommunityDetector(a_in=5.0,
                                     b_in=6.0,
                                     a_out=7.0,
                                     b_out=8.0,
                                     gamma_a=9.0,
                                     gamma_b=10.0)

        self.assertIsNotNone(detector)
        self.assertEquals(str(detector), '[INIT: 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]')
