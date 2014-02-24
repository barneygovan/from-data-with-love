
from __future__ import print_function

import unittest
import mock
from mock import MagicMock

from chess_social.pgn_file import PgnFile

TEST_PGN_DATA = '''[Event "76th Tata Steel Masters"]
[Site "Wijk aan Zee NED"]
[Date "2014.01.11"]
[Round "1.1"]
[White "Dominguez Perez,L"]
[Black "Giri,A"]
[Result "1/2-1/2"]
[WhiteTitle "GM"]
[BlackTitle "GM"]
[WhiteElo "2754"]
[BlackElo "2734"]
[ECO "C67"]
[Opening "Ruy Lopez"]
[Variation "Berlin defence, open variation"]
[WhiteFideId "3503240"]
[BlackFideId "24116068"]
[EventDate "2014.01.11"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6 4. O-O Nxe4 5. d4 Nd6 6. Bxc6 dxc6 7. dxe5 Nf5 8.
Qxd8+ Kxd8 9. Rd1+ Ke8 10. Bf4 Ne7 11. h3 Ng6 12. Bh2 Be7 13. Nc3 Bd7 14. Nd4
Nf8 15. Rd2 h5 16. Rad1 Rd8 17. a3 h4 18. Nce2 c5 19. e6 Nxe6 20. Nxe6 Bxe6 21.
Bxc7 Rxd2 22. Rxd2 f6 23. Nf4 Kf7 24. Nxe6 Kxe6 25. Kf1 Rc8 26. Ba5 b6 27. Bc3
b5 28. Re2+ Kf7 29. Rd2 Ke6 30. Re2+ Kf7 31. Rd2 1/2-1/2

[Event "76th Tata Steel Masters"]
[Site "Wijk aan Zee NED"]
[Date "2014.01.11"]
[Round "1.2"]
[White "Van Wely,L"]
[Black "Karjakin,Sergey"]
[Result "0-1"]
[WhiteTitle "GM"]
[BlackTitle "GM"]
[WhiteElo "2672"]
[BlackElo "2759"]
[ECO "E06"]
[Opening "Catalan"]
[Variation "closed, 5.Nf3"]
[WhiteFideId "1000268"]
[BlackFideId "14109603"]
[EventDate "2014.01.11"]

1. d4 Nf6 2. c4 e6 3. g3 d5 4. Bg2 Be7 5. Nf3 O-O 6. O-O dxc4 7. Qc2 a6 8. a4
Bd7 9. Qxc4 Bc6 10. Bf4 a5 11. Nc3 Na6 12. Rae1 Bd5 13. Nxd5 exd5 14. Qb5 Qc8
15. Qb3 Nb4 16. Nd2 c6 17. e4 dxe4 18. Nxe4 Nxe4 19. Rxe4 Qd7 20. Bd2 Nd5 21.
Re2 Bf6 22. Bxd5 Qxd5 23. Qxd5 cxd5 24. Rc1 Bxd4 25. Bf4 Rfd8 26. Rc7 b6 27.
Ree7 Bxb2 28. Rxf7 Rac8 29. Rb7 Rc4 30. Rfc7 Bd4 31. Be3 Rxc7 32. Rxc7 Bxe3 33.
fxe3 d4 34. exd4 Rxd4 35. Rb7 Rb4 36. Kg2 h5 37. Kf3 Kh7 38. h4 Kg6 39. Rd7 Kf6
40. Rd6+ Kf5 41. Rd5+ Ke6 42. Rg5 Kf7 43. Rxh5 Rxa4 44. Rh8 Rb4 45. Ra8 g5 46.
hxg5 Kg6 47. Rg8+ Kh5 48. g4+ Kh4 49. Rg6 a4 50. Rg7 b5 51. Rg8 a3 52. g6 Kg5
53. g7 Rb3+ 54. Ke4 Kg6 55. Ra8 Kxg7 56. Kf5 Rf3+ 57. Ke4 Rc3 58. Kf5 Rc5+ 59.
Kf4 Rc4+ 60. Kf5 Ra4 0-1

[Event "76th Tata Steel Masters"]
[Site "Wijk aan Zee NED"]
[Date "2014.01.11"]
[Round "1.3"]
[White "Harikrishna,P"]
[Black "Aronian,L"]
[Result "1/2-1/2"]
[WhiteTitle "GM"]
[BlackTitle "GM"]
[WhiteElo "2706"]
[BlackElo "2812"]
[ECO "C54"]
[Opening "Giuoco Piano"]
[WhiteFideId "5007003"]
[BlackFideId "13300474"]
[EventDate "2014.01.11"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Bd2 Nxe4 8.
Bxb4 Nxb4 9. Bxf7+ Kxf7 10. Qb3+ d5 11. Ne5+ Ke6 12. Qxb4 Qf8 13. Qxf8 Rxf8 14.
Nc3 c6 15. f3 Nd6 16. Kf2 Nf5 17. Rhe1 Kd6 18. Rad1 g6 19. b4 Bd7 20. Na4 b6 21.
Rd2 Rae8 22. Nc3 Re7 23. Rc1 Be8 24. Ne2 Ng7 25. Re1 Ne6 26. Nc1 Rf4 27. Ne2 Rf5
28. Nc3 Rf4 29. Ne2 Rf5 30. Nc3 Rf4 1/2-1/2

[Event "76th Tata Steel Masters"]
[Site "Wijk aan Zee NED"]
[Date "2014.01.11"]
[Round "1.4"]
[White "Caruana,F"]
[Black "Gelfand,B"]
[Result "1-0"]
[WhiteTitle "GM"]
[BlackTitle "GM"]
[WhiteElo "2782"]
[BlackElo "2777"]
[ECO "B90"]
[Opening "Sicilian"]
[Variation "Najdorf"]
[WhiteFideId "2020009"]
[BlackFideId "2805677"]
[EventDate "2014.01.11"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. f3 e5 7. Nb3 Be6 8. Be3
h5 9. Nd5 Bxd5 10. exd5 Nbd7 11. Qd2 g6 12. Be2 Bg7 13. O-O O-O 14. Rac1 b6 15.
h3 Re8 16. g4 hxg4 17. hxg4 Nh7 18. g5 f5 19. gxf6 Bxf6 20. Rf2 Bg5 21. Rg2
Bxe3+ 22. Qxe3 Ndf8 23. Bd3 Ra7 24. Rf1 Rf7 25. Qh6 Kh8 26. Nd2 Rf4 27. Rg4 b5
28. Ne4 Nd7 29. Rxg6 Rg8 30. Ng5 1-0
'''

class PgnFileTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @mock.patch('__builtin__.open')
    def test_parse_game(self, open_mock):

        mock_file = MagicMock(spec=file)

        results = []
        open_mock.return_value = mock_file
        mock_file.__iter__.return_value = iter(TEST_PGN_DATA.split('\n'))
        with PgnFile('blah') as pgnfile:
            for game in pgnfile:
                results.append(game)

        expected_game_1 = {'black': "Giri,A",
                           'black_id': '24116068',
                           'black_elo': '2734',
                           'black_title': 'GM',
                           'white': "Dominguez Perez,L",
                           'white_id': '3503240',
                           'white_elo': '2754',
                           'white_title': 'GM',}

        expected_game_2 = {'black': "Karjakin,Sergey",
                           'black_id': '14109603',
                           'black_elo': '2759',
                           'black_title': 'GM',
                           'white': "Van Wely,L",
                           'white_id': '1000268',
                           'white_elo': '2672',
                           'white_title': 'GM',}

        expected_game_3 = {'black': "Aronian,L",
                           'black_id': '13300474',
                           'black_elo': '2812',
                           'black_title': 'GM',
                           'white': "Harikrishna,P",
                           'white_id': '5007003',
                           'white_elo': '2706',
                           'white_title': 'GM',}

        expected_game_4 = {'black': "Gelfand,B",
                           'black_id': '2805677',
                           'black_elo': '2777',
                           'black_title': 'GM',
                           'white': "Caruana,F",
                           'white_id': '2020009',
                           'white_elo': '2782',
                           'white_title': 'GM',}

        expected_results = [expected_game_1, expected_game_2, expected_game_3, expected_game_4]

        self.assertEqual(expected_results, results)

