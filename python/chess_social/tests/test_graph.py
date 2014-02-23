from __future__ import print_function

import unittest
from mock import MagicMock

import numpy as np

from chess_social.graph import ChessGame, ChessPlayer, ChessGraph, GraphError

class ChessGameTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_equality(self):
        game_one = ChessGame(1234, 1235)
        game_two = ChessGame(1234, 1235)

        self.assertEqual(game_one, game_two)

        game_three = ChessGame(1235, 1234)

        self.assertEqual(game_one, game_three)

        game_four = ChessGame(1234, 1236)

        self.assertNotEqual(game_one, game_four)

class ChessPlayerTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_equality(self):
        player_one = ChessPlayer(1234, 'test_name_one', 'GM')
        player_two = ChessPlayer(1234, 'test_name_one', 'IM')

        self.assertEqual(player_one, player_two)

        player_three = ChessPlayer(1234, 'test_name_two', 'NM')

        self.assertEqual(player_one, player_three)

        player_four = ChessPlayer(1235, 'test_name_one', 'GM')

        self.assertNotEqual(player_one, player_four)


class ChessGraphTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_initialize_communities(self):
        pgn_file = []

        game_1 = {'black_id': '1',
                  'black': 'player_one',
                  'black_elo': '101',
                  'black_title': 'GM',
                  'white_id': '2',
                  'white': 'player_two',
                  'white_elo': '202',
                  'white_title': 'GM'}
        game_2 = {'black_id': '3',
                  'black': 'player_three',
                  'black_elo': '303',
                  'black_title': 'GM',
                  'white_id': '4',
                  'white': 'player_four',
                  'white_elo': '404',
                  'white_title': 'GM'}
        game_3 = {'black_id': '1',
                  'black': 'player_one',
                  'black_elo': '101',
                  'black_title': 'GM',
                  'white_id': '4',
                  'white': 'player_four',
                  'white_elo': '404',
                  'white_title': 'GM'}
        pgn_file.extend([game_1, game_2, game_3])

        graph = ChessGraph(pgn_file)
        self.assertEqual(len(pgn_file), graph.number_of_edges)
        self.assertEqual(4, graph.number_of_nodes)

        self.assertEqual(0, graph.number_of_communities)

        community_labels = [1, 2, 2, 3]
        expected_number_of_communities = len(np.unique(community_labels))

        graph.communities = community_labels
        self.assertEqual(expected_number_of_communities, graph.number_of_communities)
        self.assertEqual(community_labels, graph.communities)

        bad_community_labels = [1, 2, 2, 3, 4]
        with self.assertRaises(GraphError) as _:
            graph.communities = bad_community_labels

        bad_community_labels = [1, 2, 3]
        with self.assertRaises(GraphError) as _:
            graph.communities = bad_community_labels


