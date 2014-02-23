'''
Provides code for handling large pgn (portable game notation) files.
'''
from __future__ import print_function

import re


class PgnFile(object):
    '''
    PgnFile represents a large pgn file made up of many games.
    It enables you to parse the file game by game without
    having to deal with the lines in the file.
    The class is a context manager and an iterator.
    This means you can do the following:

    import chess_social.pgn_file as pgn

    with pgn.PgnFile('my_games.pgn') as pgnfile:
        for game in pgnfile:
            print(game['white'], 'vs.', game['black'])
    '''

    EVENT_PATTERN = re.compile(r'^\[Event .*')
    BLACK_PATTERN = re.compile(r'^\[Black "([^"]+)"\]$')
    WHITE_PATTERN = re.compile(r'^\[White "([^"]+)"\]$')
    BLACK_ID_PATTERN = re.compile(r'^\[BlackFideId "([^"]+)"\]$')
    WHITE_ID_PATTERN = re.compile(r'^\[WhiteFideId "([^"]+)"\]$')
    BLACK_ELO_PATTERN = re.compile(r'^\[BlackElo "([^"]+)"\]$')
    WHITE_ELO_PATTERN = re.compile(r'^\[WhiteElo "([^"]+)"\]$')
    BLACK_TITLE_PATTERN = re.compile(r'^\[BlackTitle "([^"]+)"\]$')
    WHITE_TITLE_PATTERN = re.compile(r'^\[WhiteTitle "([^"]+)"\]$')

    REQUIRED_KEYS = ('black', 'white', 'black_id', 'white_id',
                     'black_elo', 'white_elo')

    def __init__(self, filename):
        self.__filename = filename
        self.__pgn_file = None

        # with open(self.__filename) as pgn_file:
        #     pgn_data = pgn_file.read()
        # self._pgn_lines = pgn_data.split('\n')
        # del pgn_data

    def __enter__(self):
        self.__pgn_file = open(self.__filename)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__pgn_file.close()
        if exc_type is not None:
            return False
        return True


    def __iter__(self):
        return self

    @staticmethod
    def _match_me(line, pattern, game, field_name):
        if game.has_key(field_name):
            return False
        line_match = pattern.match(line)
        if line_match:
            game[field_name] = line_match.group(1)
            return True
        return False

    @staticmethod
    def _verify_game(game, game_text):
        for field in PgnFile.REQUIRED_KEYS:
            if not game.has_key(field):
                return None
        if game['black_id'] == game['white_id']:
            print('ERROR: Duplicate player ids:')
            print('\n'.join(game_text))
            return None
        if not game.has_key('black_title'):
            game['black_title'] = 'None'
        if not game.has_key('white_title'):
            game['white_title'] = 'None'
        return game

    def next(self):
        '''
        Implementation of iterator
        '''
        in_game = False
        game = {}
        empty_line_count = 0

        game_text = []

        for line in self.__pgn_file:
            line = line.strip()
            if not line:
                if in_game:
                    empty_line_count += 1
                if empty_line_count >= 2:
                    break
                continue
            game_text.append(line)
            line_match = PgnFile.EVENT_PATTERN.match(line)
            if line_match:
                in_game = True
                continue
            if PgnFile._match_me(line, PgnFile.BLACK_PATTERN, game, 'black'):
                continue
            if PgnFile._match_me(line, PgnFile.WHITE_PATTERN, game, 'white'):
                continue
            if PgnFile._match_me(line, PgnFile.BLACK_ID_PATTERN, game, 'black_id'):
                continue
            if PgnFile._match_me(line, PgnFile.WHITE_ID_PATTERN, game, 'white_id'):
                continue
            if PgnFile._match_me(line, PgnFile.BLACK_ELO_PATTERN, game, 'black_elo'):
                continue
            if PgnFile._match_me(line, PgnFile.WHITE_ELO_PATTERN, game, 'white_elo'):
                continue
            if PgnFile._match_me(line, PgnFile.BLACK_TITLE_PATTERN, game, 'black_title'):
                continue
            if PgnFile._match_me(line, PgnFile.WHITE_TITLE_PATTERN, game, 'white_title'):
                continue
        if not game:
            raise StopIteration

        game = self._verify_game(game, game_text)

        return game

