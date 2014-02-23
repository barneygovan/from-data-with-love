from __future__ import print_function

import math
import operator
from collections import defaultdict

import numpy as np
import networkx as nx
import pylab

class GraphError(Exception):
    pass


class ChessPlayer(object):

    def __init__(self, fide_id, name, title):
        self.__fide_id = fide_id
        self.__name = name
        self.__elo = None
        self.__title = title
        self.__community = None

    @property
    def elo(self):
        return self.__elo

    @elo.setter
    def elo(self, elo):
        self.__elo = elo

    @property
    def name(self):
        return self.__name

    @property
    def fide_id(self):
        return self.__fide_id

    @property
    def title(self):
        return self.__title

    @property
    def community(self):
        return self.__community

    @community.setter
    def community(self, community):
        self.__community = community

    def __hash__(self):
        return hash(self.fide_id)

    def __eq__(self, other):
        return self.fide_id == other.fide_id

    def __unicode__(self):
        return '{0}:{1}:{2}:{3}'.format(self.fide_id,
                                        self.name,
                                        self.elo,
                                        self.community)

    def __str__(self):
        return unicode(self).encode('utf-8')


class ChessGame(object):

    def __init__(self, player_one, player_two):
        self.__player_one = min(player_one, player_two)
        self.__player_two = max(player_one, player_two)

    @property
    def player_one(self):
        return self.__player_one

    @property
    def player_two(self):
        return self.__player_two

    def __hash__(self):
        return hash((self.player_one, self.player_two))

    def __eq__(self, other):
        return (self.player_one, self.player_two) == (other.player_one, other.player_two)

    def __unicode__(self):
        return '{0} vs. {1}'.format(self.player_one, self.player_two)

    def __str__(self):
        return unicode(self).encode('utf-8')


class ChessGraph(object):

    def __init__(self, pgnfile, min_elo=0):
        nodes = defaultdict(list)
        self.__edges = defaultdict(int)
        self.__communities = None
        self.__sorted_players = None

        for game in pgnfile:
            if game:
                black_elo = int(game['black_elo'])
                white_elo = int(game['white_elo'])
                if black_elo < min_elo or white_elo < min_elo:
                    continue
                black_player = ChessPlayer(game['black_id'], game['black'], game['black_title'])
                nodes[black_player].append(black_elo)
                white_player = ChessPlayer(game['white_id'], game['white'], game['white_title'])
                nodes[white_player].append(white_elo)

                chess_game = ChessGame(game['black_id'], game['white_id'])
                self.__edges[chess_game] += 1

        self.__players = {}
        for player, elo in nodes.iteritems():
            player.elo = np.mean(elo)
            self.__players[player.fide_id] = player

        self.__min_elo = min_elo

        print('Loaded', self.number_of_nodes, 'players')
        print('Loaded', self.number_of_edges, 'games')

    @property
    def number_of_edges(self):
        return len(self.__edges)

    @property
    def number_of_nodes(self):
        return len(self.__players)

    @property
    def communities(self):
        return self.__communities

    @communities.setter
    def communities(self, communities):
        '''Initialize player community assignments'''
        if len(communities) != len(self.__players):
            raise GraphError()
        for i, player_id in enumerate(sorted(self.__players.iterkeys())):
            self.__players[player_id].community = communities[i]
        self.__communities = communities


    @property
    def number_of_communities(self):
        if self.__communities is None:
            return 0
        return len(np.unique(self.__communities))


    @property
    def adjacency_matrix(self):
        return self.__edges

    @property
    def nodes(self):
        '''Returns a tuple of players ordered by FIDE id'''
        if not self.__sorted_players:
            _, self.__sorted_players = zip(*sorted(self.__players.iteritems(),
                                                 key=operator.itemgetter(0)))
        return self.__sorted_players

    def get_node(self, player_id):
        return self.__players.get(player_id, None)

    def render_graph(self, max_edges=None, min_games=1):
        added_nodes = set()
        graph = nx.Graph()
        for i, (edge, num) in enumerate(self.adjacency_matrix.iteritems()):
            if max_edges and i > max_edges:
                break
            if num < min_games:
                continue
            graph.add_edge(edge.player_one, edge.player_two)
            added_nodes.add(edge.player_one)
            added_nodes.add(edge.player_two)

        for node in self.nodes:
            if node.fide_id in added_nodes:
                graph.node[node.fide_id]['elo_rank'] = int(math.floor(node.elo/100) * 100)

        min_val = self.__min_elo
        max_val = 2900
        elo_levels = range(min_val, max_val, 100)
        color_levels = np.linspace(1, 0, num=len(elo_levels), endpoint=True)
        color_value_map = {elo: color for (elo, color) in zip(elo_levels, color_levels)}
        color_values = [color_value_map.get(graph.node[n]['elo_rank'], 0.0) for n in graph.nodes()]

        nx.draw_graphviz(graph, cmap=pylab.get_cmap('jet'), node_color=color_values)

    def render_community_graph(self, show_single_nodes=True):
        added_nodes = set()
        graph = nx.Graph()
        for edge, _ in self.adjacency_matrix.iteritems():
            player_one = self.__players[edge.player_one]
            player_two = self.__players[edge.player_two]
            added = False
            if show_single_nodes:
                graph.add_node(edge.player_one)
                graph.add_node(edge.player_two)
                added = True
            if player_one.community == player_two.community:
                graph.add_edge(edge.player_one, edge.player_two)
                added = True
            if added:
                added_nodes.add(edge.player_one)
                added_nodes.add(edge.player_two)

        for node in self.nodes:
            if node.fide_id in added_nodes:
                graph.node[node.fide_id]['elo_rank'] = math.floor(node.elo/100) * 100

        color_value_map = {
            2500.0: 1.0,
            2600.0: 0.6,
            2700.0: 0.3,
            2800.0: 0.0
        }
        color_values = [color_value_map.get(graph.node[n]['elo_rank'], 0.0) for n in graph.nodes()]

        nx.draw_graphviz(graph, cmap=pylab.get_cmap('jet'), node_color=color_values,
                         node_size=150)


