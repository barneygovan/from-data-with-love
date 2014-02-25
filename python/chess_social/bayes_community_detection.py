
from __future__ import division, print_function

import math
import operator
import datetime

import numpy as np
import numpy.random as npr

from collections import defaultdict

from chess_social.graph import ChessGame

# CONSTANTS
A_IN = 'a_in'
B_IN = 'b_in'
A_OUT = 'a_out'
B_OUT = 'b_out'
GAMMA_A = 'gamma_a'
GAMMA_B = 'gamma_b'

P_IN = 'p_in'
P_OUT = 'p_out'
ALPHA = 'alpha'

__all__ = ['CommunityDetector',]


class CommunityDetector(object):

    def __init__(self, **kw_args):
        self.__a_in = kw_args.get(A_IN, 2.0)
        self.__b_in = kw_args.get(B_IN, 1.0)
        self.__a_out = kw_args.get(A_OUT, 1.0)
        self.__b_out = kw_args.get(B_OUT, 2.0)
        self.__gamma_a = kw_args.get(GAMMA_A, 1.0)
        self.__gamma_b = kw_args.get(GAMMA_B, 1.0)

        self.__p_in_0 = kw_args.get(P_IN, 0.8)
        self.__p_out_0 = kw_args.get(P_OUT, 0.2)
        self.__alpha_0 = kw_args.get(ALPHA, 10.0)

    @staticmethod
    def __edge_count(graph):
        community_count = defaultdict(int)
        seen_players = set()
        edges_in = 0
        edges_out = 0
        node_pairs_in = 0
        node_pairs_out = 0
        for game in graph.adjacency_matrix.keys():
            player_one = graph.get_node(game.player_one)
            player_two = graph.get_node(game.player_two)
            if player_one.community == player_two.community:
                edges_in += 1
            else:
                edges_out += 1
            if player_one not in seen_players:
                community_count[player_one.community] += 1
                seen_players.add(player_one)
            if player_two not in seen_players:
                community_count[player_two.community] += 1
                seen_players.add(player_two)
        total_possible_pairs_of_nodes = graph.number_of_nodes * (graph.number_of_nodes - 1) * 0.5
        total_possible_pairs_inside = 0
        for count in community_count.itervalues():
            total_possible_pairs_inside += count * (count -1) * 0.5
        node_pairs_in = total_possible_pairs_inside - edges_in
        node_pairs_out = total_possible_pairs_of_nodes - total_possible_pairs_inside - edges_out

        return community_count, edges_in, node_pairs_in, edges_out, node_pairs_out

    def __update_p(self, i, edges_in, node_pairs_in, edges_out, node_pairs_out, p_in, p_out):
        p_in_tmp = npr.beta(edges_in + self.__a_in, node_pairs_in + self.__b_in)
        if p_in_tmp > p_out[i-1]:
            p_in[i] = p_in_tmp
        else:
            p_in[i] = p_in[i-1]
        #update p_out, given p_in, pi, and alpha, with constraint that p_out < p_in
        p_out_tmp = npr.beta(edges_out + self.__a_out, node_pairs_out + self.__b_out)
        if p_out_tmp < p_in[i]:
            p_out[i] = p_out_tmp
        else:
            p_out[i] = p_out[i-1]

    @staticmethod
    def __update_labels_for_node_i(labels, graph, i, community_count, p_in, p_out):
        player_communities = labels[i-1].copy()
        for j, player in enumerate(graph.nodes):
            c_count = community_count.copy()
            table_size = c_count[player.community]
            if table_size <= 1:
                del c_count[player.community]
            else:
                c_count[player.community] = table_size - 1
            sorted_labels = sorted(c_count.iterkeys())
            _, probabilities = np.array(zip(*sorted(c_count.iteritems(),
                                                    key=operator.itemgetter(0))),
                                        dtype=np.float64)
            for k, label in enumerate(sorted_labels):
                for node in graph.nodes:
                    if node == player:
                        continue
                    p_ij = p_out[i]
                    if label == node.community:
                        p_ij = p_in[i]
                    game = ChessGame(player.fide_id, node.fide_id)
                    if graph.adjacency_matrix.get(game, 0):
                        probabilities[k] *= p_ij
                    else:
                        probabilities[k] *= (1 - p_ij)
            # sample new label
            the_sum = np.sum(probabilities)
            rnd_unif = npr.uniform()
            cumulative = np.cumsum(probabilities/the_sum)
            # print(cumulative)
            sample_index = np.where(cumulative >= rnd_unif)[0][0]
            new_label = sorted_labels[sample_index]
            player_communities[j] = new_label

        #append new labels to our collection of labels from previous iterations
        labels = np.vstack([labels, player_communities])
        #update graph with current label set
        graph.communities = player_communities

        return labels

    def __calculate_alpha(self, graph, alpha_prev):
        '''Picks alpha from a mixture of 2 gamma distributions'''
        num_communities = graph.number_of_communities
        num_players = graph.number_of_nodes
        beta_z = npr.beta(alpha_prev + 1, num_players)
        #generate a uniform random number to pick which gamma from the mixture
        rnd_unif = npr.uniform()
        inv_mixture_scale = self.__gamma_b - math.log(beta_z)
        mixture_scale = 1.0 / inv_mixture_scale
        if rnd_unif / (1 - rnd_unif) <  ((self.__gamma_a + num_communities)/
                                          (num_players * inv_mixture_scale)):
            return npr.gamma(self.__gamma_a + num_communities, mixture_scale)
        return npr.gamma(self.__gamma_a + num_communities - 1, mixture_scale)

    def run(self, graph, start_labels=None, iterations=100):
        #1. initialize labels in graph
        if not start_labels:
            #start every node in its own community
            start_labels = range(graph.number_of_nodes)
            print('Initializing labels with {0} different labels'.format(graph.number_of_nodes))
        graph.communities = start_labels

        p_in = np.zeros(iterations + 1)
        p_out = np.zeros(iterations + 1)
        p_in[0] = self.__p_in_0
        p_out[0] = self.__p_out_0

        alpha = np.zeros(iterations + 1)
        alpha[0] = self.__alpha_0

        labels = np.array(start_labels)
        labels = labels.reshape((1, len(start_labels)))

        for i in xrange(1, iterations+1):
            (community_count, edges_in, node_pairs_in,
                edges_out, node_pairs_out) = CommunityDetector.__edge_count(graph)

            print('{0}  Number of Communities: {1}; Number of Edges In: {2}; '
                'Number of Edges Out: {3}'.format(
                    datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'),
                    len(community_count), edges_in, edges_out))

            #first update p_in, given p_out, pi, and alpha, with constraint that p_in > p_out
            self.__update_p(i, edges_in, node_pairs_in, edges_out, node_pairs_out, p_in, p_out)

            #update all the labels on each node
            labels = CommunityDetector.__update_labels_for_node_i(labels,
                                                                  graph,
                                                                  i,
                                                                  community_count,
                                                                  p_in,
                                                                  p_out)

            #update alpha
            alpha[i] = self.__calculate_alpha(graph, alpha[i-1])

        return labels

    @staticmethod
    def estimate_partitions(labels, burnin=0):
        #calculate empirical probabilities that c_i == c_j
        emp_prob = {}
        iterations, nodes = labels.shape
        iterations = iterations - burnin
        for i in range(nodes):
            for j in range(i+1, nodes):
                emp_prob[(i, j)] = np.sum(labels[burnin:, i] == labels[burnin:, j]) / iterations

        posterior_risk = np.zeros(iterations)
        for k, iteration in enumerate(labels[burnin:]):
            for i in range(len(iteration)):
                for j in range(i + 1, len(iteration)):
                    if iteration[i] == iteration[j]:
                        posterior_risk[k] += emp_prob[(i, j)] - 0.5

        index = np.where(posterior_risk == np.amax(posterior_risk))[0][0] + burnin
        return index, labels[index]


    def __unicode__(self):
        return r'[INIT: {0}, {1}, {2}, {3}, {4}, {5}]'.format(self.__a_in,
                                                              self.__b_in,
                                                              self.__a_out,
                                                              self.__b_out,
                                                              self.__gamma_a,
                                                              self.__gamma_b)

    def __str__(self):
        return unicode(self).encode('utf-8')
