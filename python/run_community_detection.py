#!/usr/bin/env python
from __future__ import print_function

import argparse
import sys

from chess_social.pgn_file import PgnFile
from chess_social.graph import ChessGraph
from chess_social.bayes_community_detection import CommunityDetector

def main(data_file_name, iterations, output_dir, min_elo, p_in, p_out, burnin):

    with PgnFile(data_file_name) as pgnfile:
        graph = ChessGraph(pgnfile, min_elo=min_elo)

    detector = CommunityDetector(p_in=p_in, p_out=p_out)

    labels = detector.run(graph, iterations=iterations)

    assert len(labels) == iterations + 1

    chosen_index, communities = CommunityDetector.estimate_partitions(labels, burnin=burnin)
    graph.communities = communities
    graph.render_community_graph(show_single_nodes=False)

    return 0


if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument('filename', metavar='datafile')
    cmdline_parser.add_argument('--iterations', action='store', dest='iterations',
                                type=int, default=100)
    cmdline_parser.add_argument('--output_dir', action='store', default='.')
    cmdline_parser.add_argument('--burnin', action='store', type=int, default=0)
    cmdline_parser.add_argument('--min_elo', action='store', type=int, default=2500)
    cmdline_parser.add_argument('--p_in', action='store', type=float, default=0.8)
    cmdline_parser.add_argument('--p_out', action='store', type=float, default=0.2)

    parsed_args = cmdline_parser.parse_args()

    if parsed_args.p_in >= 1.0 or parsed_args.p_out >= 1.0:
        print('Invalid edge probabilities: {0}, {1}'.format(parsed_args.p_in,
                                                            parsed_args.p_out))
        sys.exit(1)

    if parsed_args.min_elo < 1000:
        print('Invalid minimum ELO rating: {0}'.format(parsed_args.min_elo))
        sys.exit(1)

    sys.exit(main(parsed_args.filename,
                  parsed_args.iterations,
                  parsed_args.output_dir,
                  parsed_args.min_elo,
                  parsed_args.p_in,
                  parsed_args.p_out,
                  parsed_args.burnin))

