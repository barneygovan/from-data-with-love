#!/usr/bin/env python

import os
import urllib
import zipfile

from bs4 import BeautifulSoup


def main():
    twic = urllib.urlopen('http://www.theweekinchess.com/twic')
    twic_html = twic.read()

    soup = BeautifulSoup(twic_html)

    pgn_urls = []
    for table in soup.find_all('table'):
        if 'results-table' in table['class']:
            for a in table.find_all('a'):
                if a.string == 'PGN':
                    pgn_urls.append(a.get('href'))

    zpgn_files = []
    for url in pgn_urls:
        zpgn_filename = url.split('/')[-1]
        zpgn_files.append(zpgn_filename)
        urllib.urlretrieve(url, zpgn_filename)

    pgn_data = []
    for zpgn_file in zpgn_files:
        zfile = zipfile.ZipFile(zpgn_file)
        zfile.extractall()
        pgn_filename = zpgn_file.replace('g.zip', '.pgn')
        with open(pgn_filename) as pgn_file:
            pgn = pgn_file.read()
            pgn_data.append(pgn)

    pgn_data = '\n\n'.join(pgn_data)

    with open('twic_chess_data.pgn', 'w+') as pgn_outfile:
        pgn_outfile.write(pgn_data)

    for zpgn_file in zpgn_files:
        os.remove(zpgn_file)
        pgn_filename = zpgn_file.replace('g.zip', '.pgn')
        os.remove(pgn_filename)

if __name__ == '__main__':
    main()
