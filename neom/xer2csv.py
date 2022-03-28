## This code is a modified version of https://github.com/constantintcacenco/xer-to-csv-converter.
try:
    import xer2csv
    from xer2csv import XerToCsvConverter
except:
    from pip._internal import main as pip
    pip(['install', '--user', 'xer2csv'])
    import xer2csv
    from xer2csv import XerToCsvConverter

import sys, os

converter   = XerToCsvConverter()
xerFilePath = sys.argv[1]
csvFilePath = sys.argv[2]

converter.read_xer(xerFilePath)
converter.convert_to_csv(csvFilePath)