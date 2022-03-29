## This code is a modified version of https://github.com/constantintcacenco/xer-to-csv-converter.
import os
import pandas as pd

class XerToCsvConverter:
    def __init__(self, lang="en"):
        self.tables = []
        self.lang = lang.lower()
        if self.lang == "ko":
            self.encoding = "cp949"
        else:
            self.encoding = "utf8"

    def readXer(self, filePath):
        with open(filePath, encoding=self.encoding, errors="ignore") as f:
            content = f.read()
            tables  = content.split("%T")
            self.tables = tables[1:]

    ## auxiliary function (no call from user needed)
    def checkOutputDir(self, outputLocation):
        if not os.path.exists(outputLocation):
            os.makedirs(outputLocation)

    ## auxiliary function (no call from user needed)
    def checkMissingValues(self, columns, rowsList):
        for row in rowsList:
            if len(columns) > len(row):
                row[len(columns):len(row)] = [None]*(len(columns) - len(row))
        return rowsList

    def convert2csv(self, outputPath):
        for table in self.tables:
            tableName = table.split()[0]
            fields     = table.split(r"%F")[1].split("\n")[0].split()
            
            rows = table.split("%R")[1:]
            rowsList = [r.strip().split("\t") for r in rows]
            checkedRowsList = self.checkMissingValues(fields, rowsList)

            df = pd.DataFrame(checkedRowsList, columns=fields, index=None)
            self.checkOutputDir(outputPath)
            csvFilePath = os.path.join(outputPath, tableName + ".csv")
            if self.lang == "ko":
                df.to_csv(csvFilePath, encoding="utf-8-sig")
            else:
                df.to_csv(csvFilePath, encoding=self.encoding)

pathXER   = r"./inputs/file.xer" # target XER file directory
pathCSVs  = r"./outputs/" # target folder directory to extract several CSV files.

lang = "ko" # select language between "en" and "ko".

converter = XerToCsvConverter(lang=lang) # default lang is "en".
converter.readXer(pathXER)

outputSubDir = os.path.join(pathCSVs, pathXER.split(".xer")[0].split("/")[-1])

if not os.path.exists(outputSubDir):
    os.makedirs(outputSubDir)

converter.convert2csv(outputSubDir)