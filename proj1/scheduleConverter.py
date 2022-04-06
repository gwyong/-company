import os, glob
import pandas as pd

class Converter:
    def __init__(self, lang="en"):
        self.tables   = []
        self.lang     = lang.lower()
        if self.lang == "ko":
            self.encoding = "cp949"
        else:
            self.encoding = "utf8"
    
    def MPPtoCSV(self, inputFolderDIR, outputFolderDIR):
        import jpype, mpxj
        import datetime, json
        
        inputPath = os.path.join(inputFolderDIR, "*.mpp")
        filesMPP  = glob.glob(inputPath)
        
        jpype.startJVM()
        from net.sf.mpxj import ProjectFile
        from net.sf.mpxj.reader import UniversalProjectReader
        from net.sf.mpxj.writer import ProjectWriter, ProjectWriterUtility
        from net.sf.mpxj.json import JsonWriter
        
        csvFilePaths = []
        for file in filesMPP:
            fileJSON = file.split(".mpp")[0]+".json"
            projectFile = UniversalProjectReader().read(file)
            writer = JsonWriter()
            writer.write(projectFile, fileJSON)

            with open(fileJSON, 'r') as f:
                jsonData = json.load(f)
                
            for key in jsonData.keys():
                df = pd.DataFrame(jsonData[key])
                csvFilePath = file.split(".mpp")[0]+key+".csv"
                csvFilePaths.append(csvFilePath)
                df.to_csv(csvFilePath, encoding="utf-8-sig")
            
            os.remove(fileJSON)
        jpype.shutdownJVM()
        
        print("Reading MPP files is finished.")
        
        inputPath = os.path.join(inputFolderDIR, "*tasks.csv")
        taskFiles = glob.glob(inputPath)
        
        targetCols = ["wbs", "unique_id", "name",
                      "start", "finish", "early_start", "early_finish", "late_start", "late_finish",
                      "percent_work_complete",
                      "duration"]
        dateCols   = ["start", "finish", "early_start", "early_finish", "late_start", "late_finish"]
        
        nameChanger = {
            "wbs":"WBS",
            "Type":"Type",
            "WBS Name":"WBS Name",
            "unique_id":"Activity ID",
            "name":"Activity Name",
            "duration":"Original Duration",
            "Float":"Float",
            "start":"Start",
            "finish":"Finish",
            "early_start":"Early Start",
            "early_finish":"Early Finish",
            "late_start":"Late Start",
            "late_finish":"Late Finish",
            "BL_Project Start":"BL_Project Start",
            "BL_Project Finish":"BL_Project Finish",
            "percent_work_complete":"Activity Percentage Complete"
        }
                        
        def secDayConverter(sec):
            day = (sec/3600)/8
            return day
        
        def strDateConverter(string):
            dateFormat = "%Y-%m-%dT%H:%M:%S.%f"
            date = datetime.datetime.strptime(string, dateFormat)
            return date
        
        def dayComputer(date):
            days = date.days
            return days
        
        for file in taskFiles:
            loc = file.split("tasks.csv")[0].split("/")[-1]
            print("Processing: ",loc)
            
            df = pd.read_csv(file)[targetCols]
            
            df["duration"] = df["duration"].apply(secDayConverter)
            df["actual_duration"] = df["actual_duration"].apply(secDayConverter)
            for col in dateCols:
                df[col] = df[col].apply(strDateConverter)
            df["Float"] = (df["late_finish"] - df["early_finish"])
            df["Float"] = df["Float"].apply(dayComputer)
            df["BL_Project Start"]  = ""
            df["BL_Project Finish"] = ""
            
            df["wbsLength"] = df["wbs"].apply(len)
            df["Type"] = "WBS"
            df["WBS Name"] = df["name"]
            
            for i in range(1, len(df)-1):
                if df["wbsLength"].iloc[i] == df["wbsLength"].iloc[i-1]:
                    df["Type"].iloc[i] = df["Type"].iloc[i-1]

                elif df["wbsLength"].iloc[i] > df["wbsLength"].iloc[i-1] and\
                df["wbsLength"].iloc[i] < df["wbsLength"].iloc[i+1]:
                    df["Type"].iloc[i] = "WBS"

                elif df["wbsLength"].iloc[i] > df["wbsLength"].iloc[i-1] and\
                df["wbsLength"].iloc[i] >= df["wbsLength"].iloc[i+1]:
                    df["Type"].iloc[i] = "Activity"
                else:
                    pass

            if df["wbsLength"].iloc[len(df)-1] == df["wbsLength"].iloc[len(df)-2]:
                df["Type"].iloc[len(df)-1] = df["Type"].iloc[len(df)-2]
            else:
                df["Type"].iloc[len(df)-1] = "Activity"
            
            newtargetCols = [key for key, val in nameChanger.items()]
            df = df[newtargetCols]
            df = df.rename(columns=nameChanger)
            
            fileName = loc + ".csv"
            filePath = os.path.join(outputFolderDIR, fileName)
            df.to_csv(filePath, encoding="utf-8-sig")
        
        for path in csvFilePaths:
            os.remove(path)
            
    def checkMissingValues(self, columns, rowsList):
        for row in rowsList:
            if len(columns) > len(row):
                row[len(columns):len(row)] = [None]*(len(columns) - len(row))
        return rowsList
    
    def XERtoCSV(self, inputFolderDIR, outputFolderDIR):
        inputPath = os.path.join(inputFolderDIR, "*.xer")
        filesXER  = glob.glob(inputPath)
        
        nameChanger = {
            "WBS":"WBS",
            "Type":"Type",
            "wbs_name":"WBS Name",
            "task_code":"Activity ID",
            "task_name":" Activity Name",
            "Original Duration":"Original Duration",
            "At Completion Duration":"Float",
            "act_start_date":"Start",
            "act_end_date":"Finish",
            "early_start_date":"Early Start",
            "early_end_date":"Early Finish",
            "late_start_date":"Late Start",
            "late_end_date":"Late End",
            "target_start_date":"BL_Project Start",
            "target_end_date":"BL_Project Finish",
            "phys_complete_pct":"Activity Percentage Complete"
        }
        
        for file in filesXER:
            csvFilePaths = []
            loc = file.split(".xer")[0].split("/")[-1]
            print("Processing: ", loc)
            
            with open(file, encoding=self.encoding, errors="ignore") as f:
                content = f.read()
                tables  = content.split("%T")
            for table in tables[1:]:
                tableName = table.split()[0]
                fields    = table.split(r"%F")[1].split("\n")[0].split()

                rows = table.split("%R")[1:]
                rowsList = [r.strip().split("\t") for r in rows]
                checkedRowsList = self.checkMissingValues(fields, rowsList)

                df = pd.DataFrame(checkedRowsList, columns=fields, index=None)

                path = file.split(".xer")[0] + tableName + ".csv"
                csvFilePaths.append(path)
                if self.lang == "ko":
                    df.to_csv(path, encoding="utf-8-sig")
                else:
                    df.to_csv(path, encoding=self.encoding)
            
            ## TASK.csv
            inputPath = os.path.join(inputFolderDIR, "*TASK.csv")
            fileTASK  = glob.glob(inputPath)[0]
            
            df_task = pd.read_csv(fileTASK)
            targetCols = ["wbs_id", "task_code", "task_name", "target_drtn_hr_cnt",
                          "total_float_hr_cnt", "act_start_date", "act_end_date",
                          "early_start_date", "early_end_date",
                          "late_start_date", "late_end_date",
                          "target_start_date", "target_end_date", "phys_complete_pct",
                          "clndr_id"]
            df_task = df_task[targetCols]

            ## PROJWBS.csv
            inputPath = os.path.join(inputFolderDIR, "*PROJWBS.csv")
            fileWBS   = glob.glob(inputPath)[0]
            
            df_wbsName = pd.read_csv(fileWBS)
            targetCols = ["wbs_id", "wbs_name", "parent_wbs_id"]
            df_wbsName = df_wbsName[targetCols]
            
            
            dic_wbs = {}
            for i in reversed(range(len(df_wbsName))):
                child  = str(df_wbsName["wbs_id"].iloc[i])
                parent = str(df_wbsName["parent_wbs_id"].iloc[i])
                dic_wbs[child] = parent

            def findRoot(child, roots, dic_wbs=dic_wbs):
                if child in dic_wbs.keys():
                    roots.append(dic_wbs[child])
                    findRoot(dic_wbs[child], roots)
                else:
                    return
                
            tree_wbs = {}
            for key, val in dic_wbs.items():
                global roots
                roots = []
                findRoot(key, roots)
                tree_wbs[key] = ".".join(reversed(roots))

            ## CALENDAR.csv
            inputPath = os.path.join(inputFolderDIR, "*CALENDAR.csv")
            fileDATE  = glob.glob(inputPath)[0]
            
            df_date = pd.read_csv(fileDATE)
            targetCols = ["clndr_id", "day_hr_cnt", "week_hr_cnt", "month_hr_cnt", "year_hr_cnt"]
            df_date = df_date[targetCols]

            ## Merge
            df = pd.merge(df_task, df_wbsName, how="left", left_on="wbs_id", right_on="wbs_id")
            df = pd.merge(df, df_date, how="right", left_on="clndr_id", right_on="clndr_id")
            
            ## Calculate Duration
            df["Original Duration"] = df["target_drtn_hr_cnt"]/df["day_hr_cnt"]
            df["At Completion Duration"] = df["total_float_hr_cnt"]/df["day_hr_cnt"]
            
            ## WBS
            df["WBS"]  = ""
            df["Type"] = 0
            df = df.dropna(subset=['wbs_id'])
            for i in range(len(df)):
                code = str(int(df["wbs_id"].iloc[i]))
                if code not in tree_wbs.keys():
                    df["WBS"].iloc[i] = code
                    df["Type"].iloc[i] = len(code)
                else:
                    df["WBS"].iloc[i] = tree_wbs[code]
                    df["Type"].iloc[i] = len(tree_wbs[code])
                    
            TypeLength = list(set(sorted(list(df["Type"]), reverse=True)))
            compLength = []
            idx   = 0
            wbses = []
            df["TempType"] = ""
            
            while compLength != TypeLength:
                num = TypeLength[idx]
                for row in range(len(df)):
                    if df["Type"].iloc[row] == num and\
                    df["WBS"].iloc[row] not in wbses:
                        df["TempType"].iloc[row] = "Activity"
                        wbses.append(".".join(df["WBS"].iloc[row].split(".")[:-1]))
                        wbses = list(set(wbses))
                    else:
                        df["TempType"].iloc[row] = "WBS"
                compLength.append(TypeLength[idx])
                idx += 1
            df["Type"] = df["TempType"]
            
            targetCols = [key for key, val in nameChanger.items()]
            df = df[targetCols].dropna(how="all")
            df = df.rename(columns=nameChanger)
            
            csvFilePath = os.path.join(outputFolderDIR, loc + "_forTableau.csv")
            if self.lang == "ko":
                df.to_csv(csvFilePath, encoding="utf-8-sig")
            else:
                df.to_csv(csvFilePath, encoding=self.encoding)
        
            for path in csvFilePaths:
                os.remove(path)
        
        return df