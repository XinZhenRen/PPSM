import os
import shutil
def creatDir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path+'right')
    else:
        print(path+'no')

def copyFile(filePath,newPath):
    fileNames=os.listdir(filePath)
    for file in fileNames:
        newDir=filePath+'/'+file
        if os.path.isfile(newDir):
            print(newDir)
            newFile=newPath+file
            shutil.copyfile(newDir,newFile)
        else:
            copyFile(newDir,newPath)

if __name__=="__main__":
    #path="/home/xinzhen/VPS-main/data/SUN-SEG/TrainDataset/Frame"
    #mkPath="/home/xinzhen/PPSN-main/SUN/Frame/"
    path="/home/xinzhen/VPS-main/data/SUN-SEG/TrainDataset/GT"
    mkPath="/home/xinzhen/PPSN-main/SUN/GT/"
    creatDir(mkPath)
    copyFile(path,mkPath)