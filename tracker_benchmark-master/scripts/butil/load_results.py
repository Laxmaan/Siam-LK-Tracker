import os
import sys
import json

from config import *
from scripts import *

def save_seq_result(result):
    tracker = result[0].tracker
    seqName = result[0].seqName
    evalType = result[0].evalType
    src = os.path.join(RESULT_SRC.format(evalType),tracker)
    if not os.path.exists(src):
        os.makedirs(src)
    fileName =os.path.join( src , '{0}.json'.format(seqName))
    print(fileName)
    try:
        with open(fileName,'w') as f:
            string = json.dumps(result, default=lambda o : o.__dict__)
            f.write(string)
    except Exception as e:
        print(f"result is {[x for x in result]}")
        print(e,end="\n\n")
        print(list(map(type, list(result[0].__dict__.values()))))
        sys.exit()


def save_scores(scoreList, testname=None):
    tracker = scoreList[0].tracker
    evalType = scoreList[0].evalType
    trkSrc = os.path.join(RESULT_SRC.format(evalType) , tracker)
    print("tasksrc = "+trkSrc)
    if testname == None:
        scoreSrc =os.path.join( trkSrc, 'scores')
    else:
        scoreSrc = os.path.join( trkSrc, 'scores_{0}'.format(testname.strip()))
    if not os.path.exists(scoreSrc):
        os.makedirs(scoreSrc)
    for score in scoreList:
        string = json.dumps(score, default=lambda o : o.__dict__)
        fileName = os.path.join(scoreSrc,'{0}.json'.format(score.name))
        scoreFile = open(fileName, 'w')
        scoreFile.write(string)
        scoreFile.close()

def load_all_results(evalType):
    resultSRC = RESULT_SRC.format(evalType)
    trackers = os.listdir(resultSRC)
    resultList = dict()
    for tracker in trackers:
        results, attrs = load_result(evalType, tracker)
        resultList[tracker] = (results, attrs)

    return resultList

def load_result(evalType, tracker):
    resultSRC = RESULT_SRC.format(evalType)
    print('Loading \'{0}\'...'.format(tracker), end=' ')
    src = os.path.join(resultSRC, tracker)
    resultNames = os.listdir(src)
    attrs = []
    results = []
    for name in resultNames:
        if name == 'attributes':
            attrSrc = os.path.join(src, name)
            attrNames = os.listdir(attrSrc)
            for attrName in attrNames:
                attrFile = open(os.path.join(attrSrc, attrName))
                string = attrFile.read()
                j = json.loads(string)
                attr = Attribute(**j)
                attr.successRateList = [o*100 for o in attr.successRateList]
                attrs.append(attr)
                attrs.sort()
        elif name.endswith('.json'):
            resultFile = open(os.path.join(src, name))
            string = resultFile.read()
            jsonList = json.loads(string)
            if type(jsonList) is list:
                results.append([Result(**j) for j in jsonList])
            elif type(jsonList) is dict:
                results.append([Result(**jsonList)])
    print('({0} seqs)'.format(len(resultNames) - 1))
    return results, attrs

def load_seq_result(evalType, tracker, sequence):
    resultSRC = RESULT_SRC.format(evalType)
    print('Loading {0}/{1}...'.format(tracker, sequence))
    src = os.path.join(resultSRC, tracker)
    result_src = os.path.join(src, sequence+'.json')
    print(result_src)
    resultFile = open(result_src)
    jsonList = json.load(resultFile)
    if type(jsonList) is list:
        return [Result(**j) for j in jsonList]
    elif type(jsonList) is dict:
        return [Result(**jsonList)]
    return None

def load_all_scores(evalType, testname):
    resultSRC = RESULT_SRC.format(evalType)
    trackers = os.listdir(resultSRC)
    attrList = [(t, load_scores(evalType, t, testname)) for t in trackers]
    return attrList

def load_scores(evalType, tracker, testname):
    resultSRC = RESULT_SRC.format(evalType)
    print('Loading \'{0}\'...'.format(tracker))
    src = os.path.join(resultSRC,os.path.join( tracker,'scores_{0}'.format(testname)))
    attrNames = os.listdir(src)
    attrs = []
    for attrName in attrNames:
        attrFile = open(os.path.join(src, attrName))
        string = attrFile.read()
        j = json.loads(string)
        attr = Score(**j)
        attr.successRateList = [o*100 for o in attr.successRateList]
        attrs.append(attr)
        attrs.sort()
    return attrs
