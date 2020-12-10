#!/usr/bin/env python

import os.path
import pandas as pd
import numpy as np
import sys
import argparse
import utility
import math
import itertools
import random
from datetime import datetime
from pareto import *
from radarPlot import *
import os.path
endNodes = set()
nodesInPathToFinalNode = {}
finalNodeToRules = {}
currentIdRule = 0
rules = {}
nodesPairs = set()
maxTreeDepth = 0

deltaRuleToDeltaValue = {}
mlRules = set()
ruleToNbFeatures = {}
predicateToNodes = {}

outputFile = ""
runId = ""


def addPredicateForDiversity(predicate, ruleId):
    if predicate in predicateToNodes:
        predicateToNodes[predicate].add(ruleId)
    else:
        predicateToNodes[predicate] = set()
        predicateToNodes[predicate].add(ruleId)


def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item


def sort_and_deduplicate(l):
    return list(uniq(sorted(l, reverse=True)))


def addNewRule(predicate, node, prefix, rule, nbFeatures=0, delta=None, isML=False):
    global endNodes
    global currentIdRule
    global rules
    global finalNodeToRules
    global ruleToNbFeatures

    for finalNode in nodesInPathToFinalNode[node]:
        if finalNode in finalNodeToRules:
            finalNodeToRules[finalNode].append(currentIdRule)
        else:
            finalNodeToRules[finalNode] = [currentIdRule]
        rules[currentIdRule] = prefix + str(currentIdRule) + ": " + rule

        ruleToNbFeatures[currentIdRule] = nbFeatures

        if delta is not None:
            deltaRuleToDeltaValue[currentIdRule] = delta
        if isML:
            mlRules.add(currentIdRule)

        addPredicateForDiversity(predicate, currentIdRule)

        currentIdRule += 1


def addToNodeToFinal(node, correspondingFinalNode):
    if node in nodesInPathToFinalNode:
        nodesInPathToFinalNode[node].append(correspondingFinalNode)
    else:
        nodesInPathToFinalNode[node] = [correspondingFinalNode]


def generateTreeSucc(baseNode, dfTreeStruct, currentProf):
    global maxTreeDepth
    pathOut = []
    succOut = []

    if currentProf > maxTreeDepth:
        maxTreeDepth = currentProf

    for index, row in dfTreeStruct.iterrows():
        if row[0] == baseNode:
            (pathList, succList) = generateTreeSucc(
                row[1], dfTreeStruct, currentProf + 1)
            if len(pathList) == 0:  # base case
                pathOut.append([row[1]])
                succOut.append(
                    [row[1], "succ(" + row[0] + "," + str(row[1]) + ")"])
                nodesPairs.add((row[0], row[1]))
                endNodes.add(row[1])
                nodesInPathToFinalNode[row[1]] = [row[1]]
            else:  # recursive case
                for path in pathList:
                    pathOut.append([row[1]] + path)
                for succ in succList:
                    addToNodeToFinal(row[1], succ[0])
                    succOut.append(
                        [succ[0], "succ(" + row[0] + "," + row[1] + ") /\ " + succ[1]])
                    nodesPairs.add((row[0], row[1]))
    return (pathOut, succOut)


def generateTreePaths(baseNode, dfTreeStruct):
    (pathList, succList) = generateTreeSucc(baseNode, dfTreeStruct, 0)
    print("depth: " + str(maxTreeDepth))
    # create the tree
    for succ in succList:
        addNewRule('succ', succ[0], "P", succ[1])

    pathOut = []
    for path in pathList:
        pathTmp = [baseNode]
        pathTmp = pathTmp + path
        pathOut.append(pathTmp)

    # add the root node
    for finalNode in list(endNodes):
        addToNodeToFinal(baseNode, finalNode)

    return (pathOut, succList)


def generateFeatureChange(parentNode, childNode, dfTreeFeatures):
    rowParentNode = dfTreeFeatures[dfTreeFeatures.node == parentNode]
    rowChildNode = dfTreeFeatures[dfTreeFeatures.node == childNode]

    nbFeatures = 0
    allStable = True
    nbStableTmp = 0
    explList = []

    for feature in dfTreeFeatures.columns:
        if(feature != 'node'):
            nbFeatures += 1
            valParent = rowParentNode.iloc[0][feature]
            valChild = rowChildNode.iloc[0][feature]
            diff = np.round(valChild-valParent, decimals=3)
            if(diff == 0.0):
                nbStableTmp += 1
                explList.append(
                    ("S", "stable(" + feature + "," + parentNode + "," + childNode + ")", None, 'stable'))
            else:
                allStable = False
                if diff > 0.0:
                    explList.append(
                        ("C", "increase(" + feature
                         + "," + parentNode
                         + "," + childNode
                         + ") /\ delta(" + feature
                         + "," + parentNode
                         + "," + childNode
                         + "," + str(abs(diff)) + ")", abs(diff), 'increase'))
                else:
                    explList.append(
                        ("C", "decrease(" + feature
                         + "," + parentNode
                         + "," + childNode + ") /\ delta(" + feature
                         + "," + parentNode
                         + "," + childNode
                         + "," + str(abs(diff)) + ")", abs(diff), 'decrease'))

    if allStable:
        equivStr = "equiv(" + parentNode + "," + childNode + ")"
        addNewRule('equiv', childNode, "S", equivStr, nbFeatures=nbFeatures)
    else:
        for rule in explList:
            addNewRule(rule[3], childNode, rule[0], rule[1],
                       delta=rule[2], nbFeatures=1)


def generateTreeChanges(dfTreeStruct, dfTreeFeatures):
    # generate a list of each pair of vertices connected by one edge

    for edge in nodesPairs:
        generateFeatureChange(edge[0], edge[1], dfTreeFeatures)


def generateFeaturesRule(predicate, featuresSet, node):
    featuresString = "["
    for feature in featuresSet:
        featuresString += str(feature) + ","
    featuresString = featuresString[:-1] + "]"

    if predicate == "most" or predicate == "least":
        addNewRule(predicate, node, "C", predicate +
                   "(" + featuresString + "," + str(node) + ")",
                   nbFeatures=len(featuresSet), isML=True)
    else:
        addNewRule(predicate, node, "C", predicate +
                   "(" + featuresString + "," + str(node) + ")",
                   nbFeatures=len(featuresSet))


def generateFeaturesRuleFromDictionnary(featuresDictionnary, node):
    for key in featuresDictionnary.keys():
        featuresList = featuresDictionnary[key]
        featuresString = "["
        for feature in featuresList:
            featuresString += str(feature) + ","
        featuresString = featuresString[:-1] + "]"

        addNewRule(str(key[0]), node, "C",
                   str(key[0]) + "(" + featuresString +
                   "," + str(node) + "," + str(key[1]) + ")",
                   nbFeatures=len(featuresList))


def generateFeatureComparison(compNodes, dfTreeFeatures):
    for nodeRef in compNodes:
        rowRef = dfTreeFeatures[dfTreeFeatures.node == nodeRef]

        # store each result to allows groupping in the generated rules
        asSet = set()
        mostSet = set()
        leastSet = set()
        otherCompSet = {}
        for feature in dfTreeFeatures.columns:
            valRef = rowRef.iloc[0][feature]

            if(feature != 'node'):

                # compared values
                resComp = {'more': 0, 'less': 0, 'as': 0}

                for nodeComp in compNodes:
                    if nodeComp != nodeRef:
                        rowComp = dfTreeFeatures[dfTreeFeatures.node == nodeComp]
                        valComp = rowComp.iloc[0][feature]
                        if valRef > valComp:
                            resComp['more'] += 1
                        elif valRef < valComp:
                            resComp['less'] += 1
                        else:
                            resComp['as'] += 1

                resComp = {key: val for key, val in sorted(
                    resComp.items(), key=lambda item: item[1])}

                bestPredicate = list(resComp)[2]
                bestValue = resComp[bestPredicate]

                secondPredicate = list(resComp)[1]
                secondValue = resComp[secondPredicate]

                thirdPredicate = list(resComp)[0]
                tot = bestValue + secondValue + resComp[thirdPredicate]

                if secondValue == 0:
                    if bestPredicate == 'as':
                        asSet.add(feature)
                    elif bestPredicate == 'more':
                        mostSet.add(feature)
                    else:
                        leastSet.add(feature)
                else:
                    bestRuleValue = np.round(bestValue/tot, decimals=3)
                    secondRuleValue = np.round(secondValue/tot, decimals=3)
                    dictKeyBest = (bestPredicate, bestRuleValue)
                    dictKeySecond = (secondPredicate, secondRuleValue)

                    if dictKeyBest in otherCompSet:
                        otherCompSet[dictKeyBest].append(feature)
                    else:
                        otherCompSet[dictKeyBest] = [feature]

                    if dictKeySecond in otherCompSet:
                        otherCompSet[dictKeySecond].append(feature)
                    else:
                        otherCompSet[dictKeySecond] = [feature]

        # geberate rules
        if asSet:
            generateFeaturesRule("as", asSet, nodeRef)
        if mostSet:
            generateFeaturesRule("most", mostSet, nodeRef)
        if leastSet:
            generateFeaturesRule("least", leastSet, nodeRef)

        generateFeaturesRuleFromDictionnary(otherCompSet, nodeRef)


def divSubRoutine(n, N):
    if n == 0:
        return 0
    return (n/N) * math.log2(n/N)


def generateRules(dfTreeStruct, dfTreeFeatures, rootNode, finalNode):
    (pathOut, succOut) = generateTreePaths(rootNode, dfTreeStruct)

    generateTreeChanges(dfTreeStruct, dfTreeFeatures)

    nodesToEval = set()
    for nodeList in pathOut:
        nodesToEval = nodesToEval.union(set(nodeList))
    generateFeatureComparison(nodesToEval, dfTreeFeatures)


def computeQualityMetrics(rulesIdsSet):

    outputMetrics = {}

    deltaInput = 0
    deltaTot = 0

    surpriseDelta = 0
    surpriseTot = 0

    nbMLInput = 0
    nbFeatInput = 0

    for ruleId in rules.keys():

        if ruleId in rulesIdsSet:  # rule in evaluated set
            # polarity
            nbFeatInput += ruleToNbFeatures[ruleId]
            if ruleId in mlRules:  # most/least rule in evaluated set
                nbMLInput += ruleToNbFeatures[ruleId]
        else:
            # surprise
            surpriseTot += ruleToNbFeatures[ruleId]
            if ruleId in deltaRuleToDeltaValue:
                surpriseDelta += deltaRuleToDeltaValue[ruleId]

        # distancing
        if ruleId in deltaRuleToDeltaValue:  # delta rule
            delta = deltaRuleToDeltaValue[ruleId]
            deltaTot += delta
            if ruleId in rulesIdsSet:  # delta rule is in evaluated set
                deltaInput += delta

    # diversity
    diversitySum = 0
    for predicateSymbol in predicateToNodes.keys():
        nbFeaturesPredicateInput = 0
        for ruleId in predicateToNodes[predicateSymbol]:
            if ruleId in rulesIdsSet:
                nbFeaturesPredicateInput += ruleToNbFeatures[ruleId]
        diversitySum += divSubRoutine(nbFeaturesPredicateInput, nbFeatInput)

    nbPredicateSymbols = len(predicateToNodes.keys())

    outputMetrics['polarity'] = computeDivision(nbMLInput, nbFeatInput)
    outputMetrics['distancing'] = computeDivision(deltaInput, deltaTot)
    outputMetrics['surprise'] = computeDivision(surpriseDelta, surpriseTot)
    outputMetrics['diversity'] = abs(computeDivision(diversitySum,
                                                     nbPredicateSymbols))

    return outputMetrics


def computeDivision(numerator, denominator):
    if denominator != 0:
        return numerator / denominator
    else:
        return 0


def plotQualityIndicators(nodeMainBranch, mainBranchQI, otherQI):
    dataForPlot = {}
    dataForPlot['group'] = ['Polarity', 'Diversity', 'Distancing', 'Surprise']

    if nodeMainBranch is not None:
        dataForPlot[nodeMainBranch] = mainBranchQI

    for [endNode, bestCandidate, bestCandidateQI] in otherQI:
        dataForPlot[endNode] = bestCandidateQI

    plotRadar(dataForPlot)


def findBestRulesSetForBranch(finalNode, initMinRuleNumber, initMaxRuleNumber, criteria, isExpe=False, isMain=False, refNode=None):

    assert(initMinRuleNumber <= initMaxRuleNumber)
    if refNode is None:
        refNode = finalNode

    relatedRulesIds = finalNodeToRules[finalNode]

    maxRuleNumber = len(relatedRulesIds)
    if maxRuleNumber > initMaxRuleNumber:
        maxRuleNumber = initMaxRuleNumber

    minRuleNumber = initMinRuleNumber
    if minRuleNumber > maxRuleNumber:
        minRuleNumber = maxRuleNumber

    candidateRulesSet = []
    rulesSetsQuality = []
    # for i in range(1, n+1):
    for i in range(minRuleNumber, maxRuleNumber+1):

        combinationsList = list(itertools.combinations(relatedRulesIds, i))
        for candidate in combinationsList:
            qualityMetrics = computeQualityMetrics(candidate)

            paretoQI = [qualityMetrics['polarity'],
                        qualityMetrics['diversity'],
                        qualityMetrics['distancing'],
                        qualityMetrics['surprise']]

            candidateRulesSet.append(candidate)
            rulesSetsQuality.append(paretoQI)

    if isExpe:
        bestCandidates = getBestRankedCandidateExpe(isMain,
                                                    outputFile,
                                                    runId,
                                                    finalNode,
                                                    1,
                                                    candidateRulesSet,
                                                    rulesSetsQuality,
                                                    criteria,
                                                    len(nodesInPathToFinalNode.keys())-1,
                                                    maxTreeDepth,
                                                    refNode=refNode)
    else:
        bestCandidates = getBestRankedCandidate(1,
                                                candidateRulesSet,
                                                rulesSetsQuality,
                                                criteria)

    return bestCandidates[0]


def addQuality(dictQualityToCandidate, paretoQI, candidate):
    if paretoQI in dictQualityToCandidate:
        dictQualityToCandidate[paretoQI].append(candidate)
    else:
        dictQualityToCandidate[paretoQI] = [candidate]


def prepareToPareto(rulesSetsCandidates):
    return np.asarray(
        list(rulesSetsCandidates), dtype=np.float32)


def log(logString):
    now = datetime.now()
    currentTime = now.strftime("%H:%M:%S")
    print(currentTime + ": " + logString)


def extractBestRulesSets(nbToExtract, bestRulesSets, criteria):

    out = []
    endNodesList = []
    candidateRulesSet = []
    rulesSetsQuality = []
    # for i in range(1, n+1):
    for [endNode, candidate, candidateQI] in bestRulesSets:

        endNodesList.append(endNode)
        candidateRulesSet.append(candidate)
        rulesSetsQuality.append(candidateQI)

    bestCandidates = getBestRankedCandidate(nbToExtract,
                                            candidateRulesSet,
                                            rulesSetsQuality,
                                            criteria)
    for (bestCandidate, bestCandidateQI) in bestCandidates:
        print(str(bestCandidate) + ": " + str(bestCandidateQI))
        node = endNodesList[candidateRulesSet.index(bestCandidate)]
        out.append([node, bestCandidate, bestCandidateQI])
    return out


def prepareCriteria(criteriaString):
    critList = criteriaString.split(",")
    out = []
    for criteria in critList:
        if criteria == '+' or criteria == '1':
            out.append('+')
        elif criteria == '-' or criteria == '-1':
            out.append('-')
        elif criteria == '0':
            out.append('0')
        else:
            raise ValueError('Value ' + criteria +
                             ' in the MOO vector cannnot be recognized.')
    return out


def generateCLIparser():
    parser = argparse.ArgumentParser(prog='ExpGen',
                                     description='Generate explanation for a cleaning tree')

    parser.add_argument('struct',
                        help='a path for the struct file')
    parser.add_argument('feat',
                        help='a path for the features file')
    parser.add_argument('root',
                        help='a name for the root node')
    parser.add_argument('-m',
                        '--moo',
                        action='store',
                        default='+,+,+,+',
                        help='vector indicating if QI should be maximized (+), minimized (-) or ingored (0)')
    parser.add_argument('-o',
                        '--out',
                        action='store',
                        help='a path for the output file')
    parser.add_argument('-f',
                        '--final',
                        action='store',
                        help='a name for a final node to specify a path')
    parser.add_argument('--expeRank',
                        action='store',
                        help='Avoid radar plot and generate files for rank experiments')
    parser.add_argument('--expeTime',
                        action='store',
                        help='Take run id. Avoid radar plot and generate files for time experiments')
    parser.add_argument('--minRules',
                        action='store',
                        default=5,
                        help='minimum number of rules  in the output')
    parser.add_argument('--maxRules',
                        action='store',
                        default=5,
                        help='maximum number of rules  in the output')
    parser.add_argument('--avoidNonFinal',
                        action='store_true',
                        help='avoid computation except for main branch')

    return parser


def generateOutputFile(outputFile, finalNode):
    f = None
    if finalNode is None:
        refNode = 'allNodes'
    else:
        refNode = finalNode

    if outputFile:
        f = open(outputFile + "_rules-" + refNode + ".csv", 'w')

    if finalNode is not None:
        for ruleId in finalNodeToRules[finalNode]:
            # print(str(rules[ruleId]))
            if outputFile:
                print(str(rules[ruleId]), file=f)
    else:
        for ruleId in rules.keys():
            # print(str(rules[ruleId]))
            if outputFile:
                print(str(rules[ruleId]), file=f)
    if f is not None:
        f.close()


def generateExpeTimeFiles(runId, outputFile, finalNode, moo, time, qualityMetrics):

    if runId is None:
        runId = 0

    mooString = ""
    if moo is not None:
        for criteria in moo:
            mooString += criteria + ";"
        mooString = mooString[:-1]

    # timee
    if os.path.exists(outputFile + "_time.csv"):
        f = open(outputFile + "_time.csv", 'a')
    else:
        f = open(outputFile + "_time.csv", 'w')
        print("runId, pickID, taille_arbre, profondeur_max, MOO, Exectime, polarity, distancing, surprise, diversity", file=f)
    print(str(runId) + "," + str(finalNode) + "," + str(len(nodesInPathToFinalNode.keys())) + "," + str(maxTreeDepth) +
          "," + mooString + "," + str(time) +
          "," + str(qualityMetrics[0]) +
          "," + str(qualityMetrics[1]) +
          "," + str(qualityMetrics[2]) +
          "," + str(qualityMetrics[3]), file=f)


beginTime = datetime.now()

parser = generateCLIparser()

# Execute parse_args()
args = parser.parse_args()

# load datasets and nodes
dfTreeStruct = utility.readFile(args.struct)
dfTreeFeatures = utility.readFile(args.feat)
rootNode = args.root
outputFile = args.out
finalNode = None
if args.final:
    finalNode = args.final
runId = args.expeTime

minRules = int(args.minRules)
maxRules = int(args.maxRules)
assert(minRules <= maxRules)

criteria = prepareCriteria(args.moo)
print("MOO: " + str(criteria))

# launch generation of the rules
log("Generation of the rules")
rulesOutputString = generateRules(
    dfTreeStruct, dfTreeFeatures, rootNode, finalNode)

log("Print output file with rules")
generateOutputFile(outputFile, finalNode)

if args.expeTime or args.expeRank:
    if finalNode is None:
        finalNode = random.choice(list(endNodes))
        print("Randomly chosen final node: " + finalNode)

# find best rules subset for main branch
if finalNode is not None:
    log("Search best rule set for main branch")
    (mainBranchCandidate, mainBranchQI) = findBestRulesSetForBranch(finalNode,
                                                                    minRules,
                                                                    maxRules,
                                                                    criteria,
                                                                    isExpe=(
                                                                        args.expeRank is not None),
                                                                    isMain=True)

# find best rules subset for other branchs
if not args.avoidNonFinal:
    log("Search best rule set for other branchs")
    bestRulesSets = []

    for endNode in endNodes:
        if endNode != finalNode:
            (bestCandidate, bestCandidateQI) = findBestRulesSetForBranch(endNode,
                                                                         minRules,
                                                                         maxRules,
                                                                         criteria,
                                                                         isExpe=(
                                                                             args.expeRank is not None),
                                                                         refNode=finalNode)

            bestRulesSets.append([endNode, bestCandidate, bestCandidateQI])

    # extract best rules sets
    print(str(bestRulesSets))
    bestRulesSet = extractBestRulesSets(3, bestRulesSets, criteria)

endTime = datetime.now()

if args.expeTime:
    generateExpeTimeFiles(args.expeTime, outputFile,
                          finalNode, criteria, endTime-beginTime, mainBranchQI)
elif not args.expeRank:
    plotQualityIndicators(finalNode, mainBranchQI, bestRulesSet)
