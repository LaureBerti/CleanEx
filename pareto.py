import os
import numpy as np
from scipy import spatial
from functools import reduce

from skcriteria import Data, MIN, MAX
from skcriteria.madm import closeness, simple


def filter_(pts, pt):
    """
    Get all points in pts that are not Pareto dominated by the point pt
    """
    weakly_worse = (pts <= pt).all(axis=-1)
    strictly_worse = (pts < pt).any(axis=-1)
    return pts[~(weakly_worse & strictly_worse)]


def get_pareto_undominated_by(pts1, pts2=None):
    """
    Return all points in pts1 that are not Pareto dominated
    by any points in pts2
    """
    if pts2 is None:
        pts2 = pts1
    return reduce(filter_, pts2, pts1)


def get_pareto_frontier(pts):
    """
    Iteratively filter points based on the convex hull heuristic
    """
    pareto_groups = []

    # loop while there are points remaining
    while pts.shape[0]:
        # brute force if there are few points:
        if pts.shape[0] < 10:
            pareto_groups.append(get_pareto_undominated_by(pts))
            break

        # compute vertices of the convex hull
        hull_vertices = spatial.ConvexHull(pts).vertices

        # get corresponding points
        hull_pts = pts[hull_vertices]

        # get points in pts that are not convex hull vertices
        nonhull_mask = np.ones(pts.shape[0], dtype=bool)
        nonhull_mask[hull_vertices] = False
        pts = pts[nonhull_mask]

        # get points in the convex hull that are on the Pareto frontier
        pareto = get_pareto_undominated_by(hull_pts)
        pareto_groups.append(pareto)

        # filter remaining points to keep those not dominated by
        # Pareto points of the convex hull
        pts = get_pareto_undominated_by(pts, pareto)

    return np.vstack(pareto_groups)


def removeAttributeId(inputData, idAttributesToRemove):
    return [inputData[i] for i in range(len(inputData))
            if i not in idAttributesToRemove]


def getBestRankedCandidate(nbToExtract, candidateSet, rulesSetsQuality, criteriaQuality):
    # print(str(candidateSet))
    # print(str(rulesSetsCandidates))

    tmpRulesSetsQuality = None

    criteria = []
    idAttributesToRemove = []
    for idCrit in range(len(criteriaQuality)):
        if criteriaQuality[idCrit] == '+':
            criteria.append(MAX)
        elif criteriaQuality[idCrit] == '-':
            criteria.append(MIN)
        else:
            idAttributesToRemove.append(idCrit)

    attributes = removeAttributeId(['Polarity', 'Diversity', 'Distancing', 'Surprise'],
                                   idAttributesToRemove)
    if len(attributes) < 4:
        tmpRulesSetsQuality = rulesSetsQuality
        rulesSetsQuality = []
        for ruleQI in tmpRulesSetsQuality:
            newQI = removeAttributeId(ruleQI, idAttributesToRemove)
            rulesSetsQuality.append(newQI)

    print(str(attributes))
    candidateIds = [i for i in range(len(candidateSet))]
    data = Data(rulesSetsQuality,
                criteria,
                anames=candidateIds,
                cnames=attributes)

    # apply a simple weighted sums method
    dm = simple.WeightedSum()
    res = dm.decide(data)
    current_ranking = res.rank_

    # extract bests candidates
    if nbToExtract > len(current_ranking):
        nbToExtract = len(current_ranking)

    bestCandidates = []
    for i in range(nbToExtract):
        bestCandidateId = list(current_ranking).index(i+1)
        bestCandidate = candidateSet[bestCandidateId]

        if tmpRulesSetsQuality is None:
            bestCandidateQI = rulesSetsQuality[bestCandidateId]
        else:
            bestCandidateQI = tmpRulesSetsQuality[bestCandidateId]

        bestCandidates.append((bestCandidate, bestCandidateQI))

    return bestCandidates


def generateExpeRankFiles(isMain, outputFile, runId,
                          finalNode, criteriaQuality, nbRules,
                          rank, nbNodes, maxTreeDepth, qualityMetrics,
                          refNode=None):

    rn = ","
    if refNode is not None:
        rn += refNode

    if runId is None:
        runId = 0

    mooString = ""
    if criteriaQuality is not None:
        for criteria in criteriaQuality:
            mooString += criteria + ";"
        mooString = mooString[:-1]

    if isMain:
        # main branch
        if os.path.exists(outputFile + "_rankMain.csv"):
            f = open(outputFile + "_rankMain.csv", 'a')
        else:
            f = open(outputFile + "_rankMain.csv", 'w')
            print("runID,pickID,currentID,taille_arbre,profondeur_max,"
                  + "MOO,NbRule,RuleRanking,"
                  + "polarity,distancing,surprise,diversity", file=f)
    else:
        # other branchs
        if os.path.exists(outputFile + "_rankOthers.csv"):
            f = open(outputFile + "_rankOthers.csv", 'a')
        else:
            f = open(outputFile + "_rankOthers.csv", 'w')
            print("runID,pickID,currentID,taille_arbre,profondeur_max,"
                  + "MOO,NbRule,RuleRanking,"
                  + "polarity,distancing,surprise,diversity", file=f)

    print(str(runId) + "," + finalNode + rn + "," + str(nbNodes) + "," + str(maxTreeDepth) +
          "," + mooString + "," + str(nbRules) + "," + str(rank) +
          "," + str(qualityMetrics[0]) +
          "," + str(qualityMetrics[1]) +
          "," + str(qualityMetrics[2]) +
          "," + str(qualityMetrics[3]), file=f)
    f.close()


def getBestRankedCandidateExpe(isMain, outputFile,
                               runId, finalNode,
                               nbToExtract,
                               candidateSet, rulesSetsQuality,
                               criteriaQuality,
                               nbNodes, maxTreeDepth, refNode=None):
    # print(str(candidateSet))
    # print(str(rulesSetsCandidates))

    tmpRulesSetsQuality = None

    criteria = []
    idAttributesToRemove = []
    for idCrit in range(len(criteriaQuality)):
        if criteriaQuality[idCrit] == '+':
            criteria.append(MAX)
        elif criteriaQuality[idCrit] == '-':
            criteria.append(MIN)
        else:
            idAttributesToRemove.append(idCrit)

    attributes = removeAttributeId(['Polarity', 'Diversity', 'Distancing', 'Surprise'],
                                   idAttributesToRemove)
    if len(attributes) < 4:
        tmpRulesSetsQuality = rulesSetsQuality
        rulesSetsQuality = []
        for ruleQI in tmpRulesSetsQuality:
            newQI = removeAttributeId(ruleQI, idAttributesToRemove)
            rulesSetsQuality.append(newQI)

    print(str(attributes))
    candidateIds = [i for i in range(len(candidateSet))]
    data = Data(rulesSetsQuality,
                criteria,
                anames=candidateIds,
                cnames=attributes)

    # apply a simple weighted sums method
    dm = simple.WeightedSum()
    res = dm.decide(data)
    current_ranking = res.rank_

    # extract bests candidates
    if nbToExtract > len(current_ranking):
        nbToExtract = len(current_ranking)

    bestCandidates = []
    for i in range(500):

        bestCandidateId = list(current_ranking).index(i+1)
        bestCandidate = candidateSet[bestCandidateId]

        if tmpRulesSetsQuality is None:
            bestCandidateQI = rulesSetsQuality[bestCandidateId]
        else:
            bestCandidateQI = tmpRulesSetsQuality[bestCandidateId]

        generateExpeRankFiles(isMain, outputFile, runId,
                              finalNode, criteriaQuality,
                              len(bestCandidate),
                              i+1,
                              nbNodes, maxTreeDepth,
                              bestCandidateQI,
                              refNode=refNode)

        if i < nbToExtract:
            bestCandidates.append((bestCandidate, bestCandidateQI))

    return bestCandidates
