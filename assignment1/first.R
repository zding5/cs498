# system("wget http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data")
#reading data
library(devtools)
source_url("https://raw.githubusercontent.com/ggrothendieck/gsubfn/master/R/list.R")
d <- read.csv('./pima-indians-diabetes.data', sep=",", header=T)
dx <- d[, -c(9)]
dy <- d[, 9]

library('caret')
# preparing data with 80% traing and 20% testing
inTrain <- createDataPartition(dy, p=0.8, list=F)

trainData = d[inTrain,]
testData = d[-inTrain,]

trainX <- trainData[,-c(9)]
trainY <- trainData[,9]
testX <- testData[,-c(9)]
testY <- testData[,9]

# using gaussian distribution so that we know the best

trainGaussian <- function(trainX, trainY){
    posFilter <- trainY == 1
    trainMuP <- sapply(trainX[posFilter,], mean, na.rm=T)
    trainMuN <- sapply(trainX[!posFilter,], mean, na.rm=T)
    trainSigmaP <- sapply(trainX[posFilter,], sd, na.rm=T)
    trainSigmaN <- sapply(trainX[!posFilter,], sd, na.rm=T)
    return(list(trainMuP, trainMuN, trainSigmaP, trainSigmaN))
}

perdictGaussian <- function(x, muP, muN, sigmaP, sigmaN){
    offsetP <- t(t(x)-muP) #x-mu
    offsetN <- t(t(x)-muN)
    scaleP <- t(t(offsetP) / sigmaP) # (x-mu)/sigma
    scaleN <- t(t(offsetN) / sigmaN)
    logP <- -0.5*rowSums(apply(scaleP, c(1,2), square), na.rm=T)-sum(log(sigmaP))
    # (x-mu)^2/2sigma^2-log(simga)
    logN <- -0.5*rowSums(apply(scaleN, c(1,2), square), na.rm=T)-sum(log(sigmaN))
    return(logP>logN)
}

correctPercent <- function(x, y) sum(x==y)/length(x)

correctness = array(dim=10)

for (wi in 1:10){
    inTrain <- createDataPartition(dy, p=0.8, list=F)

    trainData = d[inTrain,]
    testData = d[-inTrain,]
    trainX = trainData[,-c(9)]
    trainY = trainData[,9]
    testX = testData[,-c(9)]
    testY = testData[,9]

    list[meanP, meanN, sdP, sdN] = trainGaussian(trainX, trainY)
    correctness = correctPercent(perdictGaussian(testX, meanP, meanN, sdP, sdN), testY)
}

