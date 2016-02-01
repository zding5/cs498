system("wget http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data")
#reading data
d <- read.csv('./pima-indians-diabetes.data', sep=",")
d <- data.matrix(d)

library('caret')
# preparing data with 80% traing and 20% testing
inTrain <- createDataPartition(d[,1], p=0.8, list=F)

trainData = d[inTrain,]
testData = d[-inTrain,]

trainX <- trainData[,-9]
trainY <- trainData[,9]
testX <- testData[,-9]
testY <- testData[,9]

posFilter <- testY==1
# using gaussian distribution so that we know the best
trainMuP <- apply(trainX[posFilter,], 2, mean)
trainMuN <- apply(trainX[!posFilter,], 2, mean)
trainSigmaP <- apply(trainX[posFilter,], 2, sd)
trainSigmaN <- apply(trainX[!posFilter,], 2, sd)

# first problem
pos <- array(0,length(testY))
neg <- array(0,length(testY))

for (i in 1:8)
    pos = pos+dnorm(testX[,i], mean=trainMuP[i], sd=trainSigmaP[i], log=T)
    neg = neg+dnorm(testX[,i], mean=trainMuN[i], sd=trainSigmaN[i], log=T)
end

predictY <- array(0, length(testY))
predictY[pos>neg] <- 1

for (i in c(3,6,4,8))
    trainX[trainX[,i]==0,i] = NA
end

meanna <- function(x){mean(x,na.rm=T)}
sdna <- function(x){sd(x,na.rm=T)}
# second problem
trainMuP <- apply(trainX[posFilter,], 2, meanna)
trainMuN <- apply(trainX[!posFilter,], 2, meanna)
trainSigmaP <- apply(trainX[posFilter,], 2, sdna)
trainSigmaN <- apply(trainX[!posFilter,], 2, sdna)

pos <- array(0, length(testY))
neg <- array(0, length(testY))

for (i in 1:8)
    pos = pos+dnorm(testX[,i], mean=trainMuP[i], sd=trainSigmaP[i], log=T)
    neg = neg+dnorm(testX[,i], mean=trainMuN[i], sd=trainSigmaN[i], log=T)
end


model = train(d[,-9], d[,9]==1, 'nb', metric="Accuracy")
