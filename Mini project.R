#Calling the libraries 
library(tidyverse)
library(caret)
library(corrplot)

#Upload the dataset
data = read_csv("C:/Users/Lenovo/Desktop/diabetes.csv")

#Checking for correlation between variables
correlationMatrix= cor(data[,2:16])
correlationMatrix
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
highlyCorrelated
corrplot::corrplot(correlationMatrix)
corrplot::corrplot(correlationMatrix, type = "lower", method = "pie")
ggplot(data, aes(age)) + geom_density(aes(fill=outcome), alpha=1/3)

#Data Preprocessing
sum(is.na(data))
set.seed(100)
trainRowNumbers = createDataPartition(data$outcome, p=0.8, list=FALSE)
trainData = data[trainRowNumbers,]
testData = data[-trainRowNumbers,]
x= trainData[,2:16]
y = trainData$outcome
dummies_model = dummyVars(outcome ~ ., data=trainData)
trainData_mat =predict(dummies_model, newdata = trainData)
trainData = data.frame(trainData_mat)
str(trainData)
preProcess_range_modeltr=preProcess(trainData,method= 'range') #normalize using preprocess method
trainData = predict(preProcess_range_modeltr, newdata = trainData)
trainData$output = y
apply(trainData[, 2:16], 2,FUN=function(x){c('min'=min(x), 'max'=max(x))})
levels(testData$outcome)=c("Class0","Class1")
testData2 <- predict(preProcess_range_modeltr, testData) 
testData3 <- predict(dummies_model, testData2)
testData4 <- predict(preProcess_range_modeltr, testData3)
head(testData4[, 2:15])
data$outcome=as.factor(data$outcome) 
summary(data) 
preProcess_range_modeltr=preProcess(data,method= 'range') 
trainData=predict(preProcess_range_modeltr, newdata=data)
levels(trainData$outcome)=c("Class0","Class1") 
head(trainData)
summary(trainData)
featurePlot(x = trainData[, 2:15], 
            y = trainData$outcome, 
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))
featurePlot(x = trainData[, 2:15], 
            y = trainData$outcome, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))
#Model training and testing
fitControl=trainControl(method = 'cv', #k-fold cross validation
                        number = 5, #number of folds
                        savePredictions = 'final', #saves predictions for optional tuning parameter
                        classProbs = T, #class probabilities are returned
                        summaryFunction = twoClassSummary) #results summary function
model1= train(outcome ~ . ,data=trainData, method='knn', tuneLength=2,trControl=fitControl)
model2= train(outcome ~ . ,data=trainData, method='svmRadial', tuneLength=2,trControl=fitControl)
model3= train(outcome ~ . ,data=trainData, method='rpart', tuneLength=2,trControl=fitControl)
models_compare= resamples(list(KNN=model1,SVM=model2,RandomForest=model3))
z= varImp(model3)
print(z)
ggplot(z)
models_compare= resamples(list(KNN=model1,SVM=model2,RandomForest=model3))
summary(models_compare)
scales=list(x=list(relation='free'),y=list(relation='free'))
bwplot(models_compare, scales=scales)
library(caretEnsemble)
trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             classProbs=TRUE)

algorithmList <- c('knn', 'svmRadial', 'rpart')
set.seed(100)
models <- caretList(outcome ~ ., data=trainData, trControl=trainControl, methodList=algorithmList) 
results <- resamples(models)
summary(results)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)
set.seed(123)
stackControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             classProbs=TRUE)
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)
stack_predicteds <- predict(stack.glm, newdata=testData)
head(stack_predicteds)

